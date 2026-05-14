use std::collections::BTreeMap;
use std::ops::Deref;
use std::sync::Arc;

use agent_client_protocol as acp;
use forge_domain::{Agent, AgentId, McpHttpServer, McpOAuthSetting, McpServerConfig, ModelId, Scope, ServerName};
use futures::future::join_all;

use crate::services::{AgentRegistry, ProviderAuthService, ProviderService};
use crate::{McpConfigManager, McpService, Services};

use super::conversion;
use super::error::{Error, Result};

/// Maximum allowed length for an MCP server name (prevents injection).
const MAX_SERVER_NAME_LEN: usize = 128;

pub(super) struct StateBuilders;

impl StateBuilders {
    pub(super) async fn build_session_mode_state<S: Services + ?Sized>(
        services: &S,
        current_agent_id: &AgentId,
    ) -> Result<acp::SessionModeState> {
        let agents = services
            .agent_registry()
            .get_agents()
            .await
            .map_err(Error::Application)?;

        Ok(conversion::build_session_mode_state(
            &agents,
            current_agent_id,
        ))
    }

    pub(super) async fn build_session_model_state<S: Services>(
        services: &Arc<S>,
        current_agent: &Agent,
        session_model_override: Option<&ModelId>,
    ) -> Result<acp::SessionModelState> {
        // Fetch all configured providers and their models concurrently.
        let all_providers = services
            .get_all_providers()
            .await
            .map_err(Error::Application)?;

        let futures: Vec<_> = all_providers
            .into_iter()
            .filter_map(|any_provider| any_provider.into_configured())
            .map(|provider| {
                let provider_id = provider.id.clone();
                let services = services.clone();
                async move {
                    let result: anyhow::Result<Vec<(forge_domain::ProviderId, forge_domain::Model)>> =
                        async {
                            let refreshed = services
                                .provider_auth_service()
                                .refresh_provider_credential(provider)
                                .await?;
                            let models = services.models(refreshed).await?;
                            Ok(models
                                .into_iter()
                                .map(|m| (provider_id.clone(), m))
                                .collect())
                        }
                        .await;
                    result
                }
            })
            .collect();

        // Execute all provider fetches concurrently, then flatten.
        let mut all_models: Vec<(forge_domain::ProviderId, forge_domain::Model)> =
            join_all(futures)
                .await
                .into_iter()
                .collect::<anyhow::Result<Vec<_>>>()
                .map_err(Error::Application)?
                .into_iter()
                .flatten()
                .collect();

        // Sort by provider ID then model ID.
        all_models.sort_by(|(pid_a, model_a), (pid_b, model_b)| {
            pid_a.cmp(pid_b).then_with(|| model_a.id.as_str().cmp(model_b.id.as_str()))
        });

        // Build available models list and find current model.
        // Use session model override if present, otherwise fall back to agent default.
        let mut available_models = Vec::new();
        let mut current_model_id = {
            let raw: &str = current_agent.provider.deref().as_ref();
            format!("{}/{}", raw, current_agent.model)
        };

        for (provider_id, model) in &all_models {
            let provider_raw: &str = provider_id.deref().as_ref();
            let model_id = format!("{}/{}", provider_raw, model.id);

            // Check if this model matches the session override or agent default.
            // Session override stores the full "provider/model" qualified ID.
            if let Some(override_model_id) = session_model_override {
                if model_id.as_str() == override_model_id.as_str() {
                    current_model_id.clone_from(&model_id);
                }
            } else if model.id == current_agent.model {
                current_model_id.clone_from(&model_id);
            }

            let display_name = format!("{}/{}", provider_raw, model.id);
            let mut model_info =
                acp::ModelInfo::new(model_id, display_name)
                    .description(model.description.clone());

            let mut meta = serde_json::Map::new();
            if let Some(context_length) = model.context_length {
                meta.insert(
                    "contextLength".to_string(),
                    serde_json::json!(context_length),
                );
            }
            if let Some(tools_supported) = model.tools_supported {
                meta.insert(
                    "toolsSupported".to_string(),
                    serde_json::json!(tools_supported),
                );
            }
            if let Some(supports_reasoning) = model.supports_reasoning {
                meta.insert(
                    "supportsReasoning".to_string(),
                    serde_json::json!(supports_reasoning),
                );
            }
            if !model.input_modalities.is_empty() {
                let modalities = model
                    .input_modalities
                    .iter()
                    .map(|modality| format!("{:?}", modality).to_lowercase())
                    .collect::<Vec<_>>();
                meta.insert("inputModalities".to_string(), serde_json::json!(modalities));
            }
            if !meta.is_empty() {
                model_info = model_info.meta(meta);
            }

            available_models.push(model_info);
        }

        Ok(
            acp::SessionModelState::new(current_model_id, available_models)
                .meta({
                    let mut meta = serde_json::Map::new();
                    meta.insert("searchable".to_string(), serde_json::json!(true));
                    meta.insert("searchThreshold".to_string(), serde_json::json!(10));
                    meta.insert("filterable".to_string(), serde_json::json!(true));
                    meta.insert("groupBy".to_string(), serde_json::json!("provider"));
                    meta
                }),
        )
    }

    /// Loads MCP server configurations provided by the ACP client.
    ///
    /// # Trust model
    ///
    /// The stdio transport inherits OS-level process isolation, so the
    /// client is the parent process (Acepe). Server names are validated
    /// to prevent injection. The configs are written to the local scope
    /// only and do not persist across Forge restarts unless the caller
    /// explicitly saves them.
    pub(super) async fn load_mcp_servers<S: Services + ?Sized>(
        services: &S,
        mcp_servers: &[acp::McpServer],
    ) -> Result<()> {
        let mut config = services
            .mcp_config_manager()
            .read_mcp_config(Some(&Scope::Local))
            .await
            .map_err(Error::Application)?;

        let server_names: Vec<String> = mcp_servers
            .iter()
            .filter_map(|s| {
                match Self::acp_to_mcp_server_config(s) {
                    Ok((name, server_config)) => {
                        config.mcp_servers.insert(name.clone(), server_config);
                        Some(name.to_string())
                    }
                    Err(error) => {
                        tracing::warn!("Skipping invalid MCP server config: {}", error);
                        None
                    }
                }
            })
            .collect();

        tracing::info!("Loading {} MCP servers from ACP client: {:?}", server_names.len(), server_names);

        services
            .mcp_config_manager()
            .write_mcp_config(&config, &Scope::Local)
            .await
            .map_err(Error::Application)?;
        services.mcp_service().reload_mcp().await.map_err(Error::Application)?;
        Ok(())
    }

    fn acp_to_mcp_server_config(server: &acp::McpServer) -> Result<(ServerName, McpServerConfig)> {
        match server {
            acp::McpServer::Stdio(stdio) => {
                Self::validate_server_name(&stdio.name)?;
                let env = stdio
                    .env
                    .iter()
                    .map(|entry| (entry.name.clone(), entry.value.clone()))
                    .collect::<BTreeMap<_, _>>();
                Ok((
                    ServerName::from(stdio.name.clone()),
                    McpServerConfig::new_stdio(stdio.command.to_string_lossy().to_string(), stdio.args.clone(), Some(env)),
                ))
            }
            acp::McpServer::Http(http) => {
                Self::validate_server_name(&http.name)?;
                Ok((
                    ServerName::from(http.name.clone()),
                    McpServerConfig::Http(McpHttpServer {
                        url: http.url.clone(),
                        headers: http
                            .headers
                            .iter()
                            .map(|header| (header.name.clone(), header.value.clone()))
                            .collect(),
                        timeout: None,
                        disable: false,
                        oauth: McpOAuthSetting::AutoDetect,
                    }),
                ))
            }
            acp::McpServer::Sse(sse) => {
                Self::validate_server_name(&sse.name)?;
                Ok((
                    ServerName::from(sse.name.clone()),
                    McpServerConfig::Http(McpHttpServer {
                        url: sse.url.clone(),
                        headers: sse
                            .headers
                            .iter()
                            .map(|header| (header.name.clone(), header.value.clone()))
                            .collect(),
                        timeout: None,
                        disable: false,
                        oauth: McpOAuthSetting::AutoDetect,
                    }),
                ))
            }
            _ => Err(Error::Application(anyhow::anyhow!(
                "Unsupported MCP server type"
            ))),
        }
    }

    /// Validates that an MCP server name is safe to use as a config key.
    fn validate_server_name(name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(Error::Application(anyhow::anyhow!(
                "MCP server name must not be empty"
            )));
        }
        if name.len() > MAX_SERVER_NAME_LEN {
            return Err(Error::Application(anyhow::anyhow!(
                "MCP server name exceeds maximum length of {} characters",
                MAX_SERVER_NAME_LEN
            )));
        }
        // Only allow alphanumeric, hyphens, underscores, and dots.
        if !name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.') {
            return Err(Error::Application(anyhow::anyhow!(
                "MCP server name '{}' contains invalid characters (allowed: alphanumeric, -, _, .)",
                name
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use agent_client_protocol as acp;
    use agent_client_protocol::{EnvVariable, HttpHeader};
    use forge_domain::{McpOAuthSetting, McpServerConfig};

    use super::StateBuilders;

    #[test]
    fn maps_stdio_servers_with_env() {
        let server = acp::McpServer::Stdio(
            acp::McpServerStdio::new("local-server", "/bin/echo")
                .args(vec!["hello".to_string()])
                .env(vec![EnvVariable::new("TOKEN", "secret")]),
        );

        let (name, config) = StateBuilders::acp_to_mcp_server_config(&server).unwrap();

        assert_eq!(name.to_string(), "local-server");
        match config {
            McpServerConfig::Stdio(stdio) => {
                assert_eq!(stdio.command, "/bin/echo");
                assert_eq!(stdio.args, vec!["hello".to_string()]);
                assert_eq!(stdio.env.get("TOKEN"), Some(&"secret".to_string()));
            }
            McpServerConfig::Http(_) => panic!("expected stdio config"),
        }
    }

    #[test]
    fn maps_http_servers_with_auto_detect_oauth() {
        let server = acp::McpServer::Http(
            acp::McpServerHttp::new("remote.server", "https://example.com/mcp").headers(vec![
                HttpHeader::new("Authorization", "Bearer token"),
            ]),
        );

        let (name, config) = StateBuilders::acp_to_mcp_server_config(&server).unwrap();

        assert_eq!(name.to_string(), "remote.server");
        match config {
            McpServerConfig::Http(http) => {
                assert_eq!(http.url, "https://example.com/mcp");
                assert_eq!(
                    http.headers.get("Authorization"),
                    Some(&"Bearer token".to_string())
                );
                assert_eq!(http.oauth, McpOAuthSetting::AutoDetect);
            }
            McpServerConfig::Stdio(_) => panic!("expected http config"),
        }
    }

    #[test]
    fn rejects_invalid_server_names() {
        let server = acp::McpServer::Sse(acp::McpServerSse::new(
            "bad server name!",
            "https://example.com/sse",
        ));

        let error = StateBuilders::acp_to_mcp_server_config(&server).unwrap_err();
        let actual = error.to_string();

        assert!(actual.contains("invalid characters"));
    }
}
