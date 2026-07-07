use std::{collections::HashMap, sync::Arc, time::SystemTime};

use serde::Serialize;

use crate::{ChatCompletionsChunkChoices, ChatCompletionsRequest, unload_model};

pub struct Telemetry {
    pub requests: HashMap<u64, Arc<std::sync::RwLock<TelemetryRequest>>>,
    pub next_id: u64,
}

impl Telemetry {
    pub fn new() -> Self {
        Self {
            requests: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn new_request(
        &mut self,
        cc_request: ChatCompletionsRequest,
    ) -> Arc<std::sync::RwLock<TelemetryRequest>> {
        let id = self.next_id;
        self.next_id += 1;

        let request = Arc::new(std::sync::RwLock::new(TelemetryRequest {
            cc_request,
            reasoning: String::new(),
            content: String::new(),
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            haywire_detected: false,
        }));
        self.requests.insert(id, request.clone());

        request
    }

    /// Cleanup old requests
    pub async fn cleanup(&mut self) {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut to_delete = Vec::new();
        for (id, request) in &self.requests {
            // FIXME: Configurable
            if timestamp.saturating_sub(request.read().unwrap().timestamp) > 3600 {
                to_delete.push(*id);
            }
        }

        for id in to_delete {
            self.requests.remove(&id);
        }
    }
}

#[derive(Serialize, Clone)]
pub struct TelemetryRequest {
    cc_request: ChatCompletionsRequest,
    reasoning: String,
    content: String,
    timestamp: u64,
    haywire_detected: bool,
}

impl TelemetryRequest {
    pub fn process_cc_response(
        &mut self,
        haywire_detector: Option<&HaywireDetector>,
        model_name: &str,
        target_host: &str,
        target_port: u16,
        created: u64,
        choices: &[ChatCompletionsChunkChoices],
    ) {
        self.timestamp = created;

        for choice in choices {
            if let Some(delta) = &choice.delta {
                if let Some(reasoning) = &delta.reasoning_content {
                    self.reasoning.push_str(reasoning);
                }
                if let Some(content) = &delta.content {
                    self.content.push_str(content);
                }
            }
            if let Some(message) = &choice.message
                && let Some(content) = &message.content
            {
                self.content.push_str(content);
            }
        }

        if let Some(detector) = haywire_detector
            && !self.haywire_detected
            && (detector.check_haywire(&self.content) || detector.check_haywire(&self.reasoning))
        {
            self.haywire_detected = true;
            log::error!("Haywire detected for model '{model_name}', unloading");

            let target_host_clone = target_host.to_string();
            let model_name_clone = model_name.to_string();
            actix_web::rt::spawn(async move {
                if let Err(e) =
                    unload_model(&target_host_clone, target_port, &model_name_clone).await
                {
                    log::error!("Failed to unload haywire model '{model_name_clone}': {e}");
                } else {
                    log::info!("Successfully unloaded haywire model '{model_name_clone}'");
                }
            });
        }
    }
}

pub struct HaywireDetector {
    char_repeats: usize,
}

impl HaywireDetector {
    pub fn new(char_repeats: usize) -> Self {
        Self { char_repeats }
    }

    /// Check if the accumulated content indicates haywire behavior.
    /// Returns true if a repeated pattern is detected.
    pub fn check_haywire(&self, text: &str) -> bool {
        if text.len() < self.char_repeats {
            return false;
        }

        // Check for single character repetition (e.g., "//////////////")
        let mut count = 1usize;
        let mut haywire = false;
        for window in text.chars().collect::<Vec<_>>().windows(2) {
            if window[0] == window[1] {
                count += 1;
                if count >= self.char_repeats {
                    haywire = true;
                    break;
                }
            } else {
                count = 1;
            }
        }

        haywire
    }
}
