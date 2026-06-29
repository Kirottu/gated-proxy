use std::{
    collections::HashMap,
    fs, io,
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime},
};

use awc::Client;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sse_core::{SseDecoder, SseEvent::Message};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};
use tokio::{
    sync::{Mutex, RwLock},
    time::Instant,
};

use actix_web::{
    App, HttpResponse, HttpServer, error, get,
    middleware::Logger,
    web::{self, BytesMut, Data, Path, PayloadConfig},
};

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<Model>,
}

#[derive(Deserialize)]
struct Model {
    status: ModelStatus,
}

#[derive(Deserialize)]
struct ModelStatus {
    value: ModelStatusValue,
}

#[derive(Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
enum ModelStatusValue {
    Unloaded,
    Loading,
    Loaded,
    Sleeping,
}

#[derive(Deserialize, Clone)]
struct Config {
    host: String,
    port: u16,
    target_host: String,
    target_port: u16,

    /// What percentage of RAM the current usage should be under
    ram_thresh: f64,
    /// What percentage of VRAM the current usage should be under
    vram_thresh: f64,
    /// How long the system should have been "idling" for based on the other
    /// thresholds
    idle_thresh: u64,

    gpu_sysfs_path: PathBuf,

    /// Whether to enable haywire detection (repeated token patterns)
    #[serde(default)]
    haywire_detection_enabled: bool,

    /// Minimum number of consecutive identical characters before flagging as haywire
    #[serde(default = "Config::default_haywire_char_repeats")]
    haywire_char_repeats: usize,
}

impl Config {
    fn default_haywire_char_repeats() -> usize {
        40
    }
}

struct ResourceMonitor {
    config: Config,
    sys: System,
    refresh_kind: RefreshKind,
    last_under_load: Option<Instant>,
    last_model_loaded: Option<Instant>,
    vram_path: PathBuf,
    total_vram: u64,
}

struct Telemetry {
    requests: HashMap<u64, Arc<std::sync::RwLock<TelemetryRequest>>>,
    next_id: u64,
}

#[derive(Serialize, Clone)]
struct TelemetryRequest {
    cc_request: ChatCompletionsRequest,
    reasoning: String,
    content: String,
    timestamp: u64,
    haywire_detected: bool,
}

struct HaywireDetector {
    char_repeats: usize,
}

impl HaywireDetector {
    fn new(char_repeats: usize) -> Self {
        Self { char_repeats }
    }

    /// Check if the accumulated content indicates haywire behavior.
    /// Returns true if a repeated pattern is detected.
    fn check_haywire(&self, text: &str) -> bool {
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

#[derive(Deserialize, Serialize, Clone)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<ChatCompletionMessage>,
}

#[derive(Deserialize, Serialize, Clone)]
struct ChatCompletionMessage {
    role: String,
    content: String,
}

#[derive(Deserialize, Serialize)]
struct ChatCompletionsChunk {
    choices: Vec<ChatCompletionsChunkChoices>,
    created: u64,
}

#[derive(Deserialize, Serialize)]
struct ChatCompletionsChunkChoices {
    delta: ChatCompletionsChunkDelta,
}

#[derive(Deserialize, Serialize)]
struct ChatCompletionsChunkDelta {
    reasoning_content: Option<String>,
    content: Option<String>,
}

impl ResourceMonitor {
    fn new(config: Config) -> Self {
        let refresh_kind = RefreshKind::nothing().with_memory(MemoryRefreshKind::everything());
        let sys = System::new_with_specifics(refresh_kind);
        let mut total_vram_path = config.gpu_sysfs_path.clone();
        let mut vram_path = config.gpu_sysfs_path.clone();

        total_vram_path.extend(&["mem_info_vram_total"]);
        vram_path.extend(&["mem_info_vram_used"]);

        let total_vram = fs::read_to_string(total_vram_path)
            .unwrap()
            .trim()
            .parse()
            .unwrap();

        Self {
            config,
            sys,
            refresh_kind,
            last_under_load: None,
            last_model_loaded: None,
            total_vram,
            vram_path,
        }
    }

    async fn refresh(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.sys.refresh_specifics(self.refresh_kind);
        let vram: u64 = fs::read_to_string(&self.vram_path)?.trim().parse()?;

        let vram_used = vram as f64 / self.total_vram as f64;
        // TODO: figure out what `used_memory` actually entails
        let ram_used = self.sys.used_memory() as f64 / self.sys.total_memory() as f64;

        if vram_used > self.config.vram_thresh || ram_used > self.config.ram_thresh {
            self.last_under_load = Some(Instant::now());
        }

        let client = Client::new();
        let models: ModelsResponse = client
            .get(format!(
                "http://{}:{}/v1/models",
                self.config.target_host, self.config.target_port
            ))
            .timeout(Duration::from_secs(3600)) // llama.cpp default HTTP read write timeout
            .send()
            .await?
            .json()
            .await?;

        if models.data.iter().any(|model| {
            model.status.value == ModelStatusValue::Loaded
                || model.status.value == ModelStatusValue::Loading
        }) {
            self.last_model_loaded = Some(Instant::now());
        }

        Ok(())
    }

    async fn api_allowed(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        self.refresh().await?;

        let load_allowed = self
            .last_under_load
            .map(|instant| instant.elapsed().as_secs())
            .unwrap_or(u64::MAX)
            > self.config.idle_thresh;
        let grace_allowed = self
            .last_model_loaded
            .map(|instant| instant.elapsed().as_secs())
            .unwrap_or(u64::MAX)
            < self.config.idle_thresh;

        log::debug!("Load allowed: {load_allowed}, grace allowed: {grace_allowed}");
        // Allow API queries if either:
        // - System has been idling for more than the threshold period
        // - A model was loaded more recently than the threshold period
        Ok(load_allowed || grace_allowed)
    }
}

impl Telemetry {
    fn new() -> Self {
        Self {
            requests: HashMap::new(),
            next_id: 0,
        }
    }

    fn new_request(
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
    async fn cleanup(&mut self) {
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

async fn proxy_handler(
    resource_monitor: Data<Mutex<ResourceMonitor>>,
    telemetry: Data<RwLock<Telemetry>>,
    config: Data<Config>,
    req: actix_web::HttpRequest,
    body: web::Bytes,
) -> Result<HttpResponse, actix_web::Error> {
    let api_allowed = match resource_monitor.lock().await.api_allowed().await {
        Ok(val) => val,
        Err(why) => {
            log::error!("Resource monitor reported an error, failing open: {why}");
            true
        }
    };

    if !api_allowed {
        log::info!("Proxied request disallowed due to system conditions");
        return Ok(HttpResponse::ServiceUnavailable()
            .insert_header(("retry-after", "120"))
            .json(serde_json::json!({
                "error": {
                    "message": "Resource load too high, try again later",
                    "type": "rate_limit_error",
                    "code": "resource_exhausted"
                }
            })));
    }

    let client = Client::new();
    let upstream_url = format!(
        "http://{}:{}{}",
        config.target_host,
        config.target_port,
        req.uri()
    );

    log::trace!("body: {}", String::from_utf8_lossy(&body));

    match client
        .request_from(&upstream_url, req.head())
        .timeout(Duration::from_secs(3600)) // llama.cpp default HTTP read write timeout
        .no_decompress()
        .send_body(body.clone())
        .await
    {
        Ok(upstream_res) => {
            log::info!("Dispatching response: status={}", upstream_res.status());
            let headers = upstream_res.headers().clone();

            // If the body parses successfully as an OpenAI Chat Completions request, start doing telemetry on it
            let mut res =
                if let Ok(cc_request) = serde_json::from_slice::<ChatCompletionsRequest>(&body) {
                    log::info!("Starting telemetry: {}", telemetry.read().await.next_id);

                    // Set up haywire detector if enabled and config is available
                    let haywire_detector = if config.haywire_detection_enabled {
                        log::info!(
                            "Haywire detection enabled: char_repeats={}",
                            config.haywire_char_repeats,
                        );
                        Some(HaywireDetector::new(config.haywire_char_repeats))
                    } else {
                        None
                    };

                    let request = telemetry.write().await.new_request(cc_request.clone());
                    let request_clone = request.clone();
                    let target_host = config.target_host.clone();
                    let target_port = config.target_port;
                    let model_name = cc_request.model.clone();
                    let mut buffer = BytesMut::new();
                    let mut decoder = SseDecoder::new();

                    HttpResponse::build(upstream_res.status())
                        .streaming(upstream_res.inspect(move |res| {
                        if let Ok(data) = res {
                            buffer.extend(data);

                            while let Some(event) = decoder.next(&mut buffer) {
                                let Ok(event) = event else {
                                    log::error!("Stream parse error: {event:?}");
                                    return;
                                };
                                let Message(msg) = event else {
                                    log::error!("Unexpected SSE event: {event:?}");
                                    return;
                                };

                                // Break out on the stream ending message
                                if &msg.data == "[DONE]" {
                                    continue;
                                }

                                match serde_json::from_str::<ChatCompletionsChunk>(&msg.data) {
                                    Ok(chunk) => {
                                        let mut request = request_clone.write().unwrap();

                                        request.timestamp = chunk.created;

                                        for choice in chunk.choices {
                                            if let Some(reasoning) = choice.delta.reasoning_content
                                            {
                                                request.reasoning.push_str(&reasoning);
                                            }
                                            if let Some(content) = choice.delta.content {
                                                request.content.push_str(&content);
                                            }
                                        }

                                        // Check for haywire behavior if detector is enabled and not already detected
                                        if let Some(detector) = &haywire_detector
                                            && !request.haywire_detected
                                            && (detector.check_haywire(&request.content)
                                                || detector.check_haywire(&request.reasoning))
                                        {
                                            request.haywire_detected = true;
                                            log::error!(
                                                "Haywire detected for model '{}', unloading",
                                                model_name
                                            );

                                            // Unload the model asynchronously in background task
                                            let target_host_clone = target_host.clone();
                                            let target_port_clone = target_port;
                                            let model_name_clone = model_name.clone();
                                            actix_web::rt::spawn(async move {
                                                let client = Client::new();
                                                if let Err(e) = client
                                                    .post(format!(
                                                        "http://{}:{}/models/unload",
                                                        target_host_clone, target_port_clone
                                                    ))
                                                    .timeout(Duration::from_secs(30))
                                                    .insert_header((
                                                        "content-type",
                                                        "application/json",
                                                    ))
                                                    .send_json(&serde_json::json!({
                                                        "model": model_name_clone
                                                    }))
                                                    .await
                                                {
                                                    log::error!(
                                                        "Failed to unload haywire model '{}': {}",
                                                        model_name_clone,
                                                        e
                                                    );
                                                } else {
                                                    log::info!(
                                                        "Successfully unloaded haywire model '{}'",
                                                        model_name_clone
                                                    );
                                                }
                                            });
                                        }
                                    }
                                    Err(why) => {
                                        log::error!("Failed to parse chunk: {why}");
                                        log::info!("chunk: {}", msg.data);
                                    }
                                }
                            }
                        }
                    }))
                } else {
                    HttpResponse::build(upstream_res.status()).streaming(upstream_res)
                };

            for (name, value) in headers {
                res.headers_mut().insert(name, value);
            }
            Ok(res)
        }
        Err(why) => {
            log::error!("Upstream request failed: {why}");
            Err(error::ErrorInternalServerError(why))
        }
    }
}

#[get("/telemetry/{id}")]
async fn telemetry_request_json(
    telemetry: Data<RwLock<Telemetry>>,
    id: Path<u64>,
) -> Result<HttpResponse, actix_web::Error> {
    let telemetry = telemetry.read().await;
    let request = telemetry
        .requests
        .get(&id)
        .ok_or(error::ErrorNotFound("No request with such ID exists"))?;

    Ok(HttpResponse::Ok().json(&*request.read().unwrap()))
}

#[actix_web::main]
async fn main() -> io::Result<()> {
    env_logger::init();

    let config_path =
        std::env::var("GATED_PROXY_CONFIG").unwrap_or_else(|_| "config.json".to_string());
    let config: Config = serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();

    let resource_monitor = Data::new(Mutex::new(ResourceMonitor::new(config.clone())));
    let resource_monitor_clone = resource_monitor.clone();

    let telemetry = Data::new(RwLock::new(Telemetry::new()));
    let telemetry_clone = telemetry.clone();

    let config_data = Data::new(config.clone());

    actix_web::rt::spawn(async move {
        loop {
            if let Err(why) = resource_monitor_clone.lock().await.refresh().await {
                log::error!("Resource monitor refresh failed: {why}");
            }
            telemetry_clone.write().await.cleanup().await;
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(resource_monitor.clone())
            .app_data(telemetry.clone())
            .app_data(config_data.clone())
            .app_data(PayloadConfig::new(100 * 1024 * 1024)) // 100 MB, the same as llama.cpp
            .service(telemetry_request_json)
            .default_service(web::to(proxy_handler))
    })
    .bind((config.host, config.port))?
    .run()
    .await
}
