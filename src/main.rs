use std::{
    collections::{HashMap, VecDeque},
    fs, io,
    time::Duration,
};

use all_smi::AllSmi;
use awc::Client;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sse_core::{SseDecoder, SseEvent::Message};
use tokio::{
    sync::{Mutex, RwLock},
    time::Instant,
};

use actix_web::{
    App, HttpResponse, HttpServer, error, get,
    middleware::Logger,
    web::{self, BytesMut, Data, Path, PayloadConfig},
};

mod telemetry;

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<Model>,
}

#[derive(Deserialize)]
struct Model {
    id: String,
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

    /// Which GPU to use
    gpu: String,

    /// How long in seconds the load average window is
    load_average_window: usize,

    cpu_load_average_thresh: f64,
    gpu_load_average_thresh: f64,

    model_loaded_grace: u64,

    /// Whether to enable haywire detection (repeated token patterns)
    #[serde(default)]
    haywire_detection_enabled: bool,

    /// Minimum number of consecutive identical characters before flagging as haywire
    #[serde(default = "Config::default_haywire_char_repeats")]
    haywire_char_repeats: usize,

    idle_unload_timeouts: HashMap<String, u64>,
    default_unload_timeout: Option<u64>,
}

impl Config {
    fn default_haywire_char_repeats() -> usize {
        40
    }
}

struct ResourceMonitor {
    config: Config,
    smi: AllSmi,
    cpu_load_samples: VecDeque<f64>,
    gpu_load_samples: VecDeque<f64>,
    last_model_loaded: Option<Instant>,
    /// Monitor when models have been active to unload when idle for a long enough time
    models_last_active: HashMap<String, Instant>,
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

#[derive(Deserialize)]
struct ChatCompletionsChunk {
    choices: Vec<ChatCompletionsChunkChoices>,
    created: u64,
}

#[derive(Deserialize)]
struct ChatCompletionsChunkChoices {
    delta: Option<ChatCompletionsChunkDelta>,
    message: Option<ChatCompletionsMessage>,
}

#[derive(Deserialize)]
struct ChatCompletionsChunkDelta {
    reasoning_content: Option<String>,
    content: Option<String>,
}

#[derive(Deserialize)]
struct ChatCompletionsMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct Slot {
    is_processing: bool,
}

impl ResourceMonitor {
    fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        let smi = AllSmi::new()?;

        Ok(Self {
            // Initialize with zeroes to allow everything at first
            cpu_load_samples: vec![0.0; config.load_average_window].into(),
            gpu_load_samples: vec![0.0; config.load_average_window].into(),
            config,
            smi,
            // sys,
            // refresh_kind,
            // last_under_load: None,
            last_model_loaded: None,
            // total_vram,
            // vram_path,
            models_last_active: HashMap::new(),
        })
    }

    async fn refresh(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(gpu) = self
            .smi
            .get_gpu_info()
            .into_iter()
            .find(|gpu| gpu.name == self.config.gpu)
        {
            self.gpu_load_samples.push_back(gpu.utilization);

            if self.gpu_load_samples.len() > self.config.load_average_window {
                self.gpu_load_samples.pop_front();
            }
        } else {
            log::warn!("Named GPU not found, failing open");
        }

        let cpu = self.smi.get_cpu_info().pop().unwrap();

        self.cpu_load_samples.push_back(cpu.utilization);

        if self.cpu_load_samples.len() > self.config.load_average_window {
            self.cpu_load_samples.pop_front();
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

        for model in models.data {
            if model.status.value == ModelStatusValue::Loaded
                || model.status.value == ModelStatusValue::Loading
            {
                self.last_model_loaded = Some(Instant::now());
            }

            if let Some(timeout) = self
                .config
                .idle_unload_timeouts
                .get(&model.id)
                .or(self.config.default_unload_timeout.as_ref())
            {
                // Only query slots if the model is loaded since otherwise it would wake the model
                if model.status.value == ModelStatusValue::Loaded
                    && let Ok(slots) = client
                        .get(format!(
                            "http://{}:{}/slots?model={}",
                            self.config.target_host, self.config.target_port, model.id
                        ))
                        .timeout(Duration::from_secs(3600)) // llama.cpp default HTTP read write timeout
                        .send()
                        .await?
                        .json::<Vec<Slot>>()
                        .await
                {
                    // When the model has just been loaded, there is no entry in the hashmap
                    if !self.models_last_active.contains_key(&model.id) {
                        self.models_last_active
                            .insert(model.id.clone(), Instant::now());
                    }

                    if slots.iter().any(|slot| slot.is_processing) {
                        self.models_last_active
                            .insert(model.id.clone(), Instant::now());
                    }
                    if self
                        .models_last_active
                        .get(&model.id)
                        .unwrap()
                        .elapsed()
                        .as_secs()
                        > *timeout
                    {
                        unload_model(&self.config.target_host, self.config.target_port, &model.id)
                            .await?;
                    }
                }
            }

            // let mut to_unload = Vec::new();

            // for (model, last_active) in &self.models_last_active {
            //     let timeout = self
            //         .config
            //         .idle_unload_timeouts
            //         .get(model)
            //         .unwrap_or(&self.config.default_unload_timeout);

            //     if last_active.elapsed().as_secs() > *timeout {
            //         to_unload.push(model.clone());
            //     }
            // }

            // for model in to_unload {
            //     unload_model(&self.config.target_host, self.config.target_port, &model).await?;
            //     self.models_last_active.remove(&model);
            // }
        }

        Ok(())
    }

    async fn api_allowed(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        self.refresh().await?;

        let cpu_average =
            self.cpu_load_samples.iter().sum::<f64>() / self.cpu_load_samples.len() as f64;
        let gpu_average =
            self.gpu_load_samples.iter().sum::<f64>() / self.gpu_load_samples.len() as f64;

        let load_allowed = cpu_average < self.config.cpu_load_average_thresh
            && gpu_average < self.config.gpu_load_average_thresh;

        let grace_allowed = self
            .last_model_loaded
            .map(|instant| instant.elapsed().as_secs())
            .unwrap_or(u64::MAX)
            < self.config.model_loaded_grace;

        log::debug!("Load allowed: {load_allowed}, grace allowed: {grace_allowed}");
        // Allow API queries if either:
        // - System has been idling for more than the threshold period
        // - A model was loaded more recently than the threshold period
        Ok(load_allowed || grace_allowed)
    }
}

async fn unload_model(
    target_host: &str,
    target_port: u16,
    model: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();

    client
        .post(format!(
            "http://{}:{}/models/unload",
            target_host, target_port
        ))
        .timeout(Duration::from_secs(30))
        .insert_header(("content-type", "application/json"))
        .send_json(&serde_json::json!({
            "model": model
        }))
        .await?;

    Ok(())
}

async fn proxy_handler(
    resource_monitor: Data<Mutex<ResourceMonitor>>,
    telemetry: Data<RwLock<telemetry::Telemetry>>,
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
        Ok(mut upstream_res) => {
            log::info!("Dispatching response: status={}", upstream_res.status());
            let headers = upstream_res.headers().clone();

            // If the body parses successfully as an OpenAI Chat Completions request, start doing telemetry on it
            let mut res = if let Ok(cc_request) =
                serde_json::from_slice::<ChatCompletionsRequest>(&body)
            {
                log::info!("Starting telemetry: {}", telemetry.read().await.next_id);

                let is_streaming = upstream_res
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .map(|ct| ct.contains("text/event-stream"))
                    .unwrap_or(false);

                // Set up haywire detector if enabled and config is available
                let haywire_detector = if config.haywire_detection_enabled {
                    log::info!(
                        "Haywire detection enabled: char_repeats={}",
                        config.haywire_char_repeats,
                    );
                    Some(telemetry::HaywireDetector::new(config.haywire_char_repeats))
                } else {
                    None
                };

                let request = telemetry.write().await.new_request(cc_request.clone());
                let target_host = config.target_host.clone();
                let target_port = config.target_port;
                let model_name = cc_request.model.clone();

                if is_streaming {
                    let request_clone = request.clone();
                    let mut buffer = BytesMut::new();
                    let mut decoder = SseDecoder::new();

                    HttpResponse::build(upstream_res.status()).streaming(upstream_res.inspect(
                        move |res| {
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
                                            let mut req = request_clone.write().unwrap();
                                            req.process_cc_response(
                                                haywire_detector.as_ref(),
                                                &model_name,
                                                &target_host,
                                                target_port,
                                                chunk.created,
                                                &chunk.choices,
                                            );
                                        }
                                        Err(why) => {
                                            log::error!("Failed to parse chunk: {why}");
                                            log::info!("chunk: {}", msg.data);
                                        }
                                    }
                                }
                            }
                        },
                    ))
                } else {
                    // Non-streamed response: parse the full body as a single JSON object
                    let response_body = upstream_res
                        .body()
                        .await
                        .map_err(error::ErrorInternalServerError)?;
                    match serde_json::from_slice::<ChatCompletionsChunk>(&response_body) {
                        Ok(cc_response) => {
                            let mut req = request.write().unwrap();
                            req.process_cc_response(
                                haywire_detector.as_ref(),
                                &model_name,
                                &target_host,
                                target_port,
                                cc_response.created,
                                &cc_response.choices,
                            );
                            HttpResponse::build(upstream_res.status()).body(response_body)
                        }
                        Err(why) => {
                            log::error!("Failed to parse non-streamed response: {why}");
                            HttpResponse::build(upstream_res.status()).body(response_body)
                        }
                    }
                }
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
    telemetry: Data<RwLock<telemetry::Telemetry>>,
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

    let resource_monitor = Data::new(Mutex::new(ResourceMonitor::new(config.clone()).unwrap()));
    let resource_monitor_clone = resource_monitor.clone();

    let telemetry = Data::new(RwLock::new(telemetry::Telemetry::new()));
    let telemetry_clone = telemetry.clone();

    let config_data = Data::new(config.clone());

    actix_web::rt::spawn(async move {
        loop {
            if let Err(why) = resource_monitor_clone.lock().await.refresh().await {
                log::error!("Resource monitor refresh failed: {why}");
            }
            telemetry_clone.write().await.cleanup().await;
            tokio::time::sleep(Duration::from_secs(5)).await;
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
