use std::{fs, io, path::PathBuf, sync::Arc, time::Duration};

use awc::Client;
use serde::Deserialize;
use sysinfo::{MemoryRefreshKind, RefreshKind, System};
use tokio::sync::Mutex;

use actix_web::{
    App, HttpResponse, HttpServer, error,
    middleware::Logger,
    web::{self, PayloadConfig},
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
}

struct ResourceMonitor {
    config: Config,
    sys: System,
    refresh_kind: RefreshKind,
    last_under_load: Option<tokio::time::Instant>,
    last_model_loaded: Option<tokio::time::Instant>,
    vram_path: PathBuf,
    total_vram: u64,
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
        let ram_used = self.sys.used_memory() as f64 / self.sys.total_memory() as f64;

        if vram_used > self.config.vram_thresh || ram_used > self.config.ram_thresh {
            self.last_under_load = Some(tokio::time::Instant::now());
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
            self.last_model_loaded = Some(tokio::time::Instant::now());
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

async fn proxy_handler(
    resource_monitor: web::Data<Arc<Mutex<ResourceMonitor>>>,
    config: web::Data<Config>,
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
            .insert_header(("retry-after", "10"))
            .insert_header(("content-type", "application/json"))
            .body(
                r#"{"error":{"message":"Resource load too high, try again later","type":"overloaded","code":503}}"#
                    .to_string(),
            ));
    }

    let client = Client::new();
    let upstream_url = format!(
        "http://{}:{}{}",
        config.target_host,
        config.target_port,
        req.uri()
    );

    match client
        .request_from(&upstream_url, req.head())
        .timeout(Duration::from_secs(3600)) // llama.cpp default HTTP read write timeout
        .send_body(body)
        .await
    {
        Ok(upstream_res) => {
            log::info!("Dispatching response: status={}", upstream_res.status());
            // let stream = BodyStream::new(upstream_res.take_payload());
            let headers = upstream_res.headers().clone();

            let mut res = HttpResponse::build(upstream_res.status()).streaming(upstream_res);
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

#[actix_web::main]
async fn main() -> io::Result<()> {
    env_logger::init();

    let config_path =
        std::env::var("GATED_PROXY_CONFIG").unwrap_or_else(|_| "config.json".to_string());
    let config: Config = serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();

    let resource_monitor = Arc::new(Mutex::new(ResourceMonitor::new(config.clone())));

    let monitor_data = web::Data::new(resource_monitor.clone());
    let config_data = web::Data::new(config.clone());

    actix_web::rt::spawn(async move {
        loop {
            if let Err(why) = resource_monitor.lock().await.refresh().await {
                log::error!("Resource monitor refresh failed: {why}");
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(monitor_data.clone())
            .app_data(config_data.clone())
            .app_data(PayloadConfig::new(100 * 1024 * 1024)) // 100 MB, the same as llama.cpp
            .default_service(web::to(proxy_handler))
    })
    .bind((config.host, config.port))?
    .run()
    .await
}
