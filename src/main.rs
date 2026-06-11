use std::{convert::Infallible, env, fs, io, path::PathBuf, pin::Pin, sync::Arc, time::Duration};

use actix_http::{
    BoxedPayloadStream, HttpMessage, HttpService, Request, Response, StatusCode, header::HeaderName,
};
use actix_server::Server;
use awc::{Client, ResponseBody, body::BodyStream, error::HeaderValue};
use bytes::{Bytes, BytesMut};
use futures_util::StreamExt;
use serde::Deserialize;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
use tokio::{
    sync::{Mutex, RwLock},
    time::Instant,
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
    last_under_load: Option<Instant>,
    last_model_loaded: Option<Instant>,
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
        // FIXME: Error handling
        log::debug!("Refreshing resource monitor");
        self.sys.refresh_specifics(self.refresh_kind);
        let vram: u64 = fs::read_to_string(&self.vram_path)?.trim().parse()?;

        let vram_used = vram as f64 / self.total_vram as f64;
        let ram_used = self.sys.used_memory() as f64 / self.sys.total_memory() as f64;

        if vram_used > self.config.vram_thresh || ram_used > self.config.ram_thresh {
            log::debug!("System is currently under too much load");
            self.last_under_load = Some(Instant::now());
        }

        let client = Client::new();
        let models: ModelsResponse = client
            .get(format!(
                "http://{}:{}/v1/models",
                self.config.target_host, self.config.target_port
            ))
            .send()
            .await?
            .json()
            .await?;

        // Check if any model is loaded or loading
        if models.data.iter().any(|model| {
            model.status.value == ModelStatusValue::Loaded
                || model.status.value == ModelStatusValue::Loading
        }) {
            log::debug!("A model is currently loaded");
            self.last_model_loaded = Some(Instant::now());
        }

        Ok(())
    }

    async fn api_allowed(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        self.refresh().await?;

        // Allow API queries to go true if either
        // - The system has been idling for more than the threshold period
        // - A model has been loaded more recently than the threshold period
        Ok(self
            .last_under_load
            .map(|instant| instant.elapsed().as_secs())
            .unwrap_or(u64::MAX)
            > self.config.idle_thresh
            || self
                .last_model_loaded
                .map(|instant| instant.elapsed().as_secs())
                .unwrap_or(u64::MAX)
                < self.config.idle_thresh)
    }
}

#[actix_rt::main]
async fn main() -> io::Result<()> {
    env_logger::init();

    let config_path = env::var("GATED_PROXY_CONFIG").unwrap();
    let config = serde_json::from_slice::<Config>(&fs::read(&config_path).unwrap()).unwrap();

    let resource_monitor = Arc::new(Mutex::new(ResourceMonitor::new(config.clone())));
    let config = Arc::new(config);

    let resource_monitor_clone = resource_monitor.clone();
    actix_rt::spawn(async move {
        loop {
            if let Err(why) = resource_monitor_clone.lock().await.refresh().await {
                log::error!("Resource monitor refresh failed: {why}");
            };
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    Server::build()
        .bind(
            "gated-proxy",
            (config.host.clone(), config.port),
            move || {
                let config = config.clone();
                let resource_monitor = resource_monitor.clone();
                HttpService::build()
                    .on_connect_ext(move |_, ext| {
                        ext.insert(config.clone());
                        ext.insert(resource_monitor.clone());
                    })
                    .finish(|mut req: Request| async move {
                        log::debug!("New request: {req:?}");
                        let resource_monitor =
                            req.conn_data::<Arc<Mutex<ResourceMonitor>>>().unwrap();
                        let api_allowed = match resource_monitor.lock().await.api_allowed().await {
                            Ok(val) => val,
                            Err(why) => {
                                log::error!("Resource monitor reported an error, failing open: {why}");
                                true
                            },
                        };
                        if api_allowed {
                            let config = req.conn_data::<Arc<Config>>().unwrap().clone();
                            let client = Client::new();

                            let mut body = BytesMut::new();

                            while let Some(chunk) = req.payload().next().await {
                                body.extend_from_slice(&chunk?);
                            }

                            let mut upstream_res = client
                                .request_from(
                                    format!(
                                        "http://{}:{}{}",
                                        config.target_host,
                                        config.target_port,
                                        req.uri()
                                    ),
                                    req.head(),
                                )
                                .send_body(body)
                                .await.unwrap();
                            let status = upstream_res.status();
                            let headers = upstream_res.headers().clone();
                            let mut upstream_body = BytesMut::new();
                            while let Some(chunk) = upstream_res.next().await {
                                upstream_body.extend(&chunk?);
                            }

                            let mut res = Response::new(status).set_body(upstream_body.freeze());
                            log::debug!("Dispatching response: {res:?}");
                            res.headers_mut().clear();
                            for (name, value) in headers {
                                res.headers_mut().append(name, value);
                            }
                            Ok::<_, actix_http::Error>(res)
                        } else {
                            log::info!("Resource use too high! Sending 503");
                            let mut res = Response::new(StatusCode::SERVICE_UNAVAILABLE);
                            res.headers_mut().insert(
                                HeaderName::from_static("retry-after"),
                                HeaderValue::from_static("10"),
                            );
                            res.headers_mut().insert(
                                HeaderName::from_static("content-type"),
                                HeaderValue::from_static("application/json"),
                            );

                            let data = Bytes::from_static(r#"{"error":{"message":"Resource load too high, try again later","type":"overloaded","code":"resource_exhausted"}}"#.as_bytes());

                            Ok::<_, actix_http::Error>(res.set_body(data))
                        }
                    })
                    .tcp()
            },
        )?
        .run()
        .await
}
