// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Async event loop driven by a single-threaded Tokio runtime.
//! - Device fds registered directly with Tokio's reactor via AsyncFd
//! - vCPU→device MMIO goes through channels (no shared mutex)
//! - Device event handlers are async, enabling future async I/O

use std::future::Future;
use std::os::unix::io::{AsRawFd, RawFd};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;

use tokio::io::unix::AsyncFd;
use tokio::io::Interest;

use crate::devices::virtio::device::VirtioDevice;
use crate::logger::{METRICS, info, warn};
use crate::rpc_interface::{ApiRequest, ApiResponse, RuntimeApiController, VmmAction};
use crate::vstate::bus::BusDevice;
use crate::{FcExitCode, Vmm};

// ---------------------------------------------------------------------------
// Latency stats
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct LatencyStats {
    device_samples: Vec<u64>,
    api_samples: Vec<u64>,
    last_report: Option<Instant>,
}

impl LatencyStats {
    pub fn record_device(&mut self, nanos: u64) { self.device_samples.push(nanos); }
    pub fn record_api(&mut self, nanos: u64) { self.api_samples.push(nanos); }
    pub fn maybe_report(&mut self) {
        let now = Instant::now();
        if self.last_report.map_or(true, |t| now.duration_since(t).as_secs() >= 10) {
            self.last_report = Some(now);
            Self::report(&mut self.device_samples, "device_io");
            Self::report(&mut self.api_samples, "api");
        }
    }
    fn report(samples: &mut Vec<u64>, label: &str) {
        if samples.is_empty() { return; }
        samples.sort_unstable();
        let n = samples.len();
        let p50 = samples[n / 2];
        let p99 = samples[(n * 99 / 100).min(n - 1)];
        info!("latency[{label}] n={n} p50={:.1}us p99={:.1}us",
              p50 as f64 / 1000.0, p99 as f64 / 1000.0);
        samples.clear();
    }
}

// ---------------------------------------------------------------------------
// Device fd handler
// ---------------------------------------------------------------------------

/// An fd + tag + reference to the device that owns it.
pub struct FdHandler {
    pub fd: RawFd,
    pub tag: u32,
    pub device: Arc<Mutex<dyn VirtioDevice>>,
}

impl std::fmt::Debug for FdHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FdHandler(fd={}, tag={})", self.fd, self.tag)
    }
}

/// Build fd→handler mappings from all devices.
/// Must be called before seccomp.
pub fn build_device_handlers(vmm: &Vmm) -> Vec<FdHandler> {
    let mut handlers = Vec::new();
    for (_dev_type, _id, device) in vmm.device_manager.collect_virtio_devices() {
        let fd_tags = device.lock().unwrap().async_fd_tags();
        for (fd, tag) in fd_tags {
            handlers.push(FdHandler { fd, tag, device: device.clone() });
        }
    }
    handlers
}

// ---------------------------------------------------------------------------
// AsyncFd helpers
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct BorrowedFd(RawFd);
impl AsRawFd for BorrowedFd {
    fn as_raw_fd(&self) -> RawFd { self.0 }
}

struct WatchedFd {
    async_fd: AsyncFd<BorrowedFd>,
    tag: u32,
    device: Arc<Mutex<dyn VirtioDevice>>,
}

struct AnyReady<'a> {
    fds: &'a [WatchedFd],
}

impl Future for AnyReady<'_> {
    type Output = usize;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<usize> {
        for (i, wfd) in self.fds.iter().enumerate() {
            if let Poll::Ready(Ok(mut guard)) = wfd.async_fd.poll_read_ready(cx) {
                guard.clear_ready();
                return Poll::Ready(i);
            }
        }
        Poll::Pending
    }
}

// ---------------------------------------------------------------------------
// Pre-seccomp setup
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TokioRuntime {
    pub rt: tokio::runtime::Runtime,
}

pub fn create_runtime() -> TokioRuntime {
    TokioRuntime {
        rt: tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime"),
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

pub fn run_with_api(
    rt: &TokioRuntime,
    vmm: Arc<Mutex<Vmm>>,
    handlers: Vec<FdHandler>,
    from_api: tokio::sync::mpsc::Receiver<ApiRequest>,
    to_api: std::sync::mpsc::Sender<ApiResponse>,
) -> Result<(), FcExitCode> {
    rt.rt.block_on(event_loop(vmm, handlers, Some((from_api, to_api))))
}

pub fn run_no_api(
    rt: &TokioRuntime,
    vmm: Arc<Mutex<Vmm>>,
    handlers: Vec<FdHandler>,
) -> Result<(), FcExitCode> {
    rt.rt.block_on(event_loop(vmm, handlers, None))
}

// ---------------------------------------------------------------------------
// Core event loop
// ---------------------------------------------------------------------------

type ApiChannel = (tokio::sync::mpsc::Receiver<ApiRequest>, std::sync::mpsc::Sender<ApiResponse>);

async fn event_loop(
    vmm: Arc<Mutex<Vmm>>,
    handlers: Vec<FdHandler>,
    api: Option<ApiChannel>,
) -> Result<(), FcExitCode> {
    let mut controller = api.as_ref().map(|_| RuntimeApiController::new(vmm.clone()));
    let mut latency = LatencyStats::default();

    let mut api_rx = None;
    let mut api_tx = None;
    if let Some((rx, tx)) = api {
        api_rx = Some(rx);
        api_tx = Some(tx);
    }

    let exit_fd = vmm.lock().unwrap().vcpus_exit_evt.as_raw_fd();
    let async_exit = AsyncFd::with_interest(BorrowedFd(exit_fd), Interest::READABLE)
        .expect("AsyncFd for exit_evt");

    let watched: Vec<WatchedFd> = handlers
        .into_iter()
        .filter_map(|fh| {
            AsyncFd::with_interest(BorrowedFd(fh.fd), Interest::READABLE)
                .map(|afd| WatchedFd {
                    async_fd: afd,
                    tag: fh.tag,
                    device: fh.device,
                })
                .map_err(|e| warn!("AsyncFd failed for fd {}: {e}", fh.fd))
                .ok()
        })
        .collect();

    let mut metrics_interval = tokio::time::interval(std::time::Duration::from_secs(60));
    let _ = METRICS.write();

    info!("Tokio event loop started: {} device fds", watched.len());

    loop {
        let any_device = AnyReady { fds: &watched };

        tokio::select! {
            biased;

            // vCPU exit
            r = async_exit.readable() => {
                if let Ok(mut g) = r { g.clear_ready(); }
                handle_vcpu_exit(&vmm);
            }

            // API request
            req = async {
                match &mut api_rx {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                if let Some(req) = req {
                    let t = Instant::now();
                    let tx = api_tx.as_ref().unwrap();
                    let ctl = controller.as_mut().unwrap();
                    handle_api_request(req, tx, ctl, &mut api_rx);
                    latency.record_api(t.elapsed().as_nanos() as u64);
                }
            }

            // Metrics
            _ = metrics_interval.tick() => {
                let _ = METRICS.write();
                latency.maybe_report();
            }

            // Device event
            idx = any_device => {
                let t = Instant::now();
                let wfd = &watched[idx];
                wfd.device.lock().unwrap().process_async_event(wfd.tag);
                latency.record_device(t.elapsed().as_nanos() as u64);
            }
        }

        match vmm.lock().unwrap().shutdown_exit_code() {
            Some(FcExitCode::Ok) => return Ok(()),
            Some(code) => return Err(code),
            None => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

fn handle_vcpu_exit(vmm: &Arc<Mutex<Vmm>>) {
    let mut v = vmm.lock().unwrap();
    let _ = v.vcpus_exit_evt.read();
    let exit_code = 'ec: {
        for h in &v.vcpus_handles {
            for r in h.response_receiver().try_iter() {
                if let crate::VcpuResponse::Exited(s) = r {
                    if s != FcExitCode::Ok { break 'ec s; }
                }
            }
        }
        FcExitCode::Ok
    };
    v.stop(exit_code);
}

fn handle_api_request(
    req: ApiRequest,
    tx: &std::sync::mpsc::Sender<ApiResponse>,
    ctl: &mut RuntimeApiController,
    api_rx: &mut Option<tokio::sync::mpsc::Receiver<ApiRequest>>,
) {
    let is_pause = *req == VmmAction::Pause;
    let resp = ctl.handle_request(*req);
    tx.send(Box::new(resp)).expect("API tx closed");

    if is_pause {
        let rx = api_rx.as_mut().unwrap();
        loop {
            let r = rx.blocking_recv().expect("API rx closed in pause");
            let is_resume = *r == VmmAction::Resume;
            let resp = ctl.handle_request(*r);
            tx.send(Box::new(resp)).expect("API tx closed");
            if is_resume { break; }
        }
    }
}
