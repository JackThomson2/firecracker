// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Async event loop driven by a single-threaded Tokio runtime.
//! - Each VirtIO device gets its own spawned task watching its fds via AsyncFd
//! - vCPU→device MMIO goes through channels with tokio::sync::Mutex
//! - Device event handlers are async, enabling future async I/O

use std::collections::HashMap;
use std::future::Future;
use std::os::unix::io::{AsRawFd, RawFd};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;

use tokio::io::unix::AsyncFd;
use tokio::io::Interest;

use crate::devices::legacy::SerialDevice;
use crate::devices::virtio::device::VirtioDevice;
use crate::logger::{METRICS, info, warn};
use crate::mmio_proxy::MmioRequest;
use crate::rpc_interface::{ApiRequest, ApiResponse, RuntimeApiController, VmmAction};
use crate::vstate::bus::BusDevice;
use crate::{DeviceMutex, FcExitCode, Vmm};

// ---------------------------------------------------------------------------
// Latency stats (shared across device tasks via Arc)
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct LatencyStats {
    device_samples: Vec<u64>,
    api_samples: Vec<u64>,
    last_report: Option<Instant>,
}

impl LatencyStats {
    pub fn record_device(&mut self, nanos: u64) {
        self.device_samples.push(nanos);
    }
    pub fn record_api(&mut self, nanos: u64) {
        self.api_samples.push(nanos);
    }
    pub fn maybe_report(&mut self) {
        let now = Instant::now();
        if self.last_report.map_or(true, |t| now.duration_since(t).as_secs() >= 10) {
            self.last_report = Some(now);
            Self::report(&mut self.device_samples, "device_io");
            Self::report(&mut self.api_samples, "api");
        }
    }
    fn report(samples: &mut Vec<u64>, label: &str) {
        if samples.is_empty() {
            return;
        }
        samples.sort_unstable();
        let n = samples.len();
        let p50 = samples[n / 2];
        let p99 = samples[(n * 99 / 100).min(n - 1)];
        info!(
            "latency[{label}] n={n} p50={:.1}us p99={:.1}us",
            p50 as f64 / 1000.0,
            p99 as f64 / 1000.0
        );
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
    pub device: Arc<DeviceMutex<dyn VirtioDevice>>,
}

impl std::fmt::Debug for FdHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FdHandler(fd={}, tag={})", self.fd, self.tag)
    }
}

/// Serial stdin handler for the async event loop.
#[derive(Debug)]
pub struct SerialHandler {
    /// The serial device.
    pub serial: Arc<Mutex<SerialDevice>>,
    /// stdin fd (or -1 if none).
    pub input_fd: i32,
    /// buffer-ready eventfd (or -1 if none).
    pub buffer_ready_fd: i32,
}

/// Build serial handler from the VMM's serial device.
pub fn build_serial_handler(vmm: &Vmm) -> Option<SerialHandler> {
    let serial = vmm.device_manager.get_serial_device()?;
    let locked = serial.lock().unwrap();
    let input_fd = locked.serial_input_fd();
    let buffer_ready_fd = locked.buffer_ready_evt_fd();
    drop(locked);
    if input_fd < 0 {
        return None;
    }
    Some(SerialHandler {
        serial,
        input_fd,
        buffer_ready_fd,
    })
}

/// Build fd→handler mappings from all devices.
/// Must be called before seccomp.
pub fn build_device_handlers(vmm: &Vmm) -> Vec<FdHandler> {
    let mut handlers = Vec::new();
    for (_dev_type, _id, device) in vmm.device_manager.collect_virtio_devices() {
        let fd_tags = device.try_lock().expect("device lock").async_fd_tags();
        for (fd, tag) in fd_tags {
            handlers.push(FdHandler {
                fd,
                tag,
                device: device.clone(),
            });
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
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

/// An fd + tag pair for a single device, used inside per-device tasks.
struct DeviceFd {
    async_fd: AsyncFd<BorrowedFd>,
    tag: u32,
}

/// Future that resolves when any fd in the slice becomes readable.
/// Returns the index of the first ready fd.
struct AnyReady<'a> {
    fds: &'a [DeviceFd],
}

impl Future for AnyReady<'_> {
    type Output = usize;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<usize> {
        for (i, dfd) in self.fds.iter().enumerate() {
            if let Poll::Ready(Ok(mut guard)) = dfd.async_fd.poll_read_ready(cx) {
                guard.clear_ready();
                return Poll::Ready(i);
            }
        }
        Poll::Pending
    }
}

// ---------------------------------------------------------------------------
// Per-device task
// ---------------------------------------------------------------------------

/// Group FdHandlers by device (Arc pointer identity) and return
/// (device, vec-of-(fd, tag)) groups.
fn group_handlers_by_device(
    handlers: Vec<FdHandler>,
) -> Vec<(Arc<DeviceMutex<dyn VirtioDevice>>, Vec<(RawFd, u32)>)> {
    let mut map: HashMap<*const DeviceMutex<dyn VirtioDevice>, (Arc<DeviceMutex<dyn VirtioDevice>>, Vec<(RawFd, u32)>)> =
        HashMap::new();
    for fh in handlers {
        let ptr = Arc::as_ptr(&fh.device);
        map.entry(ptr)
            .or_insert_with(|| (fh.device.clone(), Vec::new()))
            .1
            .push((fh.fd, fh.tag));
    }
    map.into_values().collect()
}

/// Spawn a tokio task for a single device that watches all its fds.
/// When any fd fires, the task locks the device once, then processes
/// ALL currently-ready fds in a batch before releasing the lock.
fn spawn_device_task(
    device: Arc<DeviceMutex<dyn VirtioDevice>>,
    fd_tags: Vec<(RawFd, u32)>,
    latency: Arc<Mutex<LatencyStats>>,
) {
    let device_fds: Vec<DeviceFd> = fd_tags
        .into_iter()
        .filter_map(|(fd, tag)| {
            AsyncFd::with_interest(BorrowedFd(fd), Interest::READABLE)
                .map(|afd| DeviceFd {
                    async_fd: afd,
                    tag,
                })
                .map_err(|e| warn!("AsyncFd failed for fd {fd}: {e}"))
                .ok()
        })
        .collect();

    if device_fds.is_empty() {
        return;
    }

    tokio::spawn(async move {
        loop {
            // Wait for at least one fd to become readable.
            let first_idx = AnyReady { fds: &device_fds }.await;

            let t = Instant::now();

            // Lock the device once for the whole batch.
            let mut dev = device.lock().await;

            // Process the fd that woke us.
            dev.process_async_event(device_fds[first_idx].tag);

            // Check remaining fds for readiness and process them too.
            for (i, dfd) in device_fds.iter().enumerate() {
                if i == first_idx {
                    continue;
                }
                // try_io succeeds if the fd is ready, fails with WouldBlock if not.
                if dfd.async_fd.try_io(Interest::READABLE, |_| {
                    Ok::<(), std::io::Error>(())
                }).is_ok() {
                    dev.process_async_event(dfd.tag);
                }
            }

            drop(dev);

            if let Ok(mut stats) = latency.try_lock() {
                stats.record_device(t.elapsed().as_nanos() as u64);
            }
        }
    });
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
    serial: Option<SerialHandler>,
    from_api: tokio::sync::mpsc::Receiver<ApiRequest>,
    to_api: tokio::sync::mpsc::Sender<ApiResponse>,
) -> Result<(), FcExitCode> {
    rt.rt
        .block_on(run_event_loop(vmm, handlers, serial, Some((from_api, to_api))))
}

pub fn run_no_api(
    rt: &TokioRuntime,
    vmm: Arc<Mutex<Vmm>>,
    handlers: Vec<FdHandler>,
    serial: Option<SerialHandler>,
) -> Result<(), FcExitCode> {
    rt.rt.block_on(run_event_loop(vmm, handlers, serial, None))
}

// ---------------------------------------------------------------------------
// Core event loop
// ---------------------------------------------------------------------------

type ApiChannel = (
    tokio::sync::mpsc::Receiver<ApiRequest>,
    tokio::sync::mpsc::Sender<ApiResponse>,
);

pub async fn run_event_loop(
    vmm: Arc<Mutex<Vmm>>,
    handlers: Vec<FdHandler>,
    serial: Option<SerialHandler>,
    api: Option<ApiChannel>,
) -> Result<(), FcExitCode> {
    let mut controller = api.as_ref().map(|_| RuntimeApiController::new(vmm.clone()));
    let latency = Arc::new(Mutex::new(LatencyStats::default()));

    let mut api_rx = None;
    let mut api_tx = None;
    if let Some((rx, tx)) = api {
        api_rx = Some(rx);
        api_tx = Some(tx);
    }

    let exit_fd = vmm.lock().unwrap().vcpus_exit_evt.as_raw_fd();
    let async_exit = AsyncFd::with_interest(BorrowedFd(exit_fd), Interest::READABLE)
        .expect("AsyncFd for exit_evt");

    // Group handlers by device and spawn one task per device.
    let device_groups = group_handlers_by_device(handlers);
    let num_devices = device_groups.len();
    let num_fds: usize = device_groups.iter().map(|(_, fds)| fds.len()).sum();
    for (device, fd_tags) in device_groups {
        spawn_device_task(device, fd_tags, latency.clone());
    }

    let mut metrics_interval = tokio::time::interval(std::time::Duration::from_secs(60));
    let _ = METRICS.write();

    info!(
        "Tokio event loop started: {num_devices} device tasks, {num_fds} device fds"
    );

    // Swap MMIO bus devices to channel-based proxies so vCPU MMIO accesses
    // go through the async event loop instead of locking device mutexes directly.
    let (mut mmio_rx, _mmio_proxies) = {
        let v = vmm.lock().unwrap();
        v.device_manager.swap_to_mmio_proxies(&v.vm.common.mmio_bus)
    };
    info!("MMIO proxies installed on bus");

    // Set up serial stdin watching
    let serial_input_afd = serial.as_ref().and_then(|sh| {
        if sh.input_fd >= 0 {
            // SAFETY: isatty has no invariants. If fd is invalid, returns 0.
            let is_tty = unsafe { libc::isatty(sh.input_fd) } == 1;
            if is_tty || sh.input_fd != 0 {
                AsyncFd::with_interest(BorrowedFd(sh.input_fd), Interest::READABLE).ok()
            } else {
                None
            }
        } else {
            None
        }
    });
    let serial_buf_ready_afd = serial.as_ref().and_then(|sh| {
        if sh.buffer_ready_fd >= 0 {
            AsyncFd::with_interest(BorrowedFd(sh.buffer_ready_fd), Interest::READABLE).ok()
        } else {
            None
        }
    });

    loop {
        tokio::select! {
            biased;

            // vCPU exit
            r = async_exit.readable() => {
                if let Ok(mut g) = r { g.clear_ready(); }
                handle_vcpu_exit(&vmm);
            }

            // MMIO request from vCPU thread
            Some(req) = mmio_rx.recv() => {
                handle_mmio_request(req).await;
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
                    handle_api_request(req, tx, ctl, &mut api_rx).await;
                    if let Ok(mut stats) = latency.try_lock() {
                        stats.record_api(t.elapsed().as_nanos() as u64);
                    }
                }
            }

            // Metrics
            _ = metrics_interval.tick() => {
                let _ = METRICS.write();
                if let Ok(mut stats) = latency.try_lock() {
                    stats.maybe_report();
                }
            }

            // Serial stdin readable
            r = async {
                match &serial_input_afd {
                    Some(afd) => { let g = afd.readable().await; Some(g) },
                    None => std::future::pending().await,
                }
            } => {
                if let Some(Ok(mut g)) = r { g.clear_ready(); }
                if let Some(sh) = &serial {
                    let mut s = sh.serial.lock().unwrap();
                    match s.recv_bytes() {
                        Ok(0) => { info!("Serial stdin EOF"); }
                        Ok(_) => {}
                        Err(e) if e.raw_os_error() == Some(libc::EWOULDBLOCK) => {}
                        Err(e) if e.raw_os_error() == Some(libc::ENOBUFS) => {}
                        Err(_) => {}
                    }
                }
            }

            // Serial buffer-ready event
            r = async {
                match &serial_buf_ready_afd {
                    Some(afd) => { let g = afd.readable().await; Some(g) },
                    None => std::future::pending().await,
                }
            } => {
                if let Some(Ok(mut g)) = r { g.clear_ready(); }
                if let Some(sh) = &serial {
                    let mut s = sh.serial.lock().unwrap();
                    let _ = s.consume_buffer_ready_event();
                    match s.recv_bytes() {
                        Ok(0) => { info!("Serial stdin EOF on buffer-ready"); }
                        Ok(_) => {}
                        Err(e) if e.raw_os_error() == Some(libc::EWOULDBLOCK) => {}
                        Err(e) if e.raw_os_error() == Some(libc::ENOBUFS) => {}
                        Err(_) => {}
                    }
                }
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

async fn handle_mmio_request(req: MmioRequest) {
    match req {
        MmioRequest::Read {
            device,
            base,
            offset,
            len,
            reply,
        } => {
            let mut buf = vec![0u8; len];
            device.lock().await.read(base, offset, &mut buf);
            let _ = reply.send(buf);
        }
        MmioRequest::Write {
            device,
            base,
            offset,
            data,
            reply,
        } => {
            let barrier = device.lock().await.write(base, offset, &data);
            let _ = reply.send(barrier);
        }
    }
}

fn handle_vcpu_exit(vmm: &Arc<Mutex<Vmm>>) {
    let mut v = vmm.lock().unwrap();
    let _ = v.vcpus_exit_evt.read();
    let exit_code = 'ec: {
        for h in &mut v.vcpus_handles {
            loop {
                match h.response_receiver_mut().try_recv() {
                    Ok(crate::VcpuResponse::Exited(s)) if s != FcExitCode::Ok => break 'ec s,
                    Ok(_) => continue,
                    Err(_) => break,
                }
            }
        }
        FcExitCode::Ok
    };
    v.stop(exit_code);
}

async fn handle_api_request(
    req: ApiRequest,
    tx: &tokio::sync::mpsc::Sender<ApiResponse>,
    ctl: &mut RuntimeApiController,
    api_rx: &mut Option<tokio::sync::mpsc::Receiver<ApiRequest>>,
) {
    let is_pause = *req == VmmAction::Pause;
    let resp = ctl.handle_request(*req).await;
    tx.send(Box::new(resp)).await.expect("API tx closed");

    if is_pause {
        let rx = api_rx.as_mut().unwrap();
        loop {
            let r = rx.recv().await.expect("API rx closed in pause");
            let is_resume = *r == VmmAction::Resume;
            let resp = ctl.handle_request(*r).await;
            tx.send(Box::new(resp)).await.expect("API tx closed");
            if is_resume {
                break;
            }
        }
    }
}
