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
/// For block devices with a Tokio file engine, also selects on the
/// completion channel.
fn spawn_device_task(
    device: Arc<DeviceMutex<dyn VirtioDevice>>,
    fd_tags: Vec<(RawFd, u32)>,
    latency: Arc<Mutex<LatencyStats>>,
) {
    // Take the tokio completion receiver before spawning (needs &mut).
    let mut completion_rx = device
        .try_lock()
        .expect("uncontended")
        .take_tokio_completion_rx();

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

    if device_fds.is_empty() && completion_rx.is_none() {
        return;
    }

    tokio::task::spawn_local(async move {
        // Track whether we already have a spawned unblock task pending,
        // so we don't spawn duplicates.
        let mut rl_task_deadline: Option<tokio::time::Instant> = None;

        loop {
            let t;

            tokio::select! {
                biased;

                // Tokio block I/O completion
                Some(completion) = async {
                    match &mut completion_rx {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending().await,
                    }
                } => {
                    t = Instant::now();
                    let mut dev = device.lock().await;
                    dev.process_tokio_completion(completion);
                    if let Some(rx) = &mut completion_rx {
                        while let Ok(c) = rx.try_recv() {
                            dev.process_tokio_completion(c);
                        }
                    }
                    maybe_spawn_rate_limiter_unblock(
                        &dev, &device, &mut rl_task_deadline,
                    );
                }

                // Device fd event
                first_idx = AnyReady { fds: &device_fds }, if !device_fds.is_empty() => {
                    t = Instant::now();
                    let mut dev = device.lock().await;

                    dev.process_async_event(device_fds[first_idx].tag);

                    for (i, dfd) in device_fds.iter().enumerate() {
                        if i == first_idx {
                            continue;
                        }
                        if dfd.async_fd.try_io(Interest::READABLE, |_| {
                            Ok::<(), std::io::Error>(())
                        }).is_ok() {
                            dev.process_async_event(dfd.tag);
                        }
                    }
                    maybe_spawn_rate_limiter_unblock(
                        &dev, &device, &mut rl_task_deadline,
                    );
                }
            }

            if let Ok(mut stats) = latency.try_lock() {
                stats.record_device(t.elapsed().as_nanos() as u64);
            }
        }
    });
}

/// If the device has a rate limiter deadline that we haven't already spawned
/// a task for, spawn a fire-and-forget task that sleeps until the deadline
/// then locks the device and unblocks the rate limiter.
fn maybe_spawn_rate_limiter_unblock(
    dev: &tokio::sync::MutexGuard<'_, dyn VirtioDevice>,
    device: &Arc<DeviceMutex<dyn VirtioDevice>>,
    current_task_deadline: &mut Option<tokio::time::Instant>,
) {
    let new_deadline = dev.rate_limiter_deadline();
    // Only spawn if there's a new deadline we don't already have a task for.
    if new_deadline.is_some() && new_deadline != *current_task_deadline {
        let deadline = new_deadline.unwrap();
        let device = device.clone();
        *current_task_deadline = new_deadline;
        tokio::task::spawn_local(async move {
            tokio::time::sleep_until(deadline).await;
            let mut dev = device.lock().await;
            dev.process_rate_limiter_unblock();
        });
    } else if new_deadline.is_none() {
        *current_task_deadline = None;
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
    serial: Option<SerialHandler>,
    from_api: tokio::sync::mpsc::Receiver<ApiRequest>,
    to_api: tokio::sync::mpsc::Sender<ApiResponse>,
) -> Result<(), FcExitCode> {
    let local = tokio::task::LocalSet::new();
    local.block_on(
        &rt.rt,
        run_event_loop(vmm, handlers, serial, Some((from_api, to_api))),
    )
}

pub fn run_no_api(
    rt: &TokioRuntime,
    vmm: Arc<Mutex<Vmm>>,
    handlers: Vec<FdHandler>,
    serial: Option<SerialHandler>,
) -> Result<(), FcExitCode> {
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt.rt, run_event_loop(vmm, handlers, serial, None))
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
    let latency = Arc::new(Mutex::new(LatencyStats::default()));

    // Group handlers by device and spawn one task per device.
    let device_groups = group_handlers_by_device(handlers);
    let num_devices = device_groups.len();
    let num_fds: usize = device_groups.iter().map(|(_, fds)| fds.len()).sum();
    for (device, fd_tags) in device_groups {
        spawn_device_task(device, fd_tags, latency.clone());
    }

    let _ = METRICS.write();

    // Swap MMIO bus devices to channel-based proxies so vCPU MMIO accesses
    // go through the async event loop instead of locking device mutexes directly.
    let (mmio_rx, _mmio_proxies) = {
        let v = vmm.lock().unwrap();
        v.device_manager.swap_to_mmio_proxies(&v.vm.common.mmio_bus)
    };

    info!(
        "Tokio event loop started: {num_devices} device tasks, {num_fds} device fds"
    );
    info!("MMIO proxies installed on bus");

    // Use a broadcast-style shutdown signal so all tasks can observe it.
    let (shutdown_tx, _) = tokio::sync::broadcast::channel::<FcExitCode>(1);

    // --- Spawn independent tasks ---

    spawn_vcpu_exit_task(vmm.clone(), shutdown_tx.clone());
    spawn_mmio_task(mmio_rx);
    spawn_metrics_task(latency.clone());

    if let Some(serial) = serial {
        spawn_serial_task(serial);
    }

    if let Some((from_api, to_api)) = api {
        spawn_api_task(vmm.clone(), from_api, to_api, latency, shutdown_tx.clone());
    }

    // Wait for shutdown signal from any task.
    let mut shutdown_rx = shutdown_tx.subscribe();
    match shutdown_rx.recv().await {
        Ok(FcExitCode::Ok) => Ok(()),
        Ok(code) => Err(code),
        // All senders dropped without signaling — treat as clean exit.
        Err(_) => Ok(()),
    }
}

fn spawn_vcpu_exit_task(vmm: Arc<Mutex<Vmm>>, shutdown_tx: tokio::sync::broadcast::Sender<FcExitCode>) {
    let exit_fd = vmm.lock().unwrap().vcpus_exit_evt.as_raw_fd();
    let async_exit = AsyncFd::with_interest(BorrowedFd(exit_fd), Interest::READABLE)
        .expect("AsyncFd for exit_evt");

    tokio::task::spawn_local(async move {
        loop {
            if let Ok(mut g) = async_exit.readable().await {
                g.clear_ready();
            }
            handle_vcpu_exit(&vmm);
            if let Some(code) = vmm.lock().unwrap().shutdown_exit_code() {
                let _ = shutdown_tx.send(code);
                return;
            }
        }
    });
}

fn spawn_mmio_task(mut mmio_rx: tokio::sync::mpsc::Receiver<MmioRequest>) {
    tokio::task::spawn_local(async move {
        while let Some(req) = mmio_rx.recv().await {
            handle_mmio_request(req).await;
        }
    });
}

fn spawn_serial_task(serial: SerialHandler) {
    // Serial stdin
    let input_afd = if serial.input_fd >= 0 {
        let is_tty = unsafe { libc::isatty(serial.input_fd) } == 1;
        if is_tty || serial.input_fd != 0 {
            AsyncFd::with_interest(BorrowedFd(serial.input_fd), Interest::READABLE).ok()
        } else {
            None
        }
    } else {
        None
    };

    // Serial buffer-ready
    let buf_ready_afd = if serial.buffer_ready_fd >= 0 {
        AsyncFd::with_interest(BorrowedFd(serial.buffer_ready_fd), Interest::READABLE).ok()
    } else {
        None
    };

    tokio::task::spawn_local(async move {
        loop {
            tokio::select! {
                r = async {
                    match &input_afd {
                        Some(afd) => { let g = afd.readable().await; Some(g) },
                        None => std::future::pending().await,
                    }
                } => {
                    if let Some(Ok(mut g)) = r { g.clear_ready(); }
                    let mut s = serial.serial.lock().unwrap();
                    match s.recv_bytes() {
                        Ok(0) => { info!("Serial stdin EOF"); }
                        Ok(_) => {}
                        Err(e) if e.raw_os_error() == Some(libc::EWOULDBLOCK) => {}
                        Err(e) if e.raw_os_error() == Some(libc::ENOBUFS) => {}
                        Err(_) => {}
                    }
                }

                r = async {
                    match &buf_ready_afd {
                        Some(afd) => { let g = afd.readable().await; Some(g) },
                        None => std::future::pending().await,
                    }
                } => {
                    if let Some(Ok(mut g)) = r { g.clear_ready(); }
                    let mut s = serial.serial.lock().unwrap();
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
    });
}

fn spawn_metrics_task(latency: Arc<Mutex<LatencyStats>>) {
    tokio::task::spawn_local(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        loop {
            interval.tick().await;
            let _ = METRICS.write();
            if let Ok(mut stats) = latency.try_lock() {
                stats.maybe_report();
            }
        }
    });
}

fn spawn_api_task(
    vmm: Arc<Mutex<Vmm>>,
    mut from_api: tokio::sync::mpsc::Receiver<ApiRequest>,
    to_api: tokio::sync::mpsc::Sender<ApiResponse>,
    latency: Arc<Mutex<LatencyStats>>,
    shutdown_tx: tokio::sync::broadcast::Sender<FcExitCode>,
) {
    tokio::task::spawn_local(async move {
        let mut controller = RuntimeApiController::new(vmm.clone());

        while let Some(req) = from_api.recv().await {
            let t = Instant::now();
            let is_pause = *req == VmmAction::Pause;
            let resp = controller.handle_request(*req).await;
            to_api.send(Box::new(resp)).await.expect("API tx closed");

            if is_pause {
                loop {
                    let r = from_api.recv().await.expect("API rx closed in pause");
                    let is_resume = *r == VmmAction::Resume;
                    let resp = controller.handle_request(*r).await;
                    to_api.send(Box::new(resp)).await.expect("API tx closed");
                    if is_resume {
                        break;
                    }
                }
            }

            if let Ok(mut stats) = latency.try_lock() {
                stats.record_api(t.elapsed().as_nanos() as u64);
            }

            // Check for shutdown after each API request.
            if let Some(code) = vmm.lock().unwrap().shutdown_exit_code() {
                let _ = shutdown_tx.send(code);
                return;
            }
        }
    });
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


