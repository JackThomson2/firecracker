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

use tokio::io::unix::AsyncFd;
use tokio::io::Interest;

use crate::devices::legacy::SerialDevice;
use crate::devices::virtio::device::VirtioDevice;
use crate::logger::{METRICS, info};
use crate::rpc_interface::{ApiRequest, ApiResponse, RuntimeApiController, VmmAction};
use crate::{DeviceMutex, FcExitCode, Vmm};

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
    let devices = vmm.device_manager.collect_virtio_devices();
    for (_dev_type, _id, device) in devices {
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

fn make_device_fds(fd_tags: Vec<(RawFd, u32)>) -> Vec<DeviceFd> {
    fd_tags
        .into_iter()
        .filter_map(|(fd, tag)| {
            AsyncFd::with_interest(BorrowedFd(fd), Interest::READABLE)
                .map(|afd| DeviceFd { async_fd: afd, tag })
                .ok()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Common async helpers
// ---------------------------------------------------------------------------

/// Wait for an AsyncFd to become readable, then clear the ready flag.
async fn fd_readable(afd: &AsyncFd<BorrowedFd>) {
    if let Ok(mut guard) = afd.readable().await {
        guard.clear_ready();
    }
}

/// Wait for an optional AsyncFd to become readable.
/// If the fd is None, pends forever (never resolves).
async fn optional_fd_readable(afd: &Option<AsyncFd<BorrowedFd>>) {
    match afd {
        Some(afd) => fd_readable(afd).await,
        None => std::future::pending().await,
    }
}

/// Wait while the VM is paused.
/// Returns `true` when resumed, `false` if the channel closed (caller should exit).
async fn wait_while_paused(pause_rx: &mut tokio::sync::watch::Receiver<bool>) -> bool {
    while *pause_rx.borrow() {
        if pause_rx.changed().await.is_err() {
            return false;
        }
    }
    true
}

/// Rate limiter timer wrapping an optional tokio sleep future.
///
/// Provides a clean `.wait()` / `.update()` API instead of the raw
/// `async { match &mut rl_sleep { Some(s) => ..., None => pending() } }` pattern.
struct RateLimiterTimer {
    sleep: Option<Pin<Box<tokio::time::Sleep>>>,
}

impl RateLimiterTimer {
    fn new() -> Self {
        Self { sleep: None }
    }

    /// Set the timer from a rate limiter deadline (if any).
    fn update(&mut self, deadline: Option<tokio::time::Instant>) {
        if let Some(d) = deadline {
            self.sleep = Some(Box::pin(tokio::time::sleep_until(d)));
        }
    }

    /// Wait for the rate limiter timer to fire.
    /// If no timer is set, pends forever (never resolves).
    async fn wait(&mut self) {
        match &mut self.sleep {
            Some(s) => s.as_mut().await,
            None => std::future::pending().await,
        }
        self.sleep = None;
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
    let mut map: HashMap<*const (), (Arc<DeviceMutex<dyn VirtioDevice>>, Vec<(RawFd, u32)>)> =
        HashMap::new();
    for fh in handlers {
        let ptr = Arc::as_ptr(&fh.device) as *const ();
        map.entry(ptr)
            .or_insert_with(|| (fh.device.clone(), Vec::new()))
            .1
            .push((fh.fd, fh.tag));
    }
    map.into_values().collect()
}

/// Process the first ready fd and drain any other ready fds for the same device.
fn process_all_ready_fds(
    dev: &mut dyn VirtioDevice,
    device_fds: &[DeviceFd],
    first_idx: usize,
) {
    dev.process_async_event(device_fds[first_idx].tag);
    for (i, dfd) in device_fds.iter().enumerate() {
        if i == first_idx {
            continue;
        }
        if dfd
            .async_fd
            .try_io(Interest::READABLE, |_| Ok::<(), std::io::Error>(()))
            .is_ok()
        {
            dev.process_async_event(dfd.tag);
        }
    }
}

/// Core event loop for a generic (non-net) VirtIO device.
///
/// Watches all device fds via `AnyReady`, locks the device per event batch,
/// and handles rate limiter unblock timers.
async fn run_generic_device_loop(
    device: &DeviceMutex<dyn VirtioDevice>,
    device_fds: &[DeviceFd],
    pause_rx: &mut tokio::sync::watch::Receiver<bool>,
) {
    let mut rl_timer = RateLimiterTimer::new();

    loop {
        if !wait_while_paused(pause_rx).await {
            return;
        }

        tokio::select! {
            biased;

            _ = pause_rx.changed() => { continue; }

            idx = AnyReady { fds: device_fds }, if !device_fds.is_empty() => {
                let mut dev = device.lock().await;
                process_all_ready_fds(&mut *dev, device_fds, idx);
                rl_timer.update(dev.rate_limiter_deadline());
            }

            _ = rl_timer.wait() => {
                let mut dev = device.lock().await;
                dev.process_rate_limiter_unblock();
                rl_timer.update(dev.rate_limiter_deadline());
            }
        }
    }
}

/// Spawn a tokio task for a single device that watches all its fds.
///
/// Locks the device per-event-batch: when any fd fires, acquires the lock,
/// processes ALL currently-ready fds, then releases. This allows vCPU threads
/// to acquire the same device lock for MMIO config reads between events.
///
/// Note: lock-once (holding the guard across the entire select! loop) is NOT
/// possible here because the MMIO transport's `locked_device()` acquires the
/// same `DeviceMutex` from vCPU threads. Holding it permanently would deadlock.
/// Net devices avoid this by splitting into RX/TX tasks with separate mutexes.
fn spawn_device_task(
    device: Arc<DeviceMutex<dyn VirtioDevice>>,
    fd_tags: Vec<(RawFd, u32)>,
    mut pause_rx: tokio::sync::watch::Receiver<bool>,
) {
    if fd_tags.is_empty() {
        return;
    }

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("device runtime");

        rt.block_on(async move {
            let device_fds = make_device_fds(fd_tags);
            run_generic_device_loop(&device, &device_fds, &mut pause_rx).await;
        });
    });
}

/// Spawn a dedicated task for a net device. The task waits for the device
/// to be activated (via `Notify`), then splits into RX/TX tasks.
fn spawn_net_activation_task(
    device: Arc<DeviceMutex<dyn VirtioDevice>>,
    notify: Arc<tokio::sync::Notify>,
    pause_rx: tokio::sync::watch::Receiver<bool>,
) {
    tokio::task::spawn_local(async move {
        notify.notified().await;

        let mut dev = device.lock().await;
        if let Some(split) = dev.take_net_split_info() {
            drop(dev);
            spawn_net_rx_task(split.clone(), pause_rx.clone());
            spawn_net_tx_task(split, pause_rx);
            info!("Net device split into RX/TX tasks");
        } else {
            log::error!("Net device activated but take_net_split_info returned None");
        }
    });
}

// ---------------------------------------------------------------------------
// Split net RX/TX tasks
// ---------------------------------------------------------------------------

/// Core event loop for the RX half of a net device.
///
/// Watches: TAP fd (readable), RX queue eventfd, MMDS notify, rate limiter timer.
/// Holds the RX mutex for the entire active period, only releasing on pause.
async fn run_net_rx_loop(
    split: &crate::devices::virtio::net::device::NetSplitInfo,
    pause_rx: &mut tokio::sync::watch::Receiver<bool>,
) {
    let rx_queue_afd =
        AsyncFd::with_interest(BorrowedFd(split.rx_queue_evt_fd), Interest::READABLE)
            .expect("AsyncFd for net RX queue evt");

    let tap_afd = AsyncFd::with_interest(BorrowedFd(split.tap_fd), Interest::READABLE)
        .expect("AsyncFd for net TAP");

    let mmds_notify = split.rx.lock().await.mmds_rx_notify.clone();

    loop {
        if !wait_while_paused(pause_rx).await {
            return;
        }

        let mut rx = split.rx.lock().await;
        let mut rl_timer = RateLimiterTimer::new();

        loop {
            tokio::select! {
                biased;

                _ = pause_rx.changed() => { break; }

                _ = fd_readable(&tap_afd) => {
                    rx.process_tap_rx_event();
                    rl_timer.update(rx.rate_limiter_deadline());
                }

                _ = mmds_notify.notified() => {
                    rx.process_mmds_event();
                    rl_timer.update(rx.rate_limiter_deadline());
                }

                _ = fd_readable(&rx_queue_afd) => {
                    rx.process_rx_queue_event();
                    rl_timer.update(rx.rate_limiter_deadline());
                }

                _ = rl_timer.wait() => {
                    rx.process_rate_limiter_unblock();
                    rl_timer.update(rx.rate_limiter_deadline());
                }
            }
        }
    }
}

/// Spawn a dedicated RX thread+runtime for a net device.
fn spawn_net_rx_task(
    split: std::sync::Arc<crate::devices::virtio::net::device::NetSplitInfo>,
    mut pause_rx: tokio::sync::watch::Receiver<bool>,
) {
    let result = std::thread::Builder::new()
        .name("net-rx".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("net RX runtime");

            rt.block_on(run_net_rx_loop(&split, &mut pause_rx));
        });
    if let Err(e) = result {
        log::error!("Failed to spawn net-rx thread: {e}");
    }
}

/// Core event loop for the TX half of a net device.
///
/// Watches: TX queue eventfd, rate limiter timer.
/// Holds the TX mutex for the entire active period, only releasing on pause.
async fn run_net_tx_loop(
    split: &crate::devices::virtio::net::device::NetSplitInfo,
    pause_rx: &mut tokio::sync::watch::Receiver<bool>,
) {
    let tx_queue_afd =
        AsyncFd::with_interest(BorrowedFd(split.tx_queue_evt_fd), Interest::READABLE)
            .expect("AsyncFd for net TX queue evt");

    loop {
        if !wait_while_paused(pause_rx).await {
            return;
        }

        let mut tx = split.tx.lock().await;
        let mut rl_timer = RateLimiterTimer::new();

        loop {
            tokio::select! {
                biased;

                _ = pause_rx.changed() => { break; }

                _ = fd_readable(&tx_queue_afd) => {
                    tx.process_tx_queue_event();
                    rl_timer.update(tx.rate_limiter_deadline());
                }

                _ = rl_timer.wait() => {
                    tx.process_rate_limiter_unblock();
                    rl_timer.update(tx.rate_limiter_deadline());
                }
            }
        }
    }
}

/// Spawn a dedicated TX thread+runtime for a net device.
fn spawn_net_tx_task(
    split: std::sync::Arc<crate::devices::virtio::net::device::NetSplitInfo>,
    mut pause_rx: tokio::sync::watch::Receiver<bool>,
) {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("net TX runtime");

        rt.block_on(run_net_tx_loop(&split, &mut pause_rx));
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

/// Spawn device tasks for all handlers. Call this before resume_vm
/// so device tasks are ready to process events when vCPUs start.
pub fn spawn_device_tasks_from_handlers(
    handlers: Vec<FdHandler>,
    pause_rx: &tokio::sync::watch::Receiver<bool>,
) {
    let device_groups = group_handlers_by_device(handlers);
    for (device, fd_tags) in device_groups {
        let dev = device.try_lock().expect("uncontended");
        if let Some(notify) = dev.activate_notify() {
            drop(dev);
            spawn_net_activation_task(device, notify, pause_rx.clone());
        } else {
            drop(dev);
            spawn_device_task(device, fd_tags, pause_rx.clone());
        }
    }
}

pub async fn run_event_loop(
    vmm: Arc<Mutex<Vmm>>,
    handlers: Vec<FdHandler>,
    serial: Option<SerialHandler>,
    api: Option<ApiChannel>,
) -> Result<(), FcExitCode> {
    // Pause/resume channel: device tasks watch this to pause when VM is paused.
    let (pause_tx, pause_rx) = tokio::sync::watch::channel(false);

    // Spawn device tasks — each on its own thread with a dedicated tokio runtime.
    spawn_device_tasks_from_handlers(handlers, &pause_rx);

    let _ = METRICS.write();

    // Use a broadcast-style shutdown signal so all tasks can observe it.
    let (shutdown_tx, _) = tokio::sync::broadcast::channel::<FcExitCode>(1);

    // --- Spawn independent tasks ---

    spawn_vcpu_exit_task(vmm.clone(), shutdown_tx.clone());
    spawn_metrics_task();

    if let Some(serial) = serial {
        spawn_serial_task(serial);
    }

    if let Some((from_api, to_api)) = api {
        spawn_api_task(vmm.clone(), from_api, to_api, shutdown_tx.clone(), pause_tx);
    }

    // Wait for shutdown signal from any task.
    let mut shutdown_rx = shutdown_tx.subscribe();
    match shutdown_rx.recv().await {
        Ok(FcExitCode::Ok) => Ok(()),
        Ok(code) => Err(code),
        Err(_) => Ok(()),
    }
}

fn spawn_vcpu_exit_task(
    vmm: Arc<Mutex<Vmm>>,
    shutdown_tx: tokio::sync::broadcast::Sender<FcExitCode>,
) {
    let exit_fd = vmm.lock().unwrap().vcpus_exit_evt.as_raw_fd();
    let async_exit = AsyncFd::with_interest(BorrowedFd(exit_fd), Interest::READABLE)
        .expect("AsyncFd for exit_evt");

    tokio::task::spawn_local(async move {
        loop {
            fd_readable(&async_exit).await;
            handle_vcpu_exit(&vmm);
            if let Some(code) = vmm.lock().unwrap().shutdown_exit_code() {
                let _ = shutdown_tx.send(code);
                return;
            }
        }
    });
}

fn spawn_serial_task(serial: SerialHandler) {
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

    let buf_ready_afd = if serial.buffer_ready_fd >= 0 {
        AsyncFd::with_interest(BorrowedFd(serial.buffer_ready_fd), Interest::READABLE).ok()
    } else {
        None
    };

    tokio::task::spawn_local(async move {
        loop {
            let buf_ready_fired = tokio::select! {
                _ = optional_fd_readable(&input_afd) => false,
                _ = optional_fd_readable(&buf_ready_afd) => true,
            };

            let mut s = serial.serial.lock().unwrap();
            if buf_ready_fired {
                let _ = s.consume_buffer_ready_event();
            }
            match s.recv_bytes() {
                Ok(0) => info!("Serial stdin EOF"),
                Ok(_) => {}
                Err(e) if e.raw_os_error() == Some(libc::EWOULDBLOCK) => {}
                Err(e) if e.raw_os_error() == Some(libc::ENOBUFS) => {}
                Err(_) => {}
            }
        }
    });
}

fn spawn_metrics_task() {
    tokio::task::spawn_local(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            let _ = METRICS.write();
        }
    });
}

fn spawn_api_task(
    vmm: Arc<Mutex<Vmm>>,
    mut from_api: tokio::sync::mpsc::Receiver<ApiRequest>,
    to_api: tokio::sync::mpsc::Sender<ApiResponse>,
    shutdown_tx: tokio::sync::broadcast::Sender<FcExitCode>,
    pause_tx: tokio::sync::watch::Sender<bool>,
) {
    tokio::task::spawn_local(async move {
        let mut controller = RuntimeApiController::new(vmm.clone());

        while let Some(req) = from_api.recv().await {
            let is_pause = *req == VmmAction::Pause;
            let resp = controller.handle_request(*req).await;
            to_api.send(Box::new(resp)).await.expect("API tx closed");

            if is_pause {
                // Tell device tasks to pause.
                let _ = pause_tx.send(true);
                loop {
                    let r = from_api.recv().await.expect("API rx closed in pause");
                    let is_resume = *r == VmmAction::Resume;
                    let resp = controller.handle_request(*r).await;
                    to_api.send(Box::new(resp)).await.expect("API tx closed");
                    if is_resume {
                        // Tell device tasks to resume.
                        let _ = pause_tx.send(false);
                        break;
                    }
                }
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
