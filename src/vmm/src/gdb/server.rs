use gdbstub::{
    common::{Signal, Tid},
    conn::{Connection, ConnectionExt},
    stub::{run_blocking, DisconnectReason, GdbStub, MultiThreadStopReason},
    target::Target,
};
use kvm_bindings::{
    kvm_guest_debug, KVM_GUESTDBG_ENABLE, KVM_GUESTDBG_INJECT_BP, KVM_GUESTDBG_SINGLESTEP,
    KVM_GUESTDBG_USE_HW_BP, KVM_GUESTDBG_USE_SW_BP,
};
use kvm_ioctls::VcpuFd;
use std::sync::mpsc::{Receiver, TryRecvError::Empty};
use vm_memory::GuestAddress;

use crate::{
    logger::{error, info},
    Vcpu, Vmm,
};

use std::{
    os::unix::net::{UnixListener, UnixStream},
    path::Path,
    sync::{Arc, Mutex},
};

use super::target::{cpuid_to_tid, FirecrackerTarget};

const KVM_DEBUG_ENABLE: u64 = 0x0600;

fn listen_for_connection(path: &std::path::Path) -> Result<UnixStream, Box<dyn std::error::Error>> {
    info!("Binding gdb socket");
    let listener = UnixListener::bind(path)?;
    info!("Waiting for connection on {}...", path.display());

    let (stream, addr) = listener.accept()?;
    info!("GDB connected from {addr:?}");

    Ok(stream)
}


fn set_kvm_debug(control: u32, vcpu: &VcpuFd, addrs: &[GuestAddress], step: bool) {
    let mut dbg = kvm_guest_debug {
        control,
        ..Default::default()
    };

    dbg.arch.debugreg[7] = KVM_DEBUG_ENABLE;

    for (i, addr) in addrs.iter().enumerate() {
        dbg.arch.debugreg[i] = addr.0;
        // Set global breakpoint enable flag
        dbg.arch.debugreg[7] |= 2 << (i * 2);
    }

    if vcpu.set_guest_debug(&dbg).is_err() {
        error!("Error setting debug");
    } else {
        info!(
            "Debug setup succesfully. Single Step: {step} BP count: {}",
            addrs.len()
        );
    }
}

/// TODO DOCS
pub fn kvm_debug(vcpu: &VcpuFd, addrs: &[GuestAddress], step: bool) {
    let mut control = KVM_GUESTDBG_ENABLE | KVM_GUESTDBG_USE_HW_BP | KVM_GUESTDBG_USE_SW_BP;
    if step {
        control |= KVM_GUESTDBG_SINGLESTEP;
    }

    set_kvm_debug(control, vcpu, addrs, step)
}

/// TODO DOCS
pub fn kvm_inject_bp(vcpu: &VcpuFd, addrs: &[GuestAddress], step: bool) {
    let mut control = KVM_GUESTDBG_ENABLE
        | KVM_GUESTDBG_USE_HW_BP
        | KVM_GUESTDBG_USE_SW_BP
        | KVM_GUESTDBG_INJECT_BP;

    if step {
        control |= KVM_GUESTDBG_SINGLESTEP;
    }

    set_kvm_debug(control, vcpu, addrs, step)
}

/// TODO DOCS
pub fn gdb_thread(
    vmm: Arc<Mutex<Vmm>>,
    vcpu: &[Vcpu],
    gdb_event_fd: Receiver<usize>,
    entry_addr: GuestAddress,
) {
    // We register a hw breakpoint at the entry point to allow setting
    // breakpoints in the kernel setup process
    kvm_debug(&vcpu[0].kvm_vcpu.fd, &[entry_addr], false);

    for cpu in vcpu.iter().skip(1) {
        kvm_debug(&cpu.kvm_vcpu.fd, &[], false);
    }

    let path = Path::new("/tmp/gdb.socket");
    let connection = listen_for_connection(path).unwrap();

    std::thread::Builder::new()
        .name("gdb".into())
        .spawn(move || event_loop(connection, vmm, gdb_event_fd, entry_addr))
        .unwrap();
}

fn event_loop(connection: UnixStream, vmm: Arc<Mutex<Vmm>>, gdb_event_fd: Receiver<usize>, entry_addr: GuestAddress) {
    let target = FirecrackerTarget::new(vmm, gdb_event_fd, entry_addr);
    let connection: Box<dyn ConnectionExt<Error = std::io::Error>> = { Box::new(connection) };
    let debugger = GdbStub::new(connection);

    gdb_event_loop_thread(debugger, target);
}

struct MyGdbBlockingEventLoop {}

impl run_blocking::BlockingEventLoop for MyGdbBlockingEventLoop {
    type Target = FirecrackerTarget;
    type Connection = Box<dyn ConnectionExt<Error = std::io::Error>>;

    type StopReason = MultiThreadStopReason<u64>;

    fn wait_for_stop_reason(
        target: &mut FirecrackerTarget,
        conn: &mut Self::Connection,
    ) -> Result<
        run_blocking::Event<MultiThreadStopReason<u64>>,
        run_blocking::WaitForStopReasonError<
            <Self::Target as Target>::Error,
            <Self::Connection as Connection>::Error,
        >,
    > {
        loop {
            let mut stopped_tid = None;
            match target.gdb_event.try_recv() {
                Ok(cpu_id) => {
                    // The VCPU reports it's id from raw_id so we straight convert here
                    let tid = Tid::new(cpu_id).expect("Error converting cpu id to Tid");
                    // If notify paused returns false this means we were already debugging a single
                    // core, the target will track this for us to pick up later
                    target.notify_paused_vcpu(tid);

                    stopped_tid = Some(tid);
                }
                Err(Empty) => (),
                Err(e) => {
                    info!("Error {e:?}");
                }
            }

            if let Some(tid) = stopped_tid {
                let stop_response = match target.get_stop_reason(tid) {
                    Some(res) => res,
                    None => {
                        // If we returned None this is a break which should be handled by
                        // the guest kernel (e.g. kernel int3 self testing) so we won't notify GDB
                        target.request_resume(tid);
                        continue;
                    }
                };

                return Ok(run_blocking::Event::TargetStopped(stop_response));
            }

            if conn.peek().map(|b| b.is_some()).unwrap_or(false) {
                let byte = conn
                    .read()
                    .map_err(run_blocking::WaitForStopReasonError::Connection)?;
                return Ok(run_blocking::Event::IncomingData(byte));
            }
        }
    }

    // Invoked when the GDB client sends a Ctrl-C interrupt.
    fn on_interrupt(
        target: &mut FirecrackerTarget,
    ) -> Result<Option<MultiThreadStopReason<u64>>, <FirecrackerTarget as Target>::Error> {
        // notify the target that a ctrl-c interrupt has occurred.
        let main_core = cpuid_to_tid(0);

        target.request_pause(main_core);
        target.notify_paused_vcpu(main_core);

        let exit_reason = MultiThreadStopReason::SignalWithThread {
            tid: main_core,
            signal: Signal::SIGINT,
        };
        Ok(Some(exit_reason))
    }
}

fn gdb_event_loop_thread(
    debugger: GdbStub<FirecrackerTarget, Box<dyn ConnectionExt<Error = std::io::Error>>>,
    mut target: FirecrackerTarget,
) {
    match debugger.run_blocking::<MyGdbBlockingEventLoop>(&mut target) {
        Ok(disconnect_reason) => match disconnect_reason {
            DisconnectReason::Disconnect => {
                println!("Client disconnected")
            }
            DisconnectReason::TargetExited(code) => {
                println!("Target exited with code {}", code)
            }
            DisconnectReason::TargetTerminated(sig) => {
                println!("Target terminated with signal {}", sig)
            }
            DisconnectReason::Kill => println!("GDB sent a kill command"),
        },
        Err(e) => {
            if e.is_target_error() {
                println!("target encountered a fatal error:")
            } else if e.is_connection_error() {
                println!("connection error: ")
            } else {
                println!("gdbstub encountered a fatal error {e:?}")
            }
        }
    }

    target.shutdown();
}
