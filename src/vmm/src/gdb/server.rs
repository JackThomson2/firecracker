use event_manager::{EventOps, EventSet, Events, MutEventSubscriber, SubscriberOps};
use gdbstub::{arch::Arch, common::Signal, conn::{Connection, ConnectionExt}, stub::{run_blocking, DisconnectReason, GdbStub, SingleThreadStopReason}, target::{ext::base::singlethread::SingleThreadSingleStep, Target}};
use kvm_bindings::{kvm_guest_debug, kvm_guest_debug_arch, KVM_GUESTDBG_ENABLE, KVM_GUESTDBG_INJECT_BP, KVM_GUESTDBG_SINGLESTEP, KVM_GUESTDBG_USE_HW_BP, KVM_GUESTDBG_USE_SW_BP};
use kvm_ioctls::VcpuFd;
use utils::eventfd::EventFd;
use vm_memory::GuestAddress;
use std::{io::ErrorKind, os::unix::io::AsRawFd, sync::mpsc::{channel, Receiver}};

use crate::{logger::{error, info, warn, MetricsError, METRICS}, vstate::vcpu::{self, KvmVcpu}, EventManager, Vcpu, VcpuEvent, VcpuResponse, Vmm};

use std::{io, os::unix::net::{UnixListener, UnixStream}, path::Path, rc::Rc, sync::{mpsc::Sender, Arc, Mutex}, thread, time::Duration};

use super::target::{FirecrackerTarget};

fn listen_for_connection(path: &std::path::Path) -> Result<UnixStream, Box<dyn std::error::Error>> {
    info!("Binding gdb socket");
    let listener = UnixListener::bind(path)?;
    info!("Waiting for connection on {}...", path.display());

    let (stream, addr) = listener.accept()?;
    info!("GDB connected from {addr:?}");

    return Ok(stream)
}

fn event_loop(connection: UnixStream, vmm: Arc<Mutex<Vmm>>, gdb_event_fd: EventFd) {
    let target = FirecrackerTarget::new(vmm, gdb_event_fd);
    let connection: Box<dyn ConnectionExt<Error = std::io::Error>> = {
        Box::new(connection)
    };
    let debugger = GdbStub::new(connection);

    gdb_event_loop_thread(debugger, target);
}

pub fn kvm_debug(vcpu: &VcpuFd, addrs: &[GuestAddress], step: bool) {
    let mut control = KVM_GUESTDBG_ENABLE | KVM_GUESTDBG_USE_HW_BP | KVM_GUESTDBG_USE_SW_BP;
    if step {
        control |= KVM_GUESTDBG_SINGLESTEP;
    }
    let mut dbg = kvm_guest_debug {
        control,
        ..Default::default()
    };

     dbg.arch.debugreg[7] = 0x0600;

    for (i, addr) in addrs.iter().enumerate() {
        dbg.arch.debugreg[i] = addr.0;
        // Set global breakpoint enable flag
        dbg.arch.debugreg[7] |= 2 << (i * 2);
    }

    if let Err(_) = vcpu.set_guest_debug(&dbg) {
        error!("Error setting debug");
    } else {
        info!("Debug setup succesfully. Single Step: {step} BP count: {}", addrs.len())
    }
}

pub fn kvm_inject_bp(vcpu: &VcpuFd, addrs: &[GuestAddress], step: bool) {
    let mut control = KVM_GUESTDBG_ENABLE | KVM_GUESTDBG_USE_HW_BP | KVM_GUESTDBG_USE_SW_BP | KVM_GUESTDBG_INJECT_BP;
    if step {
        control |= KVM_GUESTDBG_SINGLESTEP;
    }
    let mut dbg = kvm_guest_debug {
        control,
        ..Default::default()
    };

     dbg.arch.debugreg[7] = 0x0600;

    for (i, addr) in addrs.iter().enumerate() {
        dbg.arch.debugreg[i] = addr.0;
        // Set global breakpoint enable flag
        dbg.arch.debugreg[7] |= 2 << (i * 2);
    }

    if let Err(_) = vcpu.set_guest_debug(&dbg) {
        error!("Error setting debug");
    } else {
        info!("Injected breakpoint to guest. Single Step: {step} BP count: {}", addrs.len())
    }
}

pub fn gdb_thread(vmm: Arc<Mutex<Vmm>>, vcpu: &Vec<Vcpu>, gdb_event_fd: EventFd, entry_addr: GuestAddress) {
    kvm_debug(&vcpu[0].kvm_vcpu.fd, &vec![entry_addr], false);

    let path = Path::new("/tmp/gdb.socket");
    let connection = listen_for_connection(path).unwrap();

    std::thread::Builder::new()
        .name("gdb".into())
        .spawn(move || event_loop(connection, vmm, gdb_event_fd))
        .unwrap();
}

struct MyGdbBlockingEventLoop {}

// The `run_blocking::BlockingEventLoop` groups together various callbacks
// the `GdbStub::run_blocking` event loop requires you to implement.
impl run_blocking::BlockingEventLoop for MyGdbBlockingEventLoop {
    type Target = FirecrackerTarget;
    type Connection = Box<dyn ConnectionExt<Error = std::io::Error>>;

    // or MultiThreadStopReason on multi threaded targets
    type StopReason = SingleThreadStopReason<u64>;

    // Invoked immediately after the target's `resume` method has been
    // called. The implementation should block until either the target
    // reports a stop reason, or if new data was sent over the connection.
    fn wait_for_stop_reason(
        target: &mut FirecrackerTarget,
        conn: &mut Self::Connection,
    ) -> Result<
        run_blocking::Event<SingleThreadStopReason<u64>>,
        run_blocking::WaitForStopReasonError<
            <Self::Target as Target>::Error,
            <Self::Connection as Connection>::Error,
        >,
    > {
        loop {
            match target.gdb_event.read() {
                Ok(_) => {
                    // thread::sleep(Duration::from_secs(1));
                    info!("Got notification from gdb eventfd ensuring cpu paused");
                    target.request_pause();

                    let stop_resonse = match target.get_stop_reason() {
                        Some(res) => res,
                        None => {
                            target.request_resume();
                            continue;
                        }
                    };

                    return Ok(run_blocking::Event::TargetStopped(stop_resonse));
                },
                Err(e) => {
                    if e.kind() != ErrorKind::WouldBlock {
                        info!("Error {e:?}")
                    }

                }
            }

            if conn.peek().map(|b| b.is_some()).unwrap_or(true) {
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
    ) -> Result<Option<SingleThreadStopReason<u64>>, <FirecrackerTarget as Target>::Error> {
        // notify the target that a ctrl-c interrupt has occurred.

        // a pretty typical stop reason in response to a Ctrl-C interrupt is to
        // report a "Signal::SIGINT".
        Ok(Some(SingleThreadStopReason::Signal(Signal::SIGINT).into()))
    }
}

fn gdb_event_loop_thread(
    debugger: GdbStub<FirecrackerTarget, Box<dyn ConnectionExt<Error = std::io::Error>>>,
    mut target: FirecrackerTarget 
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
                println!(
                    "target encountered a fatal error:" 
                )
            } else if e.is_connection_error() {
                println!("connection error: ")
            } else {
                println!("gdbstub encountered a fatal error {e:?}")
            }
        }
    }
}
