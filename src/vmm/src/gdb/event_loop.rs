// Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use gdbstub::{
    common::{Signal, Tid},
    conn::{Connection, ConnectionExt},
    stub::{
        run_blocking::{self, WaitForStopReasonError},
        DisconnectReason, GdbStub, MultiThreadStopReason,
    },
    target::Target,
};
use std::{
    os::unix::net::UnixStream,
    sync::{
        mpsc::{Receiver, TryRecvError::Empty},
        Arc, Mutex,
    },
};
use vm_memory::GuestAddress;

use crate::Vmm;
use crate::logger::trace;

use super::target::{cpuid_to_tid, Error, FirecrackerTarget};

pub fn event_loop(
    connection: UnixStream,
    vmm: Arc<Mutex<Vmm>>,
    gdb_event_receiver: Receiver<usize>,
    entry_addr: GuestAddress,
) {
    let target = FirecrackerTarget::new(vmm, gdb_event_receiver, entry_addr);
    let connection: Box<dyn ConnectionExt<Error = std::io::Error>> = { Box::new(connection) };
    let debugger = GdbStub::new(connection);

    gdb_event_loop_thread(debugger, target);
}

struct GdbBlockingEventLoop {}

impl run_blocking::BlockingEventLoop for GdbBlockingEventLoop {
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
            match target.gdb_event.try_recv() {
                Ok(cpu_id) => {
                    // The VCPU reports it's id from raw_id so we straight convert here
                    let tid = Tid::new(cpu_id).expect("Error converting cpu id to Tid");
                    // If notify paused returns false this means we were already debugging a single
                    // core, the target will track this for us to pick up later
                    target.notify_paused_vcpu(tid);
                    trace!("vcpu: {tid:?} paused from debug exit");

                    let stop_response = match target.get_stop_reason(tid) {
                        Some(res) => res,
                        None => {
                            // If we returned None this is a break which should be handled by
                            // the guest kernel (e.g. kernel int3 self testing) so we won't notify GDB
                            if let Err(e) = target.request_resume(tid) {
                                return Err(WaitForStopReasonError::Target(e));
                            }

                            trace!("Injected BP into guest early exit");
                            continue;
                        }
                    };

                    trace!("Returned stop reason to gdb: {stop_response:?}");

                    return Ok(run_blocking::Event::TargetStopped(stop_response));
                }
                Err(Empty) => (),
                Err(_) => {
                    return Err(WaitForStopReasonError::Target(Error::GdbQueueError));
                }
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

        if target.request_pause(main_core).is_err() {
            return Err(Error::VCPURequestError);
        }

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
    match debugger.run_blocking::<GdbBlockingEventLoop>(&mut target) {
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
