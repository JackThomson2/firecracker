// Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use kvm_bindings::{
    kvm_guest_debug, KVM_GUESTDBG_ENABLE, KVM_GUESTDBG_INJECT_BP, KVM_GUESTDBG_SINGLESTEP,
    KVM_GUESTDBG_USE_HW_BP, KVM_GUESTDBG_USE_SW_BP,
};
use kvm_ioctls::VcpuFd;
use std::sync::mpsc::Receiver;
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

use super::event_loop::event_loop;

const KVM_DEBUG_ENABLE: u64 = 0x0600;

fn listen_for_connection(path: &std::path::Path) -> Result<UnixStream, Box<dyn std::error::Error>> {
    let listener = UnixListener::bind(path)?;
    info!("Waiting for GDB server connection on {}...", path.display());
    let (stream, _addr) = listener.accept()?;

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

/// Configures the VCPU for
pub fn kvm_debug(vcpu: &VcpuFd, addrs: &[GuestAddress], step: bool) {
    let mut control = KVM_GUESTDBG_ENABLE | KVM_GUESTDBG_USE_HW_BP | KVM_GUESTDBG_USE_SW_BP;
    if step {
        control |= KVM_GUESTDBG_SINGLESTEP;
    }

    set_kvm_debug(control, vcpu, addrs, step)
}

/// Injects a BP back into the guest kernel for it to handle, this is particularly useful for the
/// kernels selftesting which can happen during boot.
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

/// This method will kickstart the GDB debugging process, it takes in the VMM object, a slice of
/// the paused Vcpu's, the GDB event queue which is used as a mechanism for the VCPU's to notify
/// our GDB thread that they've been paused, then finally the entry address of the kernel.
///
/// Firstly the function will start by configuring the vcpus with KVM for debugging
///
/// This will then create the GDB socket which will be used for communication to the GDB process.
/// After creating this, the function will block while waiting for GDB to connect.
///
/// After the connection has been established the function will start a new thread for handling
/// communcation to the GDB server
pub fn gdb_thread(
    vmm: Arc<Mutex<Vmm>>,
    vcpus: &[Vcpu],
    gdb_event_receiver: Receiver<usize>,
    entry_addr: GuestAddress,
) {
    // We register a hw breakpoint at the entry point as GDB expects the application
    // to be stopped as it connects. This also allows us to set breakpoints before kernel starts
    kvm_debug(&vcpus[0].kvm_vcpu.fd, &[entry_addr], false);

    for vcpu in vcpus.iter().skip(1) {
        kvm_debug(&vcpu.kvm_vcpu.fd, &[], false);
    }

    let path = Path::new("/tmp/gdb.socket");
    let connection = listen_for_connection(path).expect("Error waiting for connection");

    std::thread::Builder::new()
        .name("gdb".into())
        .spawn(move || event_loop(connection, vmm, gdb_event_receiver, entry_addr))
        .expect("Error spawning GDB thread");
}
