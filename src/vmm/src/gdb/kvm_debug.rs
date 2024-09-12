// Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

/// Modified functions from `kvm-ioctls` to take `RawFd` instead of a `VcpuFd`
use std::os::fd::RawFd;

use kvm_bindings::*;
use kvm_ioctls::kvm_ioctls::*;
use vm_memory::GuestAddress;
use vmm_sys_util::errno;
use vmm_sys_util::ioctl::{ioctl_with_mut_ref, ioctl_with_ref};

use super::target::Error;

/// Sets the 9th (Global Exact Breakpoint enable) and the 10th (always 1) bits for the DR7 debug
/// control register
const X86_GLOBAL_DEBUG_ENABLE: u64 = 0b11 << 9;

/// Retrieves the last error and stores it in a target Error
fn last_error() -> Error {
    Error::KvmIoctlsError(errno::Error::last())
}

/// Returns the vCPU general purpose registers.
///
/// See: https://github.com/rust-vmm/kvm-ioctls/blob/ae039f245db0cd40ac20c32049ec536222b6dea7/src/ioctls/vcpu.rs#L226
#[cfg(target_arch = "x86_64")]
pub fn get_regs(vcpu_fd: &RawFd) -> Result<kvm_regs, Error> {
    let mut regs = kvm_regs::default();
    // SAFETY: Safe because we know that our file is a vCPU fd, we know the kernel will only
    // read the correct amount of memory from our pointer, and we verify the return result.
    let ret = unsafe { ioctl_with_mut_ref(vcpu_fd, KVM_GET_REGS(), &mut regs) };
    if ret != 0 {
        return Err(last_error());
    }
    Ok(regs)
}

/// Sets the vCPU general purpose registers using the `KVM_SET_REGS` ioctl.
///
/// See: https://github.com/rust-vmm/kvm-ioctls/blob/ae039f245db0cd40ac20c32049ec536222b6dea7/src/ioctls/vcpu.rs#L342
#[cfg(target_arch = "x86_64")]
pub fn set_regs(vcpu_fd: &RawFd, regs: &kvm_regs) -> Result<(), Error> {
    // SAFETY: Safe because we know that our file is a vCPU fd, we know the kernel will only
    // read the correct amount of memory from our pointer, and we verify the return result.
    let ret = unsafe { ioctl_with_ref(vcpu_fd, KVM_SET_REGS(), regs) };
    if ret != 0 {
        return Err(last_error());
    }
    Ok(())
}

/// Translates a virtual address according to the vCPU's current address translation mode.
///
/// See: https://github.com/rust-vmm/kvm-ioctls/blob/ae039f245db0cd40ac20c32049ec536222b6dea7/src/ioctls/vcpu.rs#L1691
#[cfg(target_arch = "x86_64")]
pub fn translate_gva(vcpu_fd: &RawFd, gva: u64) -> Result<u64, Error> {
    let mut tr = kvm_translation {
        linear_address: gva,
        ..Default::default()
    };

    // SAFETY: Safe because we know that our file is a vCPU fd, we know the kernel will only
    // write the correct amount of memory to our pointer, and we verify the return result.
    let ret = unsafe { ioctl_with_mut_ref(vcpu_fd, KVM_TRANSLATE(), &mut tr) };
    if ret != 0 {
        return Err(last_error());
    }

    if tr.valid == 0 {
        return Err(Error::KvmGvaTranslateError);
    }

    Ok(tr.physical_address)
}

/// Sets processor-specific debug registers and configures the vcpu for handling
/// certain guest debug events using the `KVM_SET_GUEST_DEBUG` ioctl.
///
/// See: https://github.com/rust-vmm/kvm-ioctls/blob/ae039f245db0cd40ac20c32049ec536222b6dea7/src/ioctls/vcpu.rs#L1278
pub fn set_guest_debug(vcpu_fd: &RawFd, debug_struct: &kvm_guest_debug) -> Result<(), Error> {
    // SAFETY: Safe because we allocated the structure and we trust the kernel.
    let ret = unsafe { ioctl_with_ref(vcpu_fd, KVM_SET_GUEST_DEBUG(), debug_struct) };
    if ret < 0 {
        return Err(last_error());
    }
    Ok(())
}

/// Configures the kvm guest debug regs to register the hardware breakpoints, the `arch.debugreg`
/// attribute is used to store the location of the hardware breakpoints, with the 8th slot being
/// used as a bitfield to track which registers are enabled and setting the
/// `X86_GLOBAL_DEBUG_ENABLE` flags. Further reading on the DR7 register can be found here:
/// https://en.wikipedia.org/wiki/X86_debug_register#DR7_-_Debug_control
fn set_kvm_debug(control: u32, vcpu_fd: &RawFd, addrs: &[GuestAddress]) -> Result<(), Error> {
    let mut dbg = kvm_guest_debug {
        control,
        ..Default::default()
    };

    dbg.arch.debugreg[7] = X86_GLOBAL_DEBUG_ENABLE;

    for (i, addr) in addrs.iter().enumerate() {
        dbg.arch.debugreg[i] = addr.0;
        // Set global breakpoint enable flag for the specific breakpoint number by setting the bit
        dbg.arch.debugreg[7] |= 2 << (i * 2);
    }

    set_guest_debug(vcpu_fd, &dbg)
}

/// Configures the Vcpu for debugging and sets the hardware breakpoints on the Vcpu
pub fn vcpu_set_debug(vcpu_fd: &RawFd, addrs: &[GuestAddress], step: bool) -> Result<(), Error> {
    let mut control = KVM_GUESTDBG_ENABLE | KVM_GUESTDBG_USE_HW_BP | KVM_GUESTDBG_USE_SW_BP;
    if step {
        control |= KVM_GUESTDBG_SINGLESTEP;
    }

    set_kvm_debug(control, vcpu_fd, addrs)
}

/// Injects a BP back into the guest kernel for it to handle, this is particularly useful for the
/// kernels selftesting which can happen during boot.
pub fn vcpu_inject_bp(vcpu_fd: &RawFd, addrs: &[GuestAddress], step: bool) -> Result<(), Error> {
    let mut control = KVM_GUESTDBG_ENABLE
        | KVM_GUESTDBG_USE_HW_BP
        | KVM_GUESTDBG_USE_SW_BP
        | KVM_GUESTDBG_INJECT_BP;

    if step {
        control |= KVM_GUESTDBG_SINGLESTEP;
    }

    set_kvm_debug(control, vcpu_fd, addrs)
}
