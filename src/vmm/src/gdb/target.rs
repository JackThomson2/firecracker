// Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::os::fd::RawFd;
use std::sync::mpsc::{Receiver, RecvError};
use std::sync::{Arc, Mutex, PoisonError};

use arrayvec::ArrayVec;
use gdbstub::arch::Arch;
use gdbstub::common::{Signal, Tid};
use gdbstub::stub::{BaseStopReason, MultiThreadStopReason};
use gdbstub::target::ext::base::multithread::{
    MultiThreadBase, MultiThreadResume, MultiThreadResumeOps, MultiThreadSingleStep,
    MultiThreadSingleStepOps,
};
use gdbstub::target::ext::base::BaseOps;
use gdbstub::target::ext::breakpoints::{
    Breakpoints, BreakpointsOps, HwBreakpoint, HwBreakpointOps, SwBreakpoint, SwBreakpointOps,
};
use gdbstub::target::ext::thread_extra_info::{ThreadExtraInfo, ThreadExtraInfoOps};
use gdbstub::target::{Target, TargetError, TargetResult};
#[cfg(target_arch = "x86_64")]
use gdbstub_arch::x86::reg::X86_64CoreRegs as CoreRegs;
#[cfg(target_arch = "x86_64")]
use gdbstub_arch::x86::X86_64_SSE as GdbArch;
#[cfg(target_arch = "aarch64")]
use gdbstub_arch::aarch64::reg::AArch64CoreRegs as CoreRegs;
#[cfg(target_arch = "aarch64")]
use gdbstub_arch::aarch64::AArch64 as GdbArch;
use kvm_bindings::kvm_regs;
#[cfg(target_arch = "aarch64")]
use kvm_bindings::user_pt_regs;
use vm_memory::{Bytes, GuestAddress};

use super::kvm_debug;
use crate::logger::{error, info};
use crate::utils::u64_to_usize;
use crate::vstate::vcpu::VcpuSendEventError;
use crate::{arch, FcExitCode, VcpuEvent, VcpuResponse, Vmm};

#[cfg(target_arch = "x86_64")]
const X86_SW_BP_OP: u8 = 0xCC;

#[cfg(target_arch = "x86_64")]
const SW_BP_SIZE: usize = 1;
#[cfg(target_arch = "aarch64")]
const SW_BP_SIZE: usize = 4;

#[derive(Debug, Clone)]
/// Stores the current state of a Vcpu with a copy of the Vcpu file descriptor
struct VcpuState {
    single_step: bool,
    paused: bool,
    vcpu_fd: RawFd,
}

impl VcpuState {
    /// Constructs a new instance of a VcpuState from a raw fd
    fn from_raw_fd(raw_fd: &RawFd) -> Self {
        Self {
            single_step: false,
            paused: false,
            vcpu_fd: *raw_fd,
        }
    }

    /// Disables single stepping on the Vcpu state
    fn reset_vcpu_state(&mut self) {
        self.single_step = false;
    }

    /// Updates the kvm debug flags set against the Vcpu with a check
    fn update_kvm_debug(&self, hw_breakpoints: &[GuestAddress]) -> Result<(), Error> {
        if !self.paused {
            info!("Attempted to update kvm debug on a non paused Vcpu");
            return Ok(());
        }

        if let Ok(mut regs) = kvm_debug::get_regs(&self.vcpu_fd) {
            if self.single_step {
                info!("Temporarily disabled interrupts on vcpu");
                regs.regs.pstate |= 0x00000080 | 0x00000040;
            } else {
                info!("Re-enabled interrupts on vcpu");
                regs.regs.pstate &= !(0x00000080 | 0x00000040);
            }
            kvm_debug::set_regs(&self.vcpu_fd, &regs)?;
        }

        kvm_debug::vcpu_set_debug(&self.vcpu_fd, hw_breakpoints, self.single_step)
    }
}

/// Errors from interactions between GDB and the VMM
#[derive(Debug)]
pub enum Error {
    /// An error during a GDB request
    GdbRequest,
    /// An error with the queue between the target and the Vcpus
    GdbQueueError,
    /// The response from the Vcpu was not allowed
    VcuRequestError,
    /// No currently paused Vcpu error
    NoPausedVcpu,
    /// Error when setting Vcpu debug flags
    VcpuKvmError,
    /// Server socket Error
    ServerSocketError,
    /// Error with creating GDB thread
    GdbThreadError,
    /// VMM locking error
    VmmLockError,
    /// Vcpu send event error
    VcpuSendEventError,
    /// Recieve error from Vcpu channel
    VcpuRecvError,
    /// TID Conversion error
    TidConversionError,
    /// KVM set guest debug error
    KvmIoctlsError(kvm_ioctls::Error),
    /// Gva no translation available
    KvmGvaTranslateError,
}

impl From<Error> for TargetError<Error> {
    fn from(_value: Error) -> Self {
        TargetError::NonFatal
    }
}

impl<E> From<PoisonError<E>> for Error {
    fn from(_value: PoisonError<E>) -> Self {
        Error::VmmLockError
    }
}

impl From<VcpuSendEventError> for Error {
    fn from(_value: VcpuSendEventError) -> Self {
        Error::VcpuSendEventError
    }
}

impl From<RecvError> for Error {
    fn from(_value: RecvError) -> Self {
        Error::VcpuRecvError
    }
}

/// Debug Target for firecracker.
///
/// This is used the manage the debug implementation and handle requests sent via GDB
#[derive(Debug)]
pub struct FirecrackerTarget {
    /// A mutex around the VMM to allow communicataion to the Vcpus
    vmm: Arc<Mutex<Vmm>>,
    /// Store the guest entry point
    entry_addr: GuestAddress,

    /// Listener for events sent from the Vcpu
    pub gdb_event: Receiver<usize>,

    /// Used to track the currently configured hardware breakpoints.
    /// Limited to 4 in x86 see:
    /// https://elixir.bootlin.com/linux/v6.1/source/arch/x86/include/asm/kvm_host.h#L210
    hw_breakpoints: ArrayVec<GuestAddress, 4>,
    /// Used to track the currently configured software breakpoints and store the op-code
    /// which was swapped out
    sw_breakpoints: HashMap<<GdbArch as Arch>::Usize, [u8; SW_BP_SIZE]>,

    /// Stores the current state of each Vcpu
    vcpu_state: Vec<VcpuState>,

    /// Stores the current paused thread id, GDB can inact commands without providing us a Tid to
    /// run on and expects us to use the last paused thread.
    paused_vcpu: Option<Tid>,
}

/// Convert the 1 indexed Tid to the 0 indexed Vcpuid
fn tid_to_vcpuid(tid: Tid) -> usize {
    tid.get() - 1
}

/// Converts the inernal index of a Vcpu to
/// the Tid required by GDB
pub fn vcpuid_to_tid(cpu_id: usize) -> Result<Tid, Error> {
    Tid::new(get_raw_tid(cpu_id)).ok_or(Error::TidConversionError)
}

/// Converts the inernal index of a Vcpu to
/// the 1 indexed value for GDB
pub fn get_raw_tid(cpu_id: usize) -> usize {
    cpu_id + 1
}

/// Helper function to get the instruction pointer abstracted across different Cpus
#[cfg(target_arch = "x86_64")]
fn get_instruction_pointer(regs: &kvm_regs) -> u64 {
    regs.rip
}

/// Helper function to get the instruction pointer abstracted across different Cpus
#[cfg(target_arch = "aarch64")]
fn get_instruction_pointer(regs: &kvm_regs) -> u64 {
    regs.regs.pc
}

impl FirecrackerTarget {
    /// Creates a new Target for GDB stub. This is used as the layer between GDB and the VMM it
    /// will handle requests from GDB and perform the appropriate actions, while also updating GDB
    /// with the state of the VMM / Vcpu's as we hit debug events
    pub fn new(
        vmm: Arc<Mutex<Vmm>>,
        vcpu_fds: Vec<RawFd>,
        gdb_event: Receiver<usize>,
        entry_addr: GuestAddress,
    ) -> Self {
        let vcpu_state = vcpu_fds.iter().map(VcpuState::from_raw_fd).collect();

        Self {
            vmm,
            entry_addr,
            gdb_event,
            // We only support 4 hw breakpoints on x86 this will need to be configurable on arm
            hw_breakpoints: Default::default(),
            sw_breakpoints: HashMap::new(),
            vcpu_state,

            paused_vcpu: Tid::new(1),
        }
    }

    /// Retrieves the currently paused Vcpu id returns an error if there is no currently paused Vcpu
    fn get_paused_vcpu_id(&self) -> Result<Tid, Error> {
        self.paused_vcpu.ok_or(Error::NoPausedVcpu)
    }

    /// Retrieves the currently paused Vcpu state returns an error if there is no currently paused
    /// Vcpu
    fn get_paused_vcpu(&self) -> Result<&VcpuState, Error> {
        let vcpu_index = tid_to_vcpuid(self.get_paused_vcpu_id()?);
        Ok(&self.vcpu_state[vcpu_index])
    }

    /// Updates state to reference the currently paused Vcpu and store that the cpu is currently
    /// paused
    pub fn set_paused_vcpu(&mut self, tid: Tid) {
        self.vcpu_state[tid_to_vcpuid(tid)].paused = true;
        self.paused_vcpu = Some(tid);
    }

    /// Resumes execution of the vmm, this will update all paused Vcpus with current kvm debug info
    /// and resume them
    fn resume_execution(&mut self) -> Result<(), Error> {
        self.vcpu_state
            .iter()
            .try_for_each(|state| state.update_kvm_debug(&self.hw_breakpoints))?;

        for cpu_id in 0..self.vcpu_state.len() {
            let tid = vcpuid_to_tid(cpu_id)?;
            self.request_resume(tid)?;
        }

        self.paused_vcpu = None;

        Ok(())
    }

    /// Resets all Vcpus to their base state
    fn reset_all_states(&mut self) {
        for value in self.vcpu_state.iter_mut() {
            value.reset_vcpu_state();
        }
    }

    /// Shuts down the VMM
    pub fn shutdown(&self) {
        self.vmm
            .lock()
            .expect("error unlocking vmm")
            .stop(FcExitCode::Ok)
    }

    /// Pauses the requested Vcpu
    pub fn request_pause(&mut self, tid: Tid) -> Result<(), Error> {
        let vcpu_state = &mut self.vcpu_state[tid_to_vcpuid(tid)];

        if vcpu_state.paused {
            info!("Attempted to pause a vcpu already paused.");
            // Pausing an already paused vcpu is not considered an error case from GDB
            return Ok(());
        }

        let cpu_handle = &self.vmm.lock()?.vcpus_handles[tid_to_vcpuid(tid)];

        cpu_handle.send_event(VcpuEvent::Pause)?;
        let _ = cpu_handle.response_receiver().recv()?;

        vcpu_state.paused = true;
        Ok(())
    }

    /// A helper function to allow the event loop to inject this breakpoint back into the Vcpu
    #[cfg(target_arch = "x86_64")]
    pub fn inject_bp_to_guest(&mut self, tid: Tid) -> Result<(), Error> {
        let vcpu_state = &mut self.vcpu_state[tid_to_vcpuid(tid)];
        kvm_debug::vcpu_inject_bp(&vcpu_state.vcpu_fd, &self.hw_breakpoints, false)
    }

    /// Resumes the Vcpu, will return early if the Vcpu is already running
    pub fn request_resume(&mut self, tid: Tid) -> Result<(), Error> {
        let vcpu_state = &mut self.vcpu_state[tid_to_vcpuid(tid)];

        if !vcpu_state.paused {
            info!("Attempted to resume a vcpu already running.");
            // Resuming an already running Vcpu is not considered an error case from GDB
            return Ok(());
        }

        let cpu_handle = &self.vmm.lock()?.vcpus_handles[tid_to_vcpuid(tid)];
        cpu_handle.send_event(VcpuEvent::Resume)?;

        let response = cpu_handle.response_receiver().recv()?;
        if let VcpuResponse::NotAllowed(message) = response {
            error!("Response resume : {message}");
            return Err(Error::VcuRequestError);
        }

        vcpu_state.paused = false;
        Ok(())
    }

    /// Identifies why the specifc core was paused to be returned to GDB if None is returned this
    /// indicates to handle this internally and don't notify GDB
    pub fn get_stop_reason(&self, tid: Tid) -> Result<Option<BaseStopReason<Tid, u64>>, Error> {
        let vcpu_state = &self.vcpu_state[tid_to_vcpuid(tid)];
        if vcpu_state.single_step {
            return Ok(Some(MultiThreadStopReason::SignalWithThread {
                tid,
                signal: Signal::SIGTRAP,
            }));
        }

        let Ok(regs) = kvm_debug::get_regs(&vcpu_state.vcpu_fd) else {
            return Ok(Some(MultiThreadStopReason::SwBreak(tid)));
        };

        let pc = get_instruction_pointer(&regs);

        let vmm = &self.vmm.lock()?;
        let physical_addr = kvm_debug::translate_gva(&vcpu_state.vcpu_fd, pc, &vmm)?;
        if self.sw_breakpoints.contains_key(&physical_addr) {
            return Ok(Some(MultiThreadStopReason::SwBreak(tid)));
        }

        if self.hw_breakpoints.contains(&GuestAddress(pc)) {
            return Ok(Some(MultiThreadStopReason::HwBreak(tid)));
        }

        if pc == self.entry_addr.0 {
            return Ok(Some(MultiThreadStopReason::HwBreak(tid)));
        }

        // This is not a breakpoint we've set, likely one set by the guest
        Ok(None)
    }
}

impl Target for FirecrackerTarget {
    type Error = Error;
    type Arch = GdbArch;

    #[inline(always)]
    fn base_ops(&mut self) -> BaseOps<Self::Arch, Self::Error> {
        BaseOps::MultiThread(self)
    }

    #[inline(always)]
    fn support_breakpoints(&mut self) -> Option<BreakpointsOps<Self>> {
        Some(self)
    }

    /// We disable implicit sw breakpoints as we want to manage these internally so we can inject
    /// breakpoints back into the guest if we didn't create them
    #[inline(always)]
    fn guard_rail_implicit_sw_breakpoints(&self) -> bool {
        false
    }
}

impl MultiThreadBase for FirecrackerTarget {

    /// Reads the registers for the Vcpu
    #[cfg(target_arch = "aarch64")]
    fn read_registers(&mut self, regs: &mut CoreRegs, tid: Tid) -> TargetResult<(), Self> {
        let vcpu_state = &self.vcpu_state[tid_to_vcpuid(tid)];
        let cpu_regs = kvm_debug::get_regs(&vcpu_state.vcpu_fd).map_err(|e| {
            error!("Failed to read cpu registers: {e:?}");
            TargetError::NonFatal
        })?;

        regs.x = cpu_regs.regs.regs;
        regs.sp = cpu_regs.regs.sp;
        regs.pc = cpu_regs.regs.pc;

        Ok(())
    }

    /// Reads the registers for the Vcpu
    #[cfg(target_arch = "x86_64")]
    fn read_registers(&mut self, regs: &mut CoreRegs, tid: Tid) -> TargetResult<(), Self> {
        let vcpu_state = &self.vcpu_state[tid_to_vcpuid(tid)];
        let cpu_regs = kvm_debug::get_regs(&vcpu_state.vcpu_fd).map_err(|e| {
            error!("Failed to read cpu registers: {e:?}");
            TargetError::NonFatal
        })?;

        regs.regs[0] = cpu_regs.rax;
        regs.regs[1] = cpu_regs.rbx;
        regs.regs[2] = cpu_regs.rcx;
        regs.regs[3] = cpu_regs.rdx;
        regs.regs[4] = cpu_regs.rsi;
        regs.regs[5] = cpu_regs.rdi;
        regs.regs[6] = cpu_regs.rbp;
        regs.regs[7] = cpu_regs.rsp;

        regs.regs[8] = cpu_regs.r8;
        regs.regs[9] = cpu_regs.r9;
        regs.regs[10] = cpu_regs.r10;
        regs.regs[11] = cpu_regs.r11;
        regs.regs[12] = cpu_regs.r12;
        regs.regs[13] = cpu_regs.r13;
        regs.regs[14] = cpu_regs.r14;
        regs.regs[15] = cpu_regs.r15;

        regs.rip = cpu_regs.rip;
        regs.eflags = u32::try_from(cpu_regs.rflags).map_err(|e| {
            error!("Error {e:?} converting rflags to u32");
            TargetError::NonFatal
        })?;

        Ok(())
    }

    /// Writes to the registers for the Vcpu
    #[cfg(target_arch = "aarch64")]
    fn write_registers(&mut self, regs: &CoreRegs, tid: Tid) -> TargetResult<(), Self> {

        let new_regs = kvm_regs {
            regs: user_pt_regs {
                regs: regs.x,
                sp: regs.sp,
                pc: regs.pc,
                ..Default::default()
            },
            ..Default::default()
        };
        let vcpu_state = &self.vcpu_state[tid_to_vcpuid(tid)];
        kvm_debug::set_regs(&vcpu_state.vcpu_fd, &new_regs).map_err(|e| {
            error!("Error setting registers {e:?}");
            TargetError::NonFatal
        })
    }

    /// Writes to the registers for the Vcpu
    #[cfg(target_arch = "x86_64")]
    fn write_registers(&mut self, regs: &CoreRegs, tid: Tid) -> TargetResult<(), Self> {
        let new_regs = kvm_regs {
            rax: regs.regs[0],
            rbx: regs.regs[1],
            rcx: regs.regs[2],
            rdx: regs.regs[3],
            rsi: regs.regs[4],
            rdi: regs.regs[5],
            rbp: regs.regs[6],
            rsp: regs.regs[7],

            r8: regs.regs[8],
            r9: regs.regs[9],
            r10: regs.regs[10],
            r11: regs.regs[11],
            r12: regs.regs[12],
            r13: regs.regs[13],
            r14: regs.regs[14],
            r15: regs.regs[15],

            rip: regs.rip,
            rflags: regs.eflags as u64,
        };

        let vcpu_state = &self.vcpu_state[tid_to_vcpuid(tid)];
        kvm_debug::set_regs(&vcpu_state.vcpu_fd, &new_regs).map_err(|e| {
            error!("Error setting registers {e:?}");
            TargetError::NonFatal
        })
    }

    /// Writes data to a guest virtual address for the Vcpu
    fn read_addrs(
        &mut self,
        start_addr: <Self::Arch as Arch>::Usize,
        mut data: &mut [u8],
        tid: Tid,
    ) -> TargetResult<usize, Self> {
        let start_addr = u64_to_usize(start_addr);
        let mut gva = start_addr;
        let vcpu_state = &self.vcpu_state[tid_to_vcpuid(tid)];

        let vmm = &self.vmm.lock().map_err(|_| {
            error!("Error locking vmm in read addr");
            TargetError::NonFatal
        })?;

        while !data.is_empty() {
            let gpa = kvm_debug::translate_gva(&vcpu_state.vcpu_fd, gva as u64, vmm).map_err(|e| {
                error!("Error {e:?} translating gva on read address: {start_addr:X}");
                TargetError::NonFatal
            })?;

            let paddr = u64_to_usize(gpa);
            let psize = arch::PAGE_SIZE;
            // Compute the amount space left in the page after the paddr
            let read_len = std::cmp::min(data.len(), psize - (paddr & (psize - 1)));

            vmm.guest_memory()
                .read(&mut data[..read_len], GuestAddress(paddr as u64))
                .map_err(|e| {
                    error!("Error reading memory {e:?}");
                    TargetError::NonFatal
                })?;

            data = &mut data[read_len..];
            gva += read_len;
        }

        Ok(gva - start_addr)
    }

    /// Writes data at a guest virtual address for the Vcpu
    fn write_addrs(
        &mut self,
        start_addr: <Self::Arch as Arch>::Usize,
        mut data: &[u8],
        tid: Tid,
    ) -> TargetResult<(), Self> {
        let mut gva = u64_to_usize(start_addr);
        let vcpu_state = &self.vcpu_state[tid_to_vcpuid(tid)];

        let vmm = &self.vmm.lock().map_err(|_| {
            error!("Error locking vmm in write addr");
            TargetError::NonFatal
        })?;

        while !data.is_empty() {
            let gpa = match kvm_debug::translate_gva(&vcpu_state.vcpu_fd, gva as u64, &vmm) {
                Ok(paddr) => u64_to_usize(paddr),
                Err(e) => {
                    error!("Error {e:?} translating gva");
                    return Err(TargetError::NonFatal);
                }
            };

            let psize = arch::PAGE_SIZE;
            // Compute the amount space left in the page after the paddr
            let write_len = std::cmp::min(data.len(), psize - (gpa & (psize - 1)));

            vmm.guest_memory()
                .write(&data[..write_len], GuestAddress(gpa as u64))
                .map_err(|e| {
                    error!("Error {e:?} writing memory at {gpa:X}");
                    TargetError::NonFatal
                })?;

            data = &data[write_len..];
            gva += write_len;
        }

        Ok(())
    }

    #[inline(always)]
    /// Makes the callback provided with each Vcpu
    /// GDB expects us to return all threads currently running with this command, for firecracker
    /// this is all Vcpus
    fn list_active_threads(
        &mut self,
        thread_is_active: &mut dyn FnMut(Tid),
    ) -> Result<(), Self::Error> {
        for id in 0..self.vcpu_state.len() {
            thread_is_active(vcpuid_to_tid(id)?)
        }

        Ok(())
    }

    #[inline(always)]
    fn support_resume(&mut self) -> Option<MultiThreadResumeOps<Self>> {
        Some(self)
    }

    #[inline(always)]
    fn support_thread_extra_info(&mut self) -> Option<ThreadExtraInfoOps<'_, Self>> {
        Some(self)
    }
}

impl MultiThreadResume for FirecrackerTarget {
    /// Disables single step on the Vcpu
    fn set_resume_action_continue(
        &mut self,
        tid: Tid,
        _signal: Option<Signal>,
    ) -> Result<(), Self::Error> {
        let vcpu_state = &mut self.vcpu_state[tid_to_vcpuid(tid)];
        vcpu_state.single_step = false;

        Ok(())
    }

    /// Resumes the execution of all currently paused Vcpus
    fn resume(&mut self) -> Result<(), Self::Error> {
        self.resume_execution()
    }

    /// Clears the state of all Vcpus setting it back to base config
    fn clear_resume_actions(&mut self) -> Result<(), Self::Error> {
        self.reset_all_states();

        Ok(())
    }

    #[inline(always)]
    fn support_single_step(&mut self) -> Option<MultiThreadSingleStepOps<'_, Self>> {
        Some(self)
    }
}

impl MultiThreadSingleStep for FirecrackerTarget {
    /// Enabled single step on the Vcpu
    fn set_resume_action_step(
        &mut self,
        tid: Tid,
        _signal: Option<Signal>,
    ) -> Result<(), Self::Error> {
        self.vcpu_state[tid_to_vcpuid(tid)].single_step = true;

        Ok(())
    }
}

impl Breakpoints for FirecrackerTarget {
    #[inline(always)]
    fn support_hw_breakpoint(&mut self) -> Option<HwBreakpointOps<Self>> {
        Some(self)
    }

    #[inline(always)]
    fn support_sw_breakpoint(&mut self) -> Option<SwBreakpointOps<Self>> {
        Some(self)
    }
}

impl HwBreakpoint for FirecrackerTarget {
    /// Adds a hardware breakpoint The breakpoint addresses are
    /// stored in state so we can track the reason for an exit.
    fn add_hw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        _kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        let ga = GuestAddress(addr);
        if self.hw_breakpoints.contains(&ga) {
            return Ok(true);
        }

        if self.hw_breakpoints.try_push(ga).is_err() {
            return Ok(false);
        }

        let state = self.get_paused_vcpu()?;
        state.update_kvm_debug(&self.hw_breakpoints)?;

        Ok(true)
    }

    /// Removes a hardware breakpoint.
    fn remove_hw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        _kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        match self.hw_breakpoints.iter().position(|&b| b.0 == addr) {
            None => return Ok(false),
            Some(pos) => self.hw_breakpoints.remove(pos),
        };

        let state = self.get_paused_vcpu()?;
        state.update_kvm_debug(&self.hw_breakpoints)?;

        Ok(true)
    }
}

impl SwBreakpoint for FirecrackerTarget {
    /// Inserts a software breakpoint.
    /// We initially translate the guest address to a physical address and then check if this
    /// is already present, if so we return early. Otherwise we store the opcode at the specified
    /// physical location in our store and replace it with the `X86_SW_BP_OP`
    fn add_sw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        _kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        let physical_addr = {
            let vmm = &self.vmm.lock().map_err(|_| {
                error!("Error locking vmm in add sw breakpoint");
                TargetError::NonFatal
            })?;

            kvm_debug::translate_gva(&self.get_paused_vcpu()?.vcpu_fd, addr, &vmm)?
        };

        if self.sw_breakpoints.contains_key(&physical_addr) {
            return Ok(true);
        }

        let paused_vcpu_id = self.get_paused_vcpu_id()?;

        let mut saved_register = [0; SW_BP_SIZE];
        self.read_addrs(addr, &mut saved_register, paused_vcpu_id)?;
        self.sw_breakpoints.insert(physical_addr, saved_register);

        #[cfg(target_arch = "x86_64")]
        let break_point = [X86_SW_BP_OP];
        #[cfg(target_arch = "aarch64")]
        let break_point:[u8; 4] = 0x20D4u32.to_be_bytes();
        self.write_addrs(addr, &break_point, paused_vcpu_id)?;
        Ok(true)
    }

    /// Removes a software breakpoint.
    /// We firstly translate the guest address to a physical address, we then check if
    /// the specified location is in our store, if so we load the stored opcode and write this back
    fn remove_sw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        _kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        let physical_addr = {
            let vmm = &self.vmm.lock().map_err(|_| {
                error!("Error locking vmm in remove sw breakpoint");
                TargetError::NonFatal
            })?;

            kvm_debug::translate_gva(&self.get_paused_vcpu()?.vcpu_fd, addr, &vmm)?
        };

        if let Some(removed) = self.sw_breakpoints.remove(&physical_addr) {
            self.write_addrs(addr, &removed, self.get_paused_vcpu_id()?)?;
            return Ok(true);
        }

        Ok(false)
    }
}

impl ThreadExtraInfo for FirecrackerTarget {
    /// Allows us to configure the formatting of the thread information, we just return the ID of
    /// the Vcpu
    fn thread_extra_info(&self, tid: Tid, buf: &mut [u8]) -> Result<usize, Self::Error> {
        let info = format!("Vcpu ID: {}", tid_to_vcpuid(tid));
        let size = buf.len().min(info.len());

        buf[..size].copy_from_slice(&info.as_bytes()[..size]);
        Ok(size)
    }
}
