// Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use gdbstub::arch::Arch;
use gdbstub::common::{Signal, Tid};
use gdbstub::stub::{BaseStopReason, MultiThreadStopReason};
use gdbstub::target::ext::base::multithread::{
    MultiThreadBase, MultiThreadResume, MultiThreadResumeOps, MultiThreadSingleStep,
    MultiThreadSingleStepOps,
};
use gdbstub::target::ext::base::BaseOps;
use gdbstub::target::ext::breakpoints::{Breakpoints, HwBreakpoint, HwBreakpointOps, SwBreakpoint};
use gdbstub::target::ext::breakpoints::{BreakpointsOps, SwBreakpointOps};
use gdbstub::target::ext::thread_extra_info::{ThreadExtraInfo, ThreadExtraInfoOps};
use gdbstub::target::{Target, TargetError, TargetResult};
#[cfg(target_arch = "aarch64")]
use gdbstub_arch::aarch64::reg::AArch64CoreRegs as CoreRegs;
#[cfg(target_arch = "aarch64")]
use gdbstub_arch::aarch64::AArch64 as GdbArch;
#[cfg(target_arch = "x86_64")]
use gdbstub_arch::x86::reg::X86_64CoreRegs as CoreRegs;
#[cfg(target_arch = "x86_64")]
use gdbstub_arch::x86::X86_64_SSE as GdbArch;
use kvm_bindings::kvm_regs;
use libc::wait;
use vm_memory::{Bytes, GuestAddress};

use crate::logger::{error, info};
use crate::{FcExitCode, VcpuEvent, VcpuHandle, VcpuResponse, Vmm};

const X86_SW_BP_OP: u8 = 0xCC;

#[derive(Default, Debug)]
struct VCpuState {
    single_step: bool,
    paused: bool,
}

/// These are errors which can occur with interactions between GDB and the VMM
#[derive(Debug)]
pub enum Error {
    /// An error during a GDB request
    GdbRequest,
    /// An error with the queue between the target and the VCPUs
    GdbQueueError,
    /// The response from the VCPU was not allowed
    VCPURequestError,
    /// No currently paused Vcpu error
    NoPausedVCpu,
}

impl<E> From<Error> for TargetError<E> {
    fn from(_value: Error) -> Self {
        TargetError::NonFatal
    }
}

/// The debug target for firecracker. This is used the manage the debug
/// implementation and handle requests sent via GDB
#[derive(Debug)]
pub struct FirecrackerTarget {
    vmm: Arc<Mutex<Vmm>>,
    entry_addr: GuestAddress,

    /// This event fd us used to notify us of cpu debug exits
    pub gdb_event: Receiver<usize>,

    hw_breakpoints: Vec<GuestAddress>,
    sw_breakpoints: HashMap<<GdbArch as Arch>::Usize, [u8; 1]>,

    vcpu_state: HashMap<Tid, VCpuState>,

    paused_vcpu: Option<Tid>,
}

fn tid_to_cpuid(tid: Tid) -> usize {
    tid.get() - 1
}

/// Converts the inernal index of a vcpu to
/// the Tid required by GDB
pub fn cpuid_to_tid(cpu_id: usize) -> Tid {
    Tid::new(get_raw_tid(cpu_id)).expect("Error translating TID")
}

/// Converts the inernal index of a vcpu to
/// the 1 indexed value for GDB
pub fn get_raw_tid(cpu_id: usize) -> usize {
    cpu_id + 1
}

impl FirecrackerTarget {
    /// Creates a new Target for GDB stub. This is used as the layer between GDB and the VMM it
    /// will handle requests from GDB and perform the appropriate actions, while also updating GDB
    /// with the state of the VMM / VCPU's as we hit debug events
    pub fn new(vmm: Arc<Mutex<Vmm>>, gdb_event: Receiver<usize>, entry_addr: GuestAddress) -> Self {
        let mut vcpu_state = HashMap::new();
        let cpu_count = {
            vmm.lock()
                .expect("Exception unlocking vmm")
                .vcpus_handles
                .len()
        };

        for cpu_id in 0..cpu_count {
            let new_state = VCpuState {
                paused: false,
                single_step: false,
            };

            vcpu_state.insert(cpuid_to_tid(cpu_id), new_state);
        }

        Self {
            vmm,
            entry_addr,
            gdb_event,
            // We only support 4 hw breakpoints on x86 this will need to be configurable on arm
            hw_breakpoints: Vec::with_capacity(4),
            sw_breakpoints: HashMap::new(),
            vcpu_state,

            paused_vcpu: Tid::new(1),
        }
    }

    fn get_paused_vcpu(&self) -> Result<Tid, Error> {
        match self.paused_vcpu {
            Some(res) => Ok(res),
            None => Err(Error::NoPausedVCpu)
        }
    }

    /// This is used to notify the target that the provided Tid
    /// is in a paused state
    pub fn notify_paused_vcpu(&mut self, tid: Tid) {
        let found = match self.vcpu_state.get_mut(&tid) {
            Some(res) => res,
            None => {
                info!("TID {tid} was not known.");
                return;
            }
        };

        found.paused = true;
        self.paused_vcpu = Some(tid);
    }

    fn resume_execution(&mut self) -> Result<(), Error> {
        let to_resume: Vec<Tid> = self
            .vcpu_state
            .iter()
            .filter_map(|(tid, state)| match state.paused {
                true => Some(*tid),
                false => None,
            })
            .collect();

        for tid in to_resume {
            self.update_kvm_debug(tid)?;
            self.request_resume(tid)?;
        }

        self.vcpu_state.iter_mut().for_each(|(_, state)| {
            state.paused = false;
        });

        self.paused_vcpu = None;

        Ok(())
    }

    fn reset_vcpu_state(cpu_state: &mut VCpuState) {
        cpu_state.single_step = false;
    }

    fn reset_all_states(&mut self) {
        for (_, value) in self.vcpu_state.iter_mut() {
            Self::reset_vcpu_state(value);
        }
    }

    /// This method is used to shutdown the VMM
    pub fn shutdown(&self) {
        self.vmm
            .lock()
            .expect("error unlocking vmm")
            .stop(FcExitCode::Ok)
    }

    /// This method can be used to manually pause the requested vcpu
    pub fn request_pause(&mut self, tid: Tid) -> Result<(), Error> {
        let vcpu_state = match self.vcpu_state.get_mut(&tid) {
            Some(res) => res,
            None => {
                info!("Attempted to pause a vcpu we have no state for.");
                return Err(Error::VCPURequestError);
            }
        };

        if vcpu_state.paused {
            info!("Attempted to pause a vcpu already paused.");
            // Pausing an already paused vcpu is not considered an error case from GDB
            return Ok(());
        }

        let cpu_handle =
            &self.vmm.lock().expect("error unlocking vmm").vcpus_handles[tid_to_cpuid(tid)];

        cpu_handle.send_event(VcpuEvent::Pause).expect("Error sending message to vpu");
        let _ = cpu_handle.response_receiver().recv().expect("Error recieving message from vcpu");

        vcpu_state.paused = false;
        Ok(())
    }

    fn get_regs(&self, tid: Tid) -> Result<kvm_regs, Error> {
        let cpu_handle =
            &self.vmm.lock().expect("error unlocking vmm").vcpus_handles[tid_to_cpuid(tid)];

        cpu_handle.send_event(VcpuEvent::GetRegisters).expect("Error sending message to vcpu");
        let response = cpu_handle.response_receiver().recv().expect("Error recieving message from vcpu");

        if let VcpuResponse::KvmRegisters(response) = response {
            return Ok(response);
        }

        if let VcpuResponse::NotAllowed(message) = response {
            error!("Response from get regs: {message} for TID {tid:?}");
        }

        Err(Error::VCPURequestError)
    }

    fn set_regs(&self, regs: kvm_regs, tid: Tid) -> Result<(), Error> {
        let cpu_handle =
            &self.vmm.lock().expect("error unlocking vmm").vcpus_handles[tid_to_cpuid(tid)];

        cpu_handle
            .send_event(VcpuEvent::SetRegisters(regs))
            .expect("Error sending message to vcpu");
        let response = cpu_handle.response_receiver().recv().expect("Error recieving message from vcpu");
        if let VcpuResponse::NotAllowed(message) = response {
            error!("Response from set regs: {message} on tid: {tid:?}");
            return Err(Error::VCPURequestError);
        }

        Ok(())
    }

    /// Used to request the specified core to resume execution. The function
    /// will return early if the vcpu is already paused and not currently running
    pub fn request_resume(&mut self, tid: Tid) -> Result<(), Error> {
        let vcpu_state = match self.vcpu_state.get_mut(&tid) {
            Some(res) => res,
            None => {
                error!("Attempted to resume a vcpu we have no state for.");
                return Err(Error::VCPURequestError);
            }
        };

        if !vcpu_state.paused {
            info!("Attempted to resume a vcpu already running.");
            // Resuming an already running vcpu is not considered an error case from GDB
            return Ok(());
        }

        let cpu_handle =
            &self.vmm.lock().expect("error unlocking vmm").vcpus_handles[tid_to_cpuid(tid)];

        cpu_handle.send_event(VcpuEvent::Resume).expect("Error sending message to vcpu");
        let response = cpu_handle.response_receiver().recv().expect("Error recieving message from vcpu");
        if let VcpuResponse::NotAllowed(message) = response {
            error!("Response resume : {message}");
            return Err(Error::VCPURequestError);
        }

        vcpu_state.paused = false;
        Ok(())
    }

    fn update_kvm_debug(&self, tid: Tid) -> Result<(), Error> {
        let cpu_handle =
            &self.vmm.lock().expect("error unlocking vmm").vcpus_handles[tid_to_cpuid(tid)];

        let vcpu_state = match self.vcpu_state.get(&tid) {
            Some(res) => res,
            None => {
                error!("Attempted to write kvm debug to a vcpu we have no state for.");
                return Err(Error::VCPURequestError);
            }
        };

        cpu_handle
            .send_event(VcpuEvent::SetKvmDebug(
                self.hw_breakpoints.clone(),
                vcpu_state.single_step,
            ))
            .expect("Error sending message to vcpu");

        match cpu_handle.response_receiver().recv() {
            Ok(VcpuResponse::NotAllowed(message)) => {
                error!("Response resume : {message}");
                Err(Error::VCPURequestError)
            },
            Err(_) => Err(Error::VCPURequestError),
            _ => Ok(())
        }
    }

    fn translate_gva(&self, cpu_handle: &VcpuHandle, address: u64) -> Result<u64, Error> {
        cpu_handle
            .send_event(VcpuEvent::GvaTranslate(address))
            .expect("Error sending message to vcpu");
        let response = cpu_handle.response_receiver().recv().expect("Error recieving message from vcpu");

        if let VcpuResponse::GvaTranslation(response) = response {
            return Ok(response);
        }

        if let VcpuResponse::NotAllowed(message) = response {
            error!("Response from gva: {message}");
            return Err(Error::VCPURequestError);
        }

        Ok(address)
    }

    fn inject_bp_to_guest(&self, tid: Tid) -> Result<(), Error> {
        let cpu_handle =
            &self.vmm.lock().expect("error unlocking vmm").vcpus_handles[tid_to_cpuid(tid)];

        cpu_handle
            .send_event(VcpuEvent::InjectKvmBP(self.hw_breakpoints.clone(), false))
            .expect("Error sending message to vcpu");

        match cpu_handle.response_receiver().recv() {
            Ok(VcpuResponse::NotAllowed(message)) => {
                error!("Response resume : {message}");
                Err(Error::VCPURequestError)
            },
            Err(_) => Err(Error::VCPURequestError),
            _ => Ok(())
        }
    }

    fn is_tid_out_of_range(&self, tid: Tid) -> bool {
        self.vmm
            .lock()
            .expect("Exception unlocking vmm")
            .vcpus_handles
            .len()
            <= tid_to_cpuid(tid)
    }

    /// Used to identify why the specifc core was paused to be returned to GDB
    /// if the function returns None this means we want to handle this internally and don't want
    /// to notify GDB as it's likely the guest self testing
    pub fn get_stop_reason(&mut self, mut tid: Tid) -> Option<BaseStopReason<Tid, u64>> {
        if self.is_tid_out_of_range(tid) {
            tid = self.get_paused_vcpu().expect("No paused vcpu");
        }

        let cpu_regs = self.get_regs(tid);

        let vcpu_state = match self.vcpu_state.get(&tid) {
            Some(res) => res,
            None => {
                return None;
            }
        };

        if vcpu_state.single_step {
            return Some(MultiThreadStopReason::SignalWithThread {
                tid,
                signal: Signal::SIGTRAP,
            });
        }

        if let Ok(regs) = cpu_regs {
            let physical_addr = {
                let vmm = &self.vmm.lock().expect("Error unlocking vmm");
                self.translate_gva(&vmm.vcpus_handles[tid_to_cpuid(tid)], regs.rip)
                    .expect("Error translating GVA")
            };

            if self.sw_breakpoints.contains_key(&physical_addr) {
                return Some(MultiThreadStopReason::SwBreak(tid));
            }

            if self.hw_breakpoints.contains(&GuestAddress(regs.rip)) {
                return Some(MultiThreadStopReason::HwBreak(tid));
            }

            if regs.rip == self.entry_addr.0 {
                return Some(MultiThreadStopReason::HwBreak(tid));
            }

            self.inject_bp_to_guest(tid)
                .expect("Error injecting bp back to guest");
            return None;
        }

        Some(MultiThreadStopReason::SwBreak(tid))
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

    #[inline(always)]
    fn guard_rail_implicit_sw_breakpoints(&self) -> bool {
        false
    }
}

impl MultiThreadBase for FirecrackerTarget {
    fn read_registers(&mut self, regs: &mut CoreRegs, tid: Tid) -> TargetResult<(), Self> {
        if let Ok(cpu_regs) = self.get_regs(tid) {
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
            regs.eflags = u32::try_from(cpu_regs.rflags).expect("Error converting rflags");

            Ok(())
        } else {
            error!("Failed to read cpu registers");
            Err(TargetError::NonFatal)
        }
    }

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

        if self.set_regs(new_regs, tid).is_err() {
            return Err(TargetError::NonFatal);
        }

        Ok(())
    }

    fn read_addrs(
        &mut self,
        start_addr: <Self::Arch as Arch>::Usize,
        data: &mut [u8],
        tid: Tid,
    ) -> TargetResult<usize, Self> {
        let vmm = &self.vmm.lock().expect("Error unlocking vmm");
        let memory = vmm.guest_memory();

        let len = data.len();

        let mut total_read = 0_usize;
        let start_addr = usize::try_from(start_addr).expect("Unable to convert addr to usize");

        while total_read < len {
            let gaddr = start_addr + total_read;
            let paddr =
                match self.translate_gva(&vmm.vcpus_handles[tid_to_cpuid(tid)], gaddr as u64) {
                    Ok(paddr) => usize::try_from(paddr).expect("Unable to convert addr to usize"),
                    Err(e) => {
                        error!("Error {e:?} translating gva on read address: {start_addr:X}");
                        return Err(TargetError::NonFatal);
                    }
                };

            let psize = 4096;
            let read_len = std::cmp::min(len - total_read, psize - (paddr & (psize - 1)));

            if memory
                .read(
                    &mut data[total_read..total_read + read_len],
                    GuestAddress(paddr as u64),
                )
                .is_err()
            {
                return Err(TargetError::NonFatal);
            }

            total_read += read_len;
        }

        Ok(total_read)
    }

    fn write_addrs(
        &mut self,
        start_addr: <Self::Arch as Arch>::Usize,
        data: &[u8],
        tid: Tid,
    ) -> TargetResult<(), Self> {
        let vmm = &self.vmm.lock().expect("Error unlocking vmm");
        let memory = vmm.guest_memory();
        let len = data.len();

        let mut total_written = 0_usize;
        let start_addr = usize::try_from(start_addr).expect("Unable to convert addr to usize");

        while total_written < len {
            let gaddr = start_addr + total_written;
            let paddr =
                match self.translate_gva(&vmm.vcpus_handles[tid_to_cpuid(tid)], gaddr as u64) {
                    Ok(paddr) if paddr == <Self::Arch as Arch>::Usize::MIN => gaddr,
                    Ok(paddr) => usize::try_from(paddr).expect("Error converting addr to usize"),
                    Err(e) => {
                        error!("Error {e:?} translating gva");
                        return Err(TargetError::NonFatal);
                    }
                };

            let psize = 4096;
            let write_len = std::cmp::min(len - total_written, psize - (paddr & (psize - 1)));
            if let Err(e) = memory
                .write(
                    &data[total_written..total_written + write_len],
                    GuestAddress(paddr as u64),
                )
            {
                error!("Error {e:?} writing memory at {paddr:X}");
                return Err(TargetError::NonFatal);
            }
            total_written += write_len;
        }

        Ok(())
    }

    #[inline(always)]
    fn list_active_threads(
        &mut self,
        thread_is_active: &mut dyn FnMut(Tid),
    ) -> Result<(), Self::Error> {
        let vmm = &self.vmm.lock().expect("Error unlocking vmm");

        for id in 0..vmm.vcpus_handles.len() {
            thread_is_active(cpuid_to_tid(id))
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
    fn set_resume_action_continue(
        &mut self,
        tid: Tid,
        _signal: Option<Signal>,
    ) -> Result<(), Self::Error> {
        let vcpu_state = match self.vcpu_state.get_mut(&tid) {
            Some(res) => res,
            None => {
                error!("Attempted to set action on a vcpu we have no state for.");
                return Err(Error::VCPURequestError);
            }
        };
        vcpu_state.single_step = false;

        Ok(())
    }

    fn resume(&mut self) -> Result<(), Self::Error> {
        self.resume_execution()
    }

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
    fn set_resume_action_step(
        &mut self,
        tid: Tid,
        _signal: Option<Signal>,
    ) -> Result<(), Self::Error> {
        let vcpu_state = match self.vcpu_state.get_mut(&tid) {
            Some(res) => res,
            None => {
                info!("Attempted to set action on a vcpu we have no state for.");
                return Ok(());
            }
        };
        vcpu_state.single_step = true;

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
    fn add_hw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        _kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        // If we are at capacity we can't accept any more hw breakpoints
        if self.hw_breakpoints.len() == self.hw_breakpoints.capacity() {
            return Ok(false)
        }

        self.hw_breakpoints.push(GuestAddress(addr));
        self.update_kvm_debug(self.get_paused_vcpu()?)?;

        Ok(true)
    }

    fn remove_hw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        _kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        match self.hw_breakpoints.iter().position(|&b| b.0 == addr) {
            None => return Ok(false),
            Some(pos) => self.hw_breakpoints.remove(pos),
        };

        self.update_kvm_debug(self.get_paused_vcpu()?)?;

        Ok(true)
    }
}

impl SwBreakpoint for FirecrackerTarget {
    fn add_sw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        _kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        let physical_addr = {
            let vmm = &self.vmm.lock().expect("Error unlocking vmm");
            self.translate_gva(
                &vmm.vcpus_handles[tid_to_cpuid(self.get_paused_vcpu()?)],
                addr,
            )?
        };

        if self.sw_breakpoints.contains_key(&physical_addr) {
            return Ok(true);
        }

        let mut saved_register = [0];
        self.read_addrs(addr, &mut saved_register, self.get_paused_vcpu()?)?;
        self.sw_breakpoints.insert(physical_addr, saved_register);

        let break_point = [X86_SW_BP_OP];
        self.write_addrs(addr, &break_point, self.get_paused_vcpu()?)?;
        Ok(true)
    }

    fn remove_sw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        _kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        let physical_addr = {
            let vmm = &self.vmm.lock().expect("Error unlocking vmm");
            self.translate_gva(
                &vmm.vcpus_handles[tid_to_cpuid(self.get_paused_vcpu()?)],
                addr,
            )?
        };

        if let Some(removed) = self.sw_breakpoints.remove(&physical_addr) {
            let _ = self.write_addrs(addr, &removed, self.get_paused_vcpu()?);
            return Ok(true);
        }

        Ok(true)
    }
}

impl ThreadExtraInfo for FirecrackerTarget {
    fn thread_extra_info(&self, tid: Tid, buf: &mut [u8]) -> Result<usize, Self::Error> {
        let info = format!("VCPU ID: {}", tid_to_cpuid(tid));
        let size = buf.len().min(info.len());

        buf[..size].copy_from_slice(&info.as_bytes()[..size]);
        Ok(size)
    }
}
