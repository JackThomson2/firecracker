use std::collections::{HashMap, HashSet};
use std::io::Write;
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
use libc::exit;
use utils::eventfd::EventFd;
use vm_memory::{Bytes, GuestAddress};

use crate::logger::{error, info};
use crate::{FcExitCode, VcpuEvent, VcpuHandle, VcpuResponse, Vmm};

#[derive(Default, Debug)]
struct VCpuState {
    single_step: bool,
    paused: bool
}

/// TODO DOCS
#[derive(Debug)]
pub struct FirecrackerTarget {
    vmm: Arc<Mutex<Vmm>>,

    /// TODO DOCS
    pub gdb_event: EventFd,

    hw_breakpoints: Vec<GuestAddress>,
    sw_breakpoints: HashMap<<GdbArch as Arch>::Usize, [u8; 1]>,

    vcpu_state: HashMap<Tid, VCpuState>,

    paused_vcpu: Option<Tid>
}

fn tid_to_cpuid(tid: Tid) -> usize {
    tid.get() - 1
}

/// TODO DOCS
pub fn cpuid_to_tid(cpu_id: usize) -> Tid {
    Tid::new(get_raw_tid(cpu_id)).unwrap()
}

/// TODO DOCS
pub fn get_raw_tid(cpu_id: usize) -> usize {
    cpu_id + 1
}

impl FirecrackerTarget {
    /// TODO DOCS
    pub fn new(vmm: Arc<Mutex<Vmm>>, gdb_event: EventFd) -> Self {
        let mut vcpu_state = HashMap::new();
        let cpu_count = {
            vmm.lock().expect("Exception unlocking vmm").vcpus_handles.len()
        };

        for cpu_id in 0..cpu_count {
            let new_state = VCpuState {
                paused: false,
                single_step: false
            };

            vcpu_state.insert(cpuid_to_tid(cpu_id), new_state);
        }

        Self {
            vmm,
            gdb_event,
            hw_breakpoints: Vec::new(),
            sw_breakpoints: HashMap::new(),
            vcpu_state,

            paused_vcpu: None,
        }
    }

    pub fn get_paused_vcpu(&self) -> Tid {
        self.paused_vcpu.expect("Attempt to retrieve vcpu while non are paused..")
    }

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

    fn resume_execution(&mut self) {
        let to_resume: Vec<Tid> = self.vcpu_state
            .iter()
            .filter_map(|(tid, state)| {
                match state.paused {
                    true => Some(*tid),
                    false => None
                }
            })
            .collect();

        for tid in to_resume {
            self.update_kvm_debug(tid);
            self.request_resume(tid);
        }

        self.vcpu_state
            .iter_mut()
            .for_each(|(_, state)| {
                state.paused = false;
            });

        self.paused_vcpu = None;
    }

    fn reset_vcpu_state(cpu_state: &mut VCpuState) {
        cpu_state.single_step = false;
    }

    fn reset_all_states(&mut self) {
        for (_, value) in self.vcpu_state.iter_mut() {
            Self::reset_vcpu_state(value);
        }
    }

    /// TODO DOCS
    pub fn shutdown(&self) {
        info!("Shutting down the vmm");
        self.vmm.lock().expect("error unlocking vmm").stop(FcExitCode::Ok)
    }

    /// TODO DOCS
    pub fn request_pause(&mut self, tid: Tid) {
        let vcpu_state = match self.vcpu_state.get(&tid) {
            Some(res) => res,
            None => {
                info!("Attempted to resume a vcpu we have no state for.");
                return
            }
        };

        if vcpu_state.paused {
            info!("Attempted to pause a vcpu already paused.");
        }

        let cpu_handle =
            &self.vmm.lock().expect("error unlocking vmm").vcpus_handles[tid_to_cpuid(tid)];

        cpu_handle.send_event(VcpuEvent::Pause).unwrap();
        let _ = cpu_handle.response_receiver().recv().unwrap();
    }

    fn get_regs(&self, tid: Tid) -> Option<kvm_regs> {
        let cpu_handle =
            &self.vmm.lock().expect("error unlocking vmm").vcpus_handles[tid_to_cpuid(tid)];

        cpu_handle.send_event(VcpuEvent::GetRegisters).unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();

        if let VcpuResponse::KvmRegisters(response) = response {
            return Some(response);
        }

        if let VcpuResponse::NotAllowed(message) = response {
            info!("Response from get regs: {message}");
            unsafe { exit(0) }
        }

        None
    }

    fn set_regs(&self, regs: kvm_regs, tid: Tid) {
        let cpu_handle =
            &self.vmm.lock().expect("error unlocking vmm").vcpus_handles[tid_to_cpuid(tid)];

        cpu_handle
            .send_event(VcpuEvent::SetRegisters(regs))
            .unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();
        if let VcpuResponse::NotAllowed(message) = response {
            info!("Response from set regs: {message}");
            unsafe { exit(0) }
        }
    }

    /// TODO DOCS
    pub fn request_resume(&mut self, tid: Tid) {
        let vcpu_state = match self.vcpu_state.get(&tid) {
            Some(res) => res,
            None => {
                info!("Attempted to resume a vcpu we have no state for.");
                return
            }
        };

        if !vcpu_state.paused {
            info!("Attempted to resume a vcpu already running.");
        }

        info!("Sending cpu resume request to tid {tid}");
        let cpu_handle =
            &self.vmm.lock().expect("error unlocking vmm").vcpus_handles[tid_to_cpuid(tid)];

        cpu_handle.send_event(VcpuEvent::Resume).unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();
        if let VcpuResponse::NotAllowed(message) = response {
            info!("Response resume : {message}");
            unsafe { exit(0) }
        }
    }

    fn update_kvm_debug(&self, tid: Tid) {
        info!("Sending kvm debug flags to tid {tid}");
        let cpu_handle =
            &self.vmm.lock().expect("error unlocking vmm").vcpus_handles[tid_to_cpuid(tid)];

        let vcpu_state = match self.vcpu_state.get(&tid) {
            Some(res) => res,
            None => {
                info!("Attempted to write kvm debug to a vcpu we have no state for.");
                return
            }
        };

        cpu_handle
            .send_event(VcpuEvent::SetKvmDebug(
                self.hw_breakpoints.clone(),
                vcpu_state.single_step,
            ))
            .unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();
        if let VcpuResponse::NotAllowed(message) = response {
            info!("Response from set kvm debug: {message}");
            unsafe { exit(0) }
        }
    }

    fn translate_gva(&self, cpu_handle: &VcpuHandle, address: u64) -> Option<u64> {
        cpu_handle
            .send_event(VcpuEvent::GvaTranslate(address))
            .unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();

        if let VcpuResponse::GvaTranslation(response) = response {
            return Some(response);
        }

        if let VcpuResponse::NotAllowed(message) = response {
            info!("Response from gva: {message}");
            unsafe { exit(0) }
        }

        Some(address)
    }

    fn inject_bp_to_guest(&self, tid: Tid) {
        let cpu_handle =
            &self.vmm.lock().expect("error unlocking vmm").vcpus_handles[tid_to_cpuid(tid)];

        cpu_handle
            .send_event(VcpuEvent::InjectKvmBP(
                self.hw_breakpoints.clone(),
                false,
            ))
            .unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();
        if let VcpuResponse::NotAllowed(message) = response {
            info!("Response resume : {message}");
            unsafe { exit(0) }
        }
    }

    fn is_tid_out_of_range(&self, tid: Tid) -> bool {
        self.vmm.lock().expect("Exception unlocking vmm").vcpus_handles.len() <= tid_to_cpuid(tid)
    }

    /// TODO DOCS
    pub fn get_stop_reason(&mut self, mut tid: Tid) -> Option<BaseStopReason<Tid, u64>> {
        if self.is_tid_out_of_range(tid) {
            info!("WARNING tid out of range defaulting to base tid");
            tid = self.get_paused_vcpu();
        }

        let cpu_regs = self.get_regs(tid);

        let vcpu_state = match self.vcpu_state.get(&tid) {
            Some(res) => res,
            None => {
                info!("Attempted to get stop reason for a vcpu we have no state for.");
                return None
            }
        };

        if vcpu_state.single_step {
            info!("TID {tid} completed its single step");
            return Some(MultiThreadStopReason::DoneStep);
        }

        if let Some(regs) = cpu_regs {
            let physical_addr = {
                let vmm = &self.vmm.lock().expect("Error unlocking vmm");
                self.translate_gva(&vmm.vcpus_handles[tid_to_cpuid(tid)], regs.rip)
                    .unwrap()
            };
            info!("Stopped at reg: {:X}. Physical address: {physical_addr:X}", regs.rip);

            if self.sw_breakpoints.contains_key(&physical_addr) {
                info!("Hit sw breakpoint clearing it and returning that");
                return Some(MultiThreadStopReason::SwBreak(tid));
            }

            if self.hw_breakpoints.contains(&GuestAddress(regs.rip)) {
                info!("Hit hw breakpoint returning that");
                return Some(MultiThreadStopReason::HwBreak(tid));
            }

            if regs.rip == 0x1000000 {
                info!("This was the injected bp...");
                return Some(MultiThreadStopReason::HwBreak(tid));
            }

            info!("Found bp we didn't set lets notify the guest");
            self.inject_bp_to_guest(tid);
            return None;
        } else {
            info!("No reg info");
        }

        return Some(MultiThreadStopReason::SwBreak(tid));
    }
}

impl Target for FirecrackerTarget {
    type Error = ();
    type Arch = GdbArch; // as an example

    #[inline(always)]
    fn base_ops(&mut self) -> BaseOps<Self::Arch, Self::Error> {
        BaseOps::MultiThread(self)
    }

    // opt-in to support for setting/removing breakpoints
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
        if let Some(cpu_regs) = self.get_regs(tid) {
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
            regs.eflags = cpu_regs.rflags as u32;
        } else {
            info!("Failed to ready cpu registers");
        }

        Ok(())
    }

    fn write_registers(&mut self, regs: &CoreRegs, tid: Tid) -> TargetResult<(), Self> {
        let mut new_regs: kvm_regs = Default::default();

        new_regs.rax = regs.regs[0];
        new_regs.rbx = regs.regs[1];
        new_regs.rcx = regs.regs[2];
        new_regs.rdx = regs.regs[3];
        new_regs.rsi = regs.regs[4];
        new_regs.rdi = regs.regs[5];
        new_regs.rbp = regs.regs[6];
        new_regs.rsp = regs.regs[7];

        new_regs.r8 = regs.regs[8];
        new_regs.r9 = regs.regs[9];
        new_regs.r10 = regs.regs[10];
        new_regs.r11 = regs.regs[11];
        new_regs.r12 = regs.regs[12];
        new_regs.r13 = regs.regs[13];
        new_regs.r14 = regs.regs[14];
        new_regs.r15 = regs.regs[15];

        new_regs.rip = regs.rip;
        new_regs.rflags = regs.eflags as u64;

        self.set_regs(new_regs, tid);

        Ok(())
    }

    fn read_addrs(
        &mut self,
        start_addr: <Self::Arch as Arch>::Usize,
        data: &mut [u8],
        tid: Tid,
    ) -> TargetResult<usize, Self> {
        info!("Reading address");
        let vmm = &self.vmm.lock().expect("Error unlocking vmm");
        let memory = vmm.guest_memory();

        let len = data.len();

        let mut total_read = 0_u64;

        if len == 1 {
            info!("Reading 1 byte from: {start_addr:X}");
        }

        while total_read < len as u64 {
            let gaddr = start_addr + total_read;
            let paddr = match self.translate_gva(&vmm.vcpus_handles[tid_to_cpuid(tid)], gaddr) {
                Some(paddr) => paddr,
                None => {
                    info!("Error translating gva on read address: {start_addr:X}");
                    gaddr
                }
            };
            let psize = 4096;
            let read_len = std::cmp::min(len as u64 - total_read, psize - (paddr & (psize - 1)));
            if memory
                .read(
                    &mut data[total_read as usize..total_read as usize + read_len as usize],
                    GuestAddress(paddr),
                )
                .is_err()
            {
                error!("Error while reading memory");
                return Err(TargetError::NonFatal);
            }
            total_read += read_len;
        }

        if total_read == 1 {
            info!("Read data was {:X}", data[0]);
        }

        info!("Read {total_read} bytes");

        Ok(total_read as usize)
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

        info!("Editing address data, we are writing {len} bytes");
        let mut total_written = 0_u64;

        if len == 1 {
            info!("Writing {:X} at {start_addr:X}", data[0]);
        }

        while total_written < len as u64 {
            info!("Looping total written {total_written} len {len}");
            let gaddr = start_addr + total_written;
            let paddr = match self.translate_gva(&vmm.vcpus_handles[tid_to_cpuid(tid)], gaddr) {
                Some(paddr) if paddr == u64::MIN => gaddr,
                Some(paddr) => paddr,
                None => {
                    info!("Error translating gva");
                    return Err(TargetError::NonFatal);
                }
            };
            let psize = 4096;
            let write_len =
                std::cmp::min(len as u64 - total_written, psize - (paddr & (psize - 1)));
            if memory
                .write(
                    &data[total_written as usize..total_written as usize + write_len as usize],
                    GuestAddress(paddr),
                )
                .is_err()
            {
                info!("Error writing memory at {paddr:X}");
                return Err(TargetError::NonFatal);
            }
            total_written += write_len;
        }

        info!("Done writing address data");

        Ok(())
    }

    #[inline(always)]
    fn list_active_threads(
        &mut self,
        thread_is_active: &mut dyn FnMut(Tid),
    ) -> Result<(), Self::Error> {
        info!("COMMAND: list active threads");
        let vmm = &self.vmm.lock().expect("Error unlocking vmm");

        for id in 0..vmm.vcpus_handles.len() {
            info!("Active thread id: {id} tid is: {:?}", cpuid_to_tid(id));
            thread_is_active(cpuid_to_tid(id))
        }

        Ok(())
    }

    #[inline(always)]
    fn support_resume(&mut self) -> Option<MultiThreadResumeOps<Self>> {
        Some(self)
    }

    #[inline(always)]
    fn support_thread_extra_info(
            &mut self,
        ) -> Option<ThreadExtraInfoOps<'_, Self>> {
        Some(self)
    }
}

impl MultiThreadResume for FirecrackerTarget {
    fn set_resume_action_continue(
        &mut self,
        tid: Tid,
        _signal: Option<Signal>,
    ) -> Result<(), Self::Error> {
        info!("COMMAND: Got action continue for tid: {tid:?}");
        let vcpu_state = match self.vcpu_state.get_mut(&tid) {
            Some(res) => res,
            None => {
                info!("Attempted to set action on a vcpu we have no state for.");
                return Ok(())
            }
        };
        vcpu_state.single_step = false;

        Ok(())
    }

    fn resume(&mut self) -> Result<(), Self::Error> {
        info!("COMMAND: Got resume command");
        self.resume_execution();

        Ok(())
    }

    fn clear_resume_actions(&mut self) -> Result<(), Self::Error> {
        info!("COMMAND: Got clear resume command");
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
        info!("COMMAND: Got single step command tid: {tid:?}");
        let vcpu_state = match self.vcpu_state.get_mut(&tid) {
            Some(res) => res,
            None => {
                info!("Attempted to set action on a vcpu we have no state for.");
                return Ok(())
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
        info!("Setting hw breakpoint at {addr:X}");
        self.hw_breakpoints.push(GuestAddress(addr));
        self.update_kvm_debug(self.get_paused_vcpu());

        Ok(true)
    }

    fn remove_hw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        _kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        info!("Removing hw breakpoint at {addr:X}");
        match self.hw_breakpoints.iter().position(|&b| b.0 == addr) {
            None => return Ok(false),
            Some(pos) => self.hw_breakpoints.remove(pos),
        };
        info!("Removed hw breakpoint..");
        self.update_kvm_debug(self.get_paused_vcpu());

        Ok(true)
    }
}

impl SwBreakpoint for FirecrackerTarget {
    fn add_sw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        _kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        info!("Setting sw breakpoint at {addr:X}");
        let physical_addr = {
            let vmm = &self.vmm.lock().expect("Error unlocking vmm");
            self.translate_gva(&vmm.vcpus_handles[tid_to_cpuid(self.get_paused_vcpu())], addr).unwrap()
        };

        if self.sw_breakpoints.contains_key(&physical_addr) {
            return Ok(true);
        }

        let mut saved_register = [0];
        let _ = self.read_addrs(addr, &mut saved_register, self.get_paused_vcpu());
        self.sw_breakpoints.insert(physical_addr, saved_register);
        info!("Inserting SW breakpoint at {physical_addr:X}");

        let break_point = [0xCC];
        let _ = self.write_addrs(addr, &break_point, self.get_paused_vcpu());
        Ok(true)
    }

    fn remove_sw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        _kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        info!("Removing sw breakpoint at {addr:X}");

        let physical_addr = {
            let vmm = &self.vmm.lock().expect("Error unlocking vmm");
            self.translate_gva(&vmm.vcpus_handles[tid_to_cpuid(self.get_paused_vcpu())], addr).unwrap()
        };
        info!("Removing SW breakpoint at {physical_addr:X}");

        if let Some(removed) = self.sw_breakpoints.remove(&physical_addr) {
            let _ = self.write_addrs(addr, &removed, self.get_paused_vcpu());
            return Ok(true);
        } else {
            info!("Looks like the breakpoint was already remove..");
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
