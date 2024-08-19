use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use gdbstub::arch::Arch;
use gdbstub::common::Signal;
use gdbstub::stub::{BaseStopReason, SingleThreadStopReason};
use gdbstub::target::ext::thread_extra_info;
use gdbstub::target::{Target, TargetError, TargetResult};
use gdbstub::target::ext::base::BaseOps;
use gdbstub::target::ext::base::singlethread::{
    SingleThreadResumeOps, SingleThreadSingleStepOps
};
use gdbstub::target::ext::base::singlethread::{
    SingleThreadBase, SingleThreadResume, SingleThreadSingleStep
};
use gdbstub::target::ext::breakpoints::{Breakpoints, HwBreakpoint, HwBreakpointOps, SwBreakpoint};
use gdbstub::target::ext::breakpoints::{BreakpointsOps, SwBreakpointOps};
#[cfg(target_arch = "aarch64")]
use gdbstub_arch::aarch64::reg::AArch64CoreRegs as CoreRegs;
#[cfg(target_arch = "aarch64")]
use gdbstub_arch::aarch64::AArch64 as GdbArch;
#[cfg(target_arch = "x86_64")]
use gdbstub_arch::x86::reg::X86_64CoreRegs as CoreRegs;
#[cfg(target_arch = "x86_64")]
use gdbstub_arch::x86::X86_64_SSE as GdbArch;
use kvm_bindings::kvm_regs;
use utils::eventfd::EventFd;
use vm_memory::{Bytes, GuestAddress};

use crate::logger::{info, error};
use crate::{VcpuEvent, VcpuHandle, VcpuResponse, Vmm};


pub struct FirecrackerTarget {
    vmm: Arc<Mutex<Vmm>>,
    pub gdb_event: EventFd,

    hw_breakpoints: Vec<GuestAddress>,
    sw_breakpoints: HashMap<<GdbArch as Arch>::Usize, [u8;1]>,

    pub single_step: bool
}

impl FirecrackerTarget {
    pub fn new(vmm: Arc<Mutex<Vmm>>, gdb_event: EventFd) -> Self {
        Self{
            vmm,
            gdb_event,
            hw_breakpoints: Vec::new(),
            sw_breakpoints: HashMap::new(),
            single_step: false,
        }
    }

    pub fn request_pause(&self) {
        let cpu_handle = &self.vmm
            .lock()
            .expect("error unlocking vmm")
            .vcpus_handles[0];
        
        cpu_handle.send_event(VcpuEvent::Pause).unwrap();
        let _ = cpu_handle.response_receiver().recv().unwrap();
    }

    fn get_regs(&self) -> Option<kvm_regs> {
        let cpu_handle = &self.vmm
            .lock()
            .expect("error unlocking vmm")
            .vcpus_handles[0];
        
        cpu_handle.send_event(VcpuEvent::GetRegisters).unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();

        if let VcpuResponse::KvmRegisters(response) = response {
            return Some(response);
        }

        if let VcpuResponse::NotAllowed(message) = response {
            info!("Response from get regs: {message}");
        }

        None
    }

    fn set_regs(&self, regs: kvm_regs) {
        let cpu_handle = &self.vmm
            .lock()
            .expect("error unlocking vmm")
            .vcpus_handles[0];
        
        cpu_handle.send_event(VcpuEvent::SetRegisters(regs)).unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();
        if let VcpuResponse::NotAllowed(message) = response {
            info!("Response from set regs: {message}");
        }
    }

    pub fn request_resume(&self) {
        let cpu_handle = &self.vmm
            .lock()
            .expect("error unlocking vmm")
            .vcpus_handles[0];

        cpu_handle.send_event(VcpuEvent::Resume).unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();
        if let VcpuResponse::NotAllowed(message) = response {
            info!("Response resume : {message}");
        }
    }

    fn resume_execution(&self) {
        let cpu_handle = &self.vmm
            .lock()
            .expect("error unlocking vmm")
            .vcpus_handles[0];

        // Ensure breakpoints are correctly in sync before resuming
        cpu_handle.send_event(VcpuEvent::SetKvmDebug(self.hw_breakpoints.clone(), self.single_step)).unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();
        if let VcpuResponse::NotAllowed(message) = response {
            info!("Response from set kvm debug: {message}");
        }

        cpu_handle.send_event(VcpuEvent::Resume).unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();
        if let VcpuResponse::NotAllowed(message) = response {
            info!("Response resume : {message}");
        }
    }

    fn translate_gva(&self, cpu_handle: &VcpuHandle, address: u64) -> Option<u64> {
        cpu_handle.send_event(VcpuEvent::GvaTranslate(address)).unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();

        if let VcpuResponse::GvaTranslation(response) = response {
            return Some(response);
        }

        if let VcpuResponse::NotAllowed(message) = response {
            info!("Response from gva: {message}");
            return None;
        }

        Some(address)
    }

    fn inject_bp_to_guest(&self) {
        let cpu_handle = &self.vmm
            .lock()
            .expect("error unlocking vmm")
            .vcpus_handles[0];

        cpu_handle.send_event(VcpuEvent::InjectKvmBP(self.hw_breakpoints.clone(), self.single_step)).unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();
        if let VcpuResponse::NotAllowed(message) = response {
            info!("Response resume : {message}");
        }
    }

    pub fn get_stop_reason(&mut self) -> Option<BaseStopReason<(), u64>> {
        let cpu_regs = self.get_regs();

        if self.single_step {
            return Some(SingleThreadStopReason::DoneStep);
        }

        if let Some(regs) = cpu_regs {
            info!("Stopped at reg: {:X}", regs.rip);
            let physical_addr = {
               let vmm = &self.vmm.lock().expect("Error unlocking vmm");
               self.translate_gva(&vmm.vcpus_handles[0], regs.rip).unwrap()
            };

            if let Some(removed) = self.sw_breakpoints.remove(&physical_addr) {
                info!("Hit sw breakpoint clearing it and returning that");
                let _ = self.write_addrs(regs.rip, &removed);
                return Some(SingleThreadStopReason::SwBreak(()));
            }

            if self.hw_breakpoints.contains(&GuestAddress(regs.rip)) {
                info!("Hit hw breakpoint returning that");
                return Some(SingleThreadStopReason::HwBreak(()));
            }

            if regs.rip == 0x1000000 {
                info!("This was the injected bp...");
                return Some(SingleThreadStopReason::HwBreak(()));
            }

            info!("Found bp we didn't set lets notify the guest");
            self.inject_bp_to_guest();

            // return Some(SingleThreadStopReason::SwBreak(()));
            return None

            // let mut coreregs: CoreRegs = Default::default();
            // let _ = self.read_registers(&mut coreregs);
            // coreregs.rip += 1;
            // let _ = self.write_registers(&coreregs);
            //
            // return None
        } else {
            info!("No reg info");
        }

        return Some(SingleThreadStopReason::SwBreak(()));
    }
}

impl Target for FirecrackerTarget {
    type Error = ();
    type Arch = GdbArch; // as an example

    #[inline(always)]
    fn base_ops(&mut self) -> BaseOps<Self::Arch, Self::Error> {
        BaseOps::SingleThread(self)
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

impl SingleThreadBase for FirecrackerTarget {
    fn read_registers(
        &mut self,
        regs: &mut CoreRegs,
    ) -> TargetResult<(), Self> { 
        if let Some(cpu_regs) = self.get_regs() {
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

    fn write_registers(
        &mut self,
        regs: &CoreRegs
    ) -> TargetResult<(), Self> {
        info!(".................Writing cpu registers..............");
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
        new_regs.rflags = regs.eflags  as u64;

        if let Some(old_regs) = self.get_regs() {
            info!("Old registers: {old_regs:?}");
            info!("New registers: {new_regs:?}");
        }

        self.set_regs(new_regs);

        Ok(()) 
    }

    fn read_addrs(
        &mut self,
        start_addr: <Self::Arch as Arch>::Usize,
        data: &mut [u8],
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
            let paddr = match self.translate_gva(&vmm.vcpus_handles[0], gaddr) {
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
                ).is_err() {
                    error!("Error while reading memory");
                    return Err(TargetError::NonFatal)
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
            let paddr = match self.translate_gva(&vmm.vcpus_handles[0], gaddr) {
                Some(paddr) if paddr == u64::MIN => gaddr,
                Some(paddr) => paddr,
                None => {
                    info!("Error translating gva");
                    return Err(TargetError::NonFatal)
                }
            };
            let psize = 4096;
            let write_len = std::cmp::min(len as u64 - total_written, psize - (paddr & (psize - 1)));
            if memory 
                .write(
                    &data[total_written as usize..total_written as usize + write_len as usize],
                    GuestAddress(paddr),
                ).is_err() {
                    info!("Error writing memory at {paddr:X}");
                    return Err(TargetError::NonFatal)
            }
            total_written += write_len;
        }

        info!("Done writing address data");

        Ok(())
    }

    #[inline(always)]
    fn support_resume(&mut self) -> Option<SingleThreadResumeOps<Self>> {
        Some(self)
    }
}

impl SingleThreadResume for FirecrackerTarget {
    fn resume(
        &mut self,
        _signal: Option<Signal>,
    ) -> Result<(), Self::Error> {
        info!("Got resume command");
        self.single_step = false;
        self.resume_execution();

        Ok(()) 
    }

    #[inline(always)]
    fn support_single_step(
        &mut self
    ) -> Option<SingleThreadSingleStepOps<'_, Self>> {
        Some(self)
    }
}

impl SingleThreadSingleStep for FirecrackerTarget {
    fn step(
        &mut self,
        _signal: Option<Signal>,
    ) -> Result<(), Self::Error> { 
        info!("Got single step command");
        self.single_step = true;
        self.resume_execution();

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
           self.translate_gva(&vmm.vcpus_handles[0], addr).unwrap()
        };

        if self.sw_breakpoints.contains_key(&physical_addr) {
            return Ok(true);
        }

        let mut saved_register = [0];
        let _ = self.read_addrs(addr, &mut saved_register);
        self.sw_breakpoints.insert(physical_addr, saved_register);
        info!("Inserting SW breakpoint at {physical_addr:X}");
        
        let break_point = [0xCC];
        let _ = self.write_addrs(addr, &break_point);
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
           self.translate_gva(&vmm.vcpus_handles[0], addr).unwrap()
        };
        info!("Removing SW breakpoint at {physical_addr:X}");

        if let Some(removed) = self.sw_breakpoints.remove(&physical_addr) {
            let _ = self.write_addrs(addr, &removed);
            return Ok(true);
        }

        Ok(true)
    }
}
