use std::arch;
use std::sync::{Arc, Mutex};

use gdbstub::arch::Arch;
use gdbstub::common::Signal;
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
use vm_memory::{Bytes, GuestAddress, GuestMemory};

use crate::logger::{info, error};
use crate::{VcpuEvent, VcpuHandle, VcpuResponse, Vmm};


pub struct FirecrackerTarget {
    vmm: Arc<Mutex<Vmm>>,
    pub gdb_event: EventFd,

    hw_breakpoints: Vec<GuestAddress>,

    single_step: bool
}

impl FirecrackerTarget {
    pub fn new(vmm: Arc<Mutex<Vmm>>, gdb_event: EventFd) -> Self {
        Self{
            vmm,
            gdb_event,
            hw_breakpoints: Vec::new(),
            single_step: false,
        }
    }

    fn get_regs(&self) -> Option<kvm_regs> {
        let cpu_handle = &self.vmm
            .lock()
            .expect("error unlocking vmm")
            .vcpus_handles[0];
        
        cpu_handle.send_event(VcpuEvent::SaveState).unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();

        if let VcpuResponse::SavedState(response) = response {
            let regs = response.regs;
            return Some(regs);
        }

        None
    }

    fn update_breakpoints(&self) {
        let cpu_handle = &self.vmm
            .lock()
            .expect("error unlocking vmm")
            .vcpus_handles[0];

        cpu_handle.send_event(VcpuEvent::SetHwBP(self.hw_breakpoints.clone())).unwrap();
        let _ = cpu_handle.response_receiver().recv().unwrap();
    }

    fn resume_execution(&self) {
        let cpu_handle = &self.vmm
            .lock()
            .expect("error unlocking vmm")
            .vcpus_handles[0];

        // Ensure breakpoints are correctly in sync before resuming
        cpu_handle.send_event(VcpuEvent::SetHwBP(self.hw_breakpoints.clone())).unwrap();
        let _ = cpu_handle.response_receiver().recv().unwrap();

        cpu_handle.send_event(VcpuEvent::Resume).unwrap();
        let _ = cpu_handle.response_receiver().recv().unwrap();
    }

    fn set_single_step(&self, enabled: bool) {
        let cpu_handle = &self.vmm
            .lock()
            .expect("error unlocking vmm")
            .vcpus_handles[0];

        // Ensure breakpoints are correctly in sync before resuming
        cpu_handle.send_event(VcpuEvent::EnableSingleStep(enabled)).unwrap();
        let _ = cpu_handle.response_receiver().recv().unwrap();
    }

    fn send_resume(&self) {
        let cpu_handle = &self.vmm
            .lock()
            .expect("error unlocking vmm")
            .vcpus_handles[0];

        // Ensure breakpoints are correctly in sync before resuming
        cpu_handle.send_event(VcpuEvent::Resume).unwrap();
        let _ = cpu_handle.response_receiver().recv().unwrap();
    }

    fn translate_gva(&self, cpu_handle: &VcpuHandle, address: u64) -> Option<u64> {
        info!("Sending event to cpu");
        
        cpu_handle.send_event(VcpuEvent::GvaTranslate(address)).unwrap();
        let response = cpu_handle.response_receiver().recv().unwrap();

        info!("Got event back from cpu");

        if let VcpuResponse::GvaTranslation(response) = response {
            return Some(response);
        }

        None
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
        true
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
        }

        Ok(()) 
    }

    fn write_registers(
        &mut self,
        regs: &CoreRegs
    ) -> TargetResult<(), Self> {
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

        while total_read < len as u64 {
            let gaddr = start_addr + total_read;
            let paddr = match self.translate_gva(&vmm.vcpus_handles[0], gaddr) {
                Some(paddr) => paddr,
                None => return Err(TargetError::NonFatal)
            };
            let psize = 4096;
            let read_len = std::cmp::min(len as u64 - total_read, psize - (paddr & (psize - 1)));
            memory 
                .read(
                    &mut data[total_read as usize..total_read as usize + read_len as usize],
                    GuestAddress(paddr),
                )
                .expect("Error");
            total_read += read_len;
        }

        let real_address = match self.translate_gva(&vmm.vcpus_handles[0], start_addr) {
            Some(res ) if res > 0 => res,
            _ => {
                info!("Error translating gva");
                return Err(TargetError::NonFatal)
            }
        };
        info!("Got real address");

        match memory.read(data, GuestAddress(real_address)) {
            Ok(res) => Ok(res),
            Err(e) => {
                error!("Failed to request ReadMem: {:?}", e);
                Err(TargetError::NonFatal)
            }
        }
    }

    fn write_addrs(
        &mut self,
        start_addr: <Self::Arch as Arch>::Usize,
        data: &[u8],
    ) -> TargetResult<(), Self> {
        let vmm = &self.vmm.lock().expect("Error unlocking vmm");
        let memory = vmm.guest_memory();

        match memory.write(data, GuestAddress(start_addr)) {
            Ok(_) => Ok(()),
            Err(e) => {
                error!("Failed to request ReadMem: {:?}", e);
                Err(TargetError::NonFatal)
            }
        }
    }

    // most targets will want to support at resumption as well...

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
        self.set_single_step(true);
        self.send_resume();
        Ok(()) 
    }
}

impl Breakpoints for FirecrackerTarget {
    // there are several kinds of breakpoints - this target uses software breakpoints
    #[inline(always)]
    fn support_hw_breakpoint(&mut self) -> Option<HwBreakpointOps<Self>> {
        Some(self)
    }
}

impl HwBreakpoint for FirecrackerTarget {
    fn add_hw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        _kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        info!("Setting hw breakpoint..");
        self.hw_breakpoints.push(GuestAddress(addr));
        self.update_breakpoints();

        Ok(true)
    }

    fn remove_hw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        _kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        info!("Removing hw breakpoint..");
        match self.hw_breakpoints.iter().position(|&b| b.0 == addr) {
            None => return Ok(false),
            Some(pos) => self.hw_breakpoints.remove(pos),
        };
        info!("Removed hw breakpoint..");
        self.update_breakpoints();

        Ok(true)
    }
}
