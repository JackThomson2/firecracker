use std::os::fd::RawFd;
use std::mem::{self, offset_of};

use super::target::Error;
use crate::arch::aarch64::regs::{self, arm64_core_reg_id, Aarch64RegisterRef, Aarch64RegisterVec};
use crate::arch::aarch64::vcpu::{get_all_registers, get_registers};
use crate::Vmm;
use kvm_bindings::{
    kvm_regs, user_fpsimd_state, user_pt_regs, KVM_GUESTDBG_USE_HW, KVM_NR_SPSR, KVM_REG_ARM64,
    KVM_REG_ARM64_SYSREG, KVM_REG_ARM64_SYSREG_CRM_MASK, KVM_REG_ARM64_SYSREG_CRN_MASK,
    KVM_REG_ARM64_SYSREG_OP0_MASK, KVM_REG_ARM64_SYSREG_OP1_MASK, KVM_REG_ARM64_SYSREG_OP2_MASK,
    KVM_REG_ARM_CORE, KVM_REG_SIZE_U128, KVM_REG_SIZE_U32, KVM_REG_SIZE_U64,
};

use vm_memory::Bytes;
use kvm_bindings::*;
use kvm_ioctls::{kvm_ioctls_raw::*, VcpuFd};
use vm_memory::{guest_memory, GuestAddress, GuestMemory};
use vmm_sys_util::ioctl::{ioctl_with_mut_ref, ioctl_with_ref};
use vmm_sys_util::errno;

/// Helper method to obtain the size of the register through its id
fn reg_size(reg_id: u64) -> usize {
    2_usize.pow(((reg_id & KVM_REG_SIZE_MASK) >> KVM_REG_SIZE_SHIFT) as u32)
}


/// Sets the value of one register for this vCPU.
///
/// The id of the register is encoded as specified in the kernel documentation
/// for `KVM_SET_ONE_REG`.
///
/// # Arguments
///
/// * `reg_id` - ID of the register for which we are setting the value.
/// * `data` - byte slice where the register value will be written to.
///
/// # Note
///
/// `data` should be equal or bigger then the register size
/// oterwise function will return EINVAL error
pub fn set_one_reg(vcpu_fd: &RawFd, reg_id: u64, data: &[u8]) -> Result<usize, Error> {
    let reg_size = reg_size(reg_id);
    if data.len() < reg_size {
        return Err(Error::KvmIoctlsError(errno::Error::last()));
    }
    let onereg = kvm_one_reg {
        id: reg_id,
        addr: data.as_ptr() as u64,
    };
    // SAFETY: This is safe because we allocated the struct and we know the kernel will read
    // exactly the size of the struct.
    let ret = unsafe { ioctl_with_ref(vcpu_fd, KVM_SET_ONE_REG(), &onereg) };
    if ret < 0 {
        return Err(Error::KvmIoctlsError(errno::Error::last()));
    }
    Ok(reg_size)
}

/// Writes the value of the specified vCPU register into provided buffer.
///
/// The id of the register is encoded as specified in the kernel documentation
/// for `KVM_GET_ONE_REG`.
///
/// # Arguments
///
/// * `reg_id` - ID of the register.
/// * `data` - byte slice where the register value will be written to.
/// # Note
///
/// `data` should be equal or bigger then the register size
/// oterwise function will return EINVAL error
fn get_one_reg(vcpu_fd: &RawFd, reg_id: u64, data: &mut [u8]) -> Result<usize, Error> {
    let reg_size = reg_size(reg_id);
    if data.len() < reg_size {
        return Err(Error::KvmIoctlsError(errno::Error::last()));
    }

    let mut onereg = kvm_one_reg {
        id: reg_id,
        addr: data.as_ptr() as u64,
    };
    // SAFETY: This is safe because we allocated the struct and we know the kernel will read
    // exactly the size of the struct.
    let ret = unsafe { ioctl_with_mut_ref(vcpu_fd, KVM_GET_ONE_REG(), &mut onereg) };
    if ret < 0 {
        return Err(Error::KvmIoctlsError(errno::Error::last()));
    }
    Ok(reg_size)
}

// Following are macros that help with getting the ID of a aarch64 core register.
// The core register are represented by the user_pt_regs structure. Look for it in
// arch/arm64/include/uapi/asm/ptrace.h.
#[macro_export]
/// Get the ofset of a core register
macro_rules! offset_of {
    ($str:ty, $field:ident) => {{
        let tmp: std::mem::MaybeUninit<$str> = std::mem::MaybeUninit::uninit();
        let base = tmp.as_ptr();
        // Avoid warnings when nesting `unsafe` blocks.
        #[allow(unused_unsafe)]
        // SAFETY: The pointer is valid and aligned, just not initialised. Using `addr_of` ensures
        // that we don't actually read from `base` (which would be UB) nor create an intermediate
        // reference.
        let member = unsafe { core::ptr::addr_of!((*base).$field) } as *const u8;
        // Avoid warnings when nesting `unsafe` blocks.
        #[allow(unused_unsafe)]
        // SAFETY: The two pointers are within the same allocated object `tmp`. All requirements
        // from offset_from are upheld.
        unsafe {
            member.offset_from(base as *const u8) as usize
        }
    }};
}

/// Extract the specified bits of a 64-bit integer.
/// For example, to extrace 2 bits from offset 1 (zero based) of `6u64`,
/// following expression should return 3 (`0b11`):
/// `extract_bits_64!(0b0000_0110u64, 1, 2)`
///
macro_rules! extract_bits_64 {
    ($value: tt, $offset: tt, $length: tt) => {
        ($value >> $offset) & (!0u64 >> (64 - $length))
    };
}

macro_rules! extract_bits_64_without_offset {
    ($value: tt, $length: tt) => {
        $value & (!0u64 >> (64 - $length))
    };
}

fn get_sys_reg(id: u64, vcpu: &RawFd) -> Result<u64, Error> {
    //
    // Arm Architecture Reference Manual defines the encoding of
    // AArch64 system registers, see
    // https://developer.arm.com/documentation/ddi0487 (chapter D12).
    // While KVM defines another ID for each AArch64 system register,
    // which is used in calling `KVM_G/SET_ONE_REG` to access a system
    // register of a guest.
    // A mapping exists between the Arm standard encoding and the KVM ID.
    // This function takes the standard u32 ID as input parameter, converts
    // it to the corresponding KVM ID, and call `KVM_GET_ONE_REG` API to
    // get the value of the system parameter.
    //
    //
    //
    let mut bytes = [0_u8; 8];
    get_one_reg(vcpu, id, &mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

const CORE_REG_COUNT: usize = 34;
const CORE_REG_IDS: [u64; CORE_REG_COUNT]  = {
    let mut regs = [0; CORE_REG_COUNT];
    let mut idx = 0;

    let mut off = offset_of!(user_pt_regs, regs);
    while idx < 32 {
        regs[idx] = arm64_core_reg_id!(KVM_REG_SIZE_U64, off);
        idx += 1;
        off += std::mem::size_of::<u64>();
    }

    regs[idx] = offset_of!(user_pt_regs, sp) as u64;
    idx += 1;

    regs[idx] = offset_of!(user_pt_regs, pc) as u64;

    regs
};

pub fn get_regs(vcpu_fd: &VcpuFd) -> Result<kvm_regs, Error> {
    let mut state: kvm_regs = kvm_regs::default();
    let mut register_vec = Aarch64RegisterVec::default();
    get_registers(vcpu_fd, &CORE_REG_IDS, &mut register_vec).map_err(|_| Error::VcpuKvmError)?;

    let mut registers = register_vec.iter();

    // There are 31 user_pt_regs:
    // https://elixir.free-electrons.com/linux/v4.14.174/source/arch/arm64/include/uapi/asm/ptrace.h#L72
    // These actually are the general-purpose registers of the Armv8-a
    // architecture (i.e x0-x30 if used as a 64bit register or w0-30 when used as a 32bit register).
    for i in 0..31 {
        let reg_val = registers.next().unwrap().value();
        state.regs.regs[i] = reg_val;
    }

    state.regs.sp = registers.next().unwrap().value();
    state.regs.pc = registers.next().unwrap().value();

    Ok(state)
}

/// Sets the registers against the provided vcpu fd
pub fn set_regs(vcpu_fd: &VcpuFd, regs: &kvm_regs) -> Result<(), Error> {
    let mut off = offset_of!(user_pt_regs, regs);
    for i in 0..31 {
        vcpu_fd.set_one_reg(
            arm64_core_reg_id!(KVM_REG_SIZE_U64, off), 
            &regs.regs.regs[i].to_le_bytes(),
        );
        off += std::mem::size_of::<u64>();
    }

    let off = offset_of!(user_pt_regs, sp);
    vcpu_fd.set_one_reg(
        arm64_core_reg_id!(KVM_REG_SIZE_U64, off),
        &regs.regs.sp.to_le_bytes(),
    );

    let off = offset_of!(user_pt_regs, pc);
    vcpu_fd.set_one_reg(
        arm64_core_reg_id!(KVM_REG_SIZE_U64, off),
        &regs.regs.pc.to_le_bytes(),
    );

    Ok(())
}

/// Manually translate gva on aarch64
pub fn translate_gva(vcpu_fd: &RawFd, gva: u64, vmm: &Vmm) -> Result<u64, Error> {
    let tcr_el1: u64 = get_sys_reg(regs::TCR_EL1, vcpu_fd)?;
    let ttbr1_el1: u64 = get_sys_reg(regs::TTBR1_EL1, vcpu_fd)?;
    let id_aa64mmfr0_el1: u64 = get_sys_reg(regs::ID_AA64MMFR0_EL1, vcpu_fd)?;
    // Bit 55 of the VA determines the range, high (0xFFFxxx...)
    // or low (0x000xxx...).
    let high_range = extract_bits_64!(gva, 55, 1);
    if high_range == 0 {
        return Ok(gva);
    }
    // Granule size
    let tg = extract_bits_64!(tcr_el1, 30, 2);
    // We only support 4kb pages
    if tg == 1 || tg == 3 {
        panic!("Unsupported page size");
    }

    // High range size offset
    let tsz = extract_bits_64!(tcr_el1, 16, 6);
    if tsz == 0 {
        return Ok(gva);
    }

    // VA size is determined by TCR_BL1.T1SZ
    let va_size = 64 - tsz;
    let stride = 9;

    // Starting level of walking
    let mut level = 4 - (va_size - 4) / 9;
    // PA or IPA size is determined
    let tcr_ips = extract_bits_64!(tcr_el1, 32, 3);
    let pa_range = extract_bits_64_without_offset!(id_aa64mmfr0_el1, 4);
    // The IPA size in TCR_BL1 and PA Range in ID_AA64MMFR0_EL1 should match.
    // To be safe, we use the minimum value if they are different.
    let pa_range = std::cmp::min(tcr_ips, pa_range);
    // PA size in bits
    let pa_size = match pa_range {
        0 => 32,
        1 => 36,
        2 => 40,
        3 => 42,
        4 => 44,
        5 => 48,
        6 => 52,
        _ => {
            return Ok(0)
        }
    };
    let indexmask_grainsize = (!0u64) >> (64 - (stride + 3));
    let mut indexmask = (!0u64) >> (64 - (va_size - (stride * (4 - level))));

    // Indication of 48-bits (0) or 52-bits (1) for FEAT_LPA2
    let ds = extract_bits_64!(tcr_el1, 59, 1);
    // If FEAT_LPA2 is present, the translation table descriptor holds
    // 50 bits of the table address of next level.
    // Otherwise, it is 48 bits.
    let descaddrmask = if ds == 1 {
        !0u64 >> (64 - 50) // mask with 50 least significant bits
    } else {
        !0u64 >> (64 - 48) // mask with 48 least significant bits
    };
    let descaddrmask = descaddrmask & !indexmask_grainsize;
    // Translation table base address
    let mut descaddr: u64 = extract_bits_64_without_offset!(ttbr1_el1, 48);
    // In the case of FEAT_LPA and FEAT_LPA2, the initial translation table
    // address bits [48:51] comes from TTBR1_EL1 bits [2:5].
    if pa_size == 52 {
        descaddr |= extract_bits_64!(ttbr1_el1, 2, 4) << 48;
    }
    // Loop through tables of each level
    loop {
        // Table offset for current level
        let table_offset: u64 = (gva >> (stride * (4 - level))) & indexmask;
        descaddr |= table_offset;
        descaddr &= !7u64;
        let mut buf = [0; 8];
        vmm.guest_memory()
            .read(&mut buf, GuestAddress(descaddr))
            .unwrap();
        let descriptor = u64::from_le_bytes(buf);
        descaddr = descriptor & descaddrmask;
        // In the case of FEAT_LPA, the next-level translation table address
        // bits [48:51] comes from bits [12:15] of the current descriptor.
        // For FEAT_LPA2, the next-level translation table address
        // bits [50:51] comes from bits [8:9] of the current descriptor,
        // bits [48:49] comes from bits [48:49] of the descriptor which was
        // handled previously.
        if pa_size == 52 {
            if ds == 1 {
                // FEAT_LPA2
                descaddr |= extract_bits_64!(descriptor, 8, 2) << 50;
            } else {
                // FEAT_LPA
                descaddr |= extract_bits_64!(descriptor, 12, 4) << 48;
            }
        }
        if (descriptor & 2) != 0 && (level < 3) {
            // This is a table entry. Go down to next level.
            level += 1;
            indexmask = indexmask_grainsize;
            continue;
        }
        break;
    }
    // We have reached either:
    // - a page entry at level 3 or
    // - a block entry at level 1 or 2
    let page_size = 1u64 << ((stride * (4 - level)) + 3);
    descaddr &= !(page_size - 1);
    descaddr |= gva & (page_size - 1);
    Ok(descaddr)
}
