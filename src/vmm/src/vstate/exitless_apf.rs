// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exitless Async Page Fault support via eventfd and shared ring buffer.
//!
//! The kernel exposes a single shared page (`kvm_apf_shared_page`) containing
//! two rings — notification (kernel→userspace) and completion (userspace→kernel).
//! Userspace passes the address of this page via `KVM_SET_APF_EVENTFD`.
//!
//! Notification: KVM writes APF entries to the notify ring and signals the
//! notification eventfd.
//!
//! Completion: The handler writes resolved GPAs to the completion ring and
//! signals the completion eventfd. KVM drains the ring internally.
//!
//! The shared page is backed by a memfd so it can be sent to the UFFD handler
//! process. Both Firecracker and the handler mmap the same memfd, and the
//! kernel writes to the same physical pages via the address passed in the ioctl.

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::io::OwnedFd;
use std::sync::atomic::{AtomicU32, Ordering};

use vmm_sys_util::eventfd::EventFd;
use vmm_sys_util::ioctl::ioctl_with_ref;
use vmm_sys_util::ioctl_iow_nr;

const KVMIO: u32 = 0xAE;

/// `KVM_SET_APF_EVENTFD` ioctl — registers eventfds for exitless APF.
mod apf_eventfd_ioctl {
    use super::*;
    ioctl_iow_nr!(KVM_SET_APF_EVENTFD, KVMIO, 0xd9, KvmApfEventfd);
}
use apf_eventfd_ioctl::KVM_SET_APF_EVENTFD;

/// Matches kernel `KVM_APF_RING_SIZE`.
pub const KVM_APF_RING_SIZE: usize = 32;

/// Matches kernel `struct kvm_apf_eventfd`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct KvmApfEventfd {
    /// Notification eventfd file descriptor.
    pub fd: i32,
    /// Completion eventfd file descriptor.
    pub complete_fd: i32,
    /// Address of the shared page.
    pub page_addr: u64,
    /// Flags (currently unused).
    pub flags: u32,
    /// Padding for alignment.
    pub padding: u32,
}

/// Matches kernel `struct kvm_apf_ring_entry`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct KvmApfRingEntry {
    /// Guest physical address.
    pub gpa: u64,
    /// Entry flags.
    pub flags: u64,
}

/// Matches kernel `struct kvm_apf_ring`.
/// Producer writes head, consumer writes tail.
#[repr(C)]
pub struct KvmApfRing {
    /// Producer index.
    pub head: AtomicU32,
    /// Consumer index.
    pub tail: AtomicU32,
    /// Reserved field.
    pub reserved: u32,
    /// Padding for alignment.
    pub padding: u32,
    /// Ring entries.
    pub entries: [KvmApfRingEntry; KVM_APF_RING_SIZE],
}

// Debug cannot be derived due to AtomicU32 fields.
impl std::fmt::Debug for KvmApfRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvmApfRing")
            .field("head", &self.head.load(Ordering::Relaxed))
            .field("tail", &self.tail.load(Ordering::Relaxed))
            .finish()
    }
}

/// Matches kernel `struct kvm_apf_shared_page`.
/// Both rings live in a single PAGE_SIZE mmap.
#[repr(C)]
pub struct KvmApfSharedPage {
    /// Notification ring (kernel → userspace).
    pub notify: KvmApfRing,
    /// Completion ring (userspace → kernel).
    pub complete: KvmApfRing,
}

// Debug cannot be derived due to KvmApfRing.
impl std::fmt::Debug for KvmApfSharedPage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvmApfSharedPage")
            .field("notify", &self.notify)
            .field("complete", &self.complete)
            .finish()
    }
}

impl KvmApfRing {
    /// Returns true if the ring has entries to consume.
    pub fn has_entries(&self) -> bool {
        self.head.load(Ordering::Acquire) != self.tail.load(Ordering::Relaxed)
    }

    /// Pop an entry (consumer side: advances tail).
    pub fn pop(&self) -> Option<KvmApfRingEntry> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        if head == tail {
            return None;
        }
        let entry = self.entries[tail as usize];
        self.tail
            .store((tail + 1) % KVM_APF_RING_SIZE as u32, Ordering::Release);
        Some(entry)
    }

    /// Push an entry (producer side: advances head). Returns false if full.
    ///
    /// # Safety
    /// Caller must ensure exclusive producer access.
    pub unsafe fn push(&self, entry: KvmApfRingEntry) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let next = (head + 1) % KVM_APF_RING_SIZE as u32;
        if next == tail {
            return false;
        }
        // SAFETY: Caller guarantees exclusive producer access. The pointer
        // arithmetic is within the bounds of the entries array.
        unsafe {
            let slot = (self as *const KvmApfRing as *mut KvmApfRing)
                .cast::<u8>()
                .add(std::mem::offset_of!(KvmApfRing, entries))
                .cast::<KvmApfRingEntry>()
                .add(head as usize);
            std::ptr::write(slot, entry);
        }
        self.head.store(next, Ordering::Release);
        true
    }
}

/// Exitless APF context for a single vCPU.
///
/// The shared page is backed by a memfd created upfront. The kernel, Firecracker,
/// and the UFFD handler all mmap the same memfd, so writes from any side are
/// visible to all others.
pub struct ExitlessApfContext {
    /// Notification eventfd (kernel → userspace)
    eventfd: EventFd,
    /// Completion eventfd (userspace → kernel)
    complete_eventfd: EventFd,
    /// Memfd backing the shared page
    memfd: OwnedFd,
    /// Shared page containing both rings (mmap of memfd)
    shared_page: *mut KvmApfSharedPage,
    /// vCPU fd for cleanup
    vcpu_fd: RawFd,
}

impl std::fmt::Debug for ExitlessApfContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExitlessApfContext")
            .field("vcpu_fd", &self.vcpu_fd)
            .finish()
    }
}

// SAFETY: Ring buffer pointer is valid for the lifetime of the context
// and access is synchronized via atomic operations.
unsafe impl Send for ExitlessApfContext {}
unsafe impl Sync for ExitlessApfContext {}

impl ExitlessApfContext {
    /// Create a new exitless APF context for the given vCPU fd.
    pub fn new(vcpu_fd: RawFd) -> io::Result<Self> {
        let eventfd = EventFd::new(libc::EFD_NONBLOCK)?;
        let complete_eventfd = EventFd::new(libc::EFD_NONBLOCK)?;

        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;

        // Create a memfd to back the shared page. This is the single source
        // of truth — kernel, Firecracker, and handler all mmap this same fd.
        let raw_memfd = unsafe { libc::memfd_create(c"apf_shared".as_ptr(), libc::MFD_CLOEXEC) };
        if raw_memfd < 0 {
            return Err(io::Error::last_os_error());
        }
        let memfd = unsafe { OwnedFd::from_raw_fd(raw_memfd) };
        if unsafe { libc::ftruncate(memfd.as_raw_fd(), page_size as libc::off_t) } < 0 {
            return Err(io::Error::last_os_error());
        }

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                page_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                memfd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        unsafe { std::ptr::write_bytes(ptr, 0, page_size) };
        let shared_page = ptr as *mut KvmApfSharedPage;

        let apf_eventfd = KvmApfEventfd {
            fd: eventfd.as_raw_fd(),
            complete_fd: complete_eventfd.as_raw_fd(),
            page_addr: shared_page as u64,
            flags: 0,
            padding: 0,
        };

        let ret = unsafe { ioctl_with_ref(&vcpu_fd, KVM_SET_APF_EVENTFD(), &apf_eventfd) };
        if ret < 0 {
            let err = io::Error::last_os_error();
            unsafe { libc::munmap(ptr, page_size) };
            return Err(err);
        }

        Ok(Self {
            eventfd,
            complete_eventfd,
            memfd,
            shared_page,
            vcpu_fd,
        })
    }

    /// Returns a reference to the notification eventfd.
    pub fn eventfd(&self) -> &EventFd {
        &self.eventfd
    }

    /// Returns the raw fd of the notification eventfd.
    pub fn eventfd_fd(&self) -> RawFd {
        self.eventfd.as_raw_fd()
    }

    /// Returns a reference to the completion eventfd.
    pub fn complete_eventfd(&self) -> &EventFd {
        &self.complete_eventfd
    }

    /// Returns the raw fd of the completion eventfd.
    pub fn complete_eventfd_fd(&self) -> RawFd {
        self.complete_eventfd.as_raw_fd()
    }

    /// Returns true if there are pending notification entries.
    pub fn has_pending(&self) -> bool {
        unsafe { (*self.shared_page).notify.has_entries() }
    }

    /// Pop the next notification entry, if any.
    pub fn pop_entry(&self) -> Option<KvmApfRingEntry> {
        unsafe { (*self.shared_page).notify.pop() }
    }

    /// Signal completion of an APF by writing GPA to the completion ring
    /// and signaling the completion eventfd.
    pub fn signal_complete(&self, gpa: u64) -> bool {
        let entry = KvmApfRingEntry { gpa, flags: 0 };
        let pushed = unsafe { (*self.shared_page).complete.push(entry) };
        if pushed {
            let _ = self.complete_eventfd.write(1);
        }
        pushed
    }

    /// Drain the notification eventfd counter.
    pub fn drain_eventfd(&self) {
        let _ = self.eventfd.read();
    }

    /// Returns fds to send to the UFFD handler:
    /// (notify_eventfd, complete_eventfd, shared_page_memfd)
    ///
    /// The handler mmaps the same memfd, so all three parties (kernel,
    /// Firecracker, handler) share the same physical pages.
    pub fn fds_for_handler(&self) -> (RawFd, RawFd, RawFd) {
        (
            self.eventfd.as_raw_fd(),
            self.complete_eventfd.as_raw_fd(),
            self.memfd.as_raw_fd(),
        )
    }
}

impl Drop for ExitlessApfContext {
    fn drop(&mut self) {
        let dereg = KvmApfEventfd {
            fd: -1,
            complete_fd: -1,
            page_addr: 0,
            flags: 0,
            padding: 0,
        };
        unsafe {
            ioctl_with_ref(&self.vcpu_fd, KVM_SET_APF_EVENTFD(), &dereg);
        }
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        unsafe {
            libc::munmap(self.shared_page as *mut libc::c_void, page_size);
        }
        // memfd closed automatically by OwnedFd drop
    }
}
