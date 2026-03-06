// Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::undocumented_unsafe_blocks,
    clippy::ptr_as_ptr,
    clippy::cast_possible_wrap,
    // Not everything is used by both binaries
    dead_code
)]

mod userfault_bitmap;

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Write};
use std::num::NonZero;
use std::os::fd::RawFd;
use std::os::unix::io::{AsRawFd, FromRawFd, IntoRawFd};
use std::os::unix::net::UnixStream;
use std::ptr;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::spawn;
use std::time::Duration;

use async_channel::{unbounded, Sender};
use serde::{Deserialize, Serialize};
use userfaultfd::{Error, Event, Uffd};
use vmm_sys_util::ioctl::ioctl_with_mut_ref;
use vmm_sys_util::ioctl_iowr_nr;
use vmm_sys_util::sock_ctrl_msg::ScmSocket;

use crate::uffd_utils::userfault_bitmap::UserfaultBitmap;

// TODO: remove when UFFDIO_CONTINUE for guest_memfd is available in the crate
#[repr(C)]
struct uffdio_continue {
    range: uffdio_range,
    mode: u64,
    mapped: u64,
}

ioctl_iowr_nr!(UFFDIO_CONTINUE, 0xAA, 0x7, uffdio_continue);

// Exitless APF support — must match kernel UAPI definitions
pub const KVM_APF_RING_SIZE: usize = 32;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct KvmApfRingEntry {
    pub gpa: u64,
    pub flags: u64,
}

#[repr(C)]
pub struct KvmApfRing {
    pub head: AtomicU32,
    pub tail: AtomicU32,
    pub reserved: u32,
    pub padding: u32,
    pub entries: [KvmApfRingEntry; KVM_APF_RING_SIZE],
}

/// Matches kernel `struct kvm_apf_shared_page` — both rings in one page.
#[repr(C)]
pub struct KvmApfSharedPage {
    pub notify: KvmApfRing,
    pub complete: KvmApfRing,
}

impl KvmApfRing {
    pub fn pop(&self) -> Option<KvmApfRingEntry> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        if head == tail {
            return None;
        }
        let entry = self.entries[tail as usize];
        self.tail.store((tail + 1) % KVM_APF_RING_SIZE as u32, Ordering::Release);
        Some(entry)
    }
}

/// Per-vCPU exitless APF context (handler side).
/// Receives a single shared-page memfd from Firecracker containing both rings.
pub struct ExitlessApfVcpu {
    pub eventfd: RawFd,
    pub complete_eventfd: RawFd,
    pub shared_page: *mut KvmApfSharedPage,
    buff: [u8; 8],
}

impl ExitlessApfVcpu {
    /// Create from fds sent by Firecracker:
    /// (notify_eventfd, complete_eventfd, shared_page_memfd)
    pub fn from_fds(
        eventfd: RawFd,
        complete_eventfd: RawFd,
        shared_page_memfd: RawFd,
    ) -> std::io::Result<Self> {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                page_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                shared_page_memfd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        // Close the memfd — the mapping keeps it alive
        unsafe { libc::close(shared_page_memfd) };

        Ok(Self {
            eventfd,
            complete_eventfd,
            shared_page: ptr as *mut KvmApfSharedPage,
            buff: [0; 8]
        })
    }

    #[inline]
    pub fn notify_ring(&self) -> &KvmApfRing {
        unsafe { &(*self.shared_page).notify }
    }

    #[inline]
    fn complete_ring(&self) -> &KvmApfRing {
        unsafe { &(*self.shared_page).complete }
    }

    #[inline]
    pub fn drain_eventfd(&mut self) {
        unsafe { libc::read(self.eventfd, self.buff.as_mut_ptr() as *mut c_void, 8) };
    }

    /// Signal APF completion via the completion ring + eventfd.
    #[inline]
    pub fn signal_ready(&self, gpa: u64) {
        let ring = self.complete_ring();
        let head = ring.head.load(Ordering::Relaxed);
        let tail = ring.tail.load(Ordering::Acquire);
        let next = (head + 1) % KVM_APF_RING_SIZE as u32;
        if next == tail {
            return; // Ring full
        }
        let entry = KvmApfRingEntry { gpa, flags: 0 };
        unsafe {
            let slot = &raw const ring.entries[head as usize] as *mut KvmApfRingEntry;
            ptr::write(slot, entry);
        }
        ring.head.store(next, Ordering::Release);
        let val: u64 = 1;
        unsafe { libc::write(self.complete_eventfd, &val as *const u64 as *const c_void, 8) };
    }
}

impl Drop for ExitlessApfVcpu {
    fn drop(&mut self) {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        unsafe {
            libc::munmap(self.shared_page as *mut c_void, page_size);
            libc::close(self.eventfd);
            libc::close(self.complete_eventfd);
        }
    }
}

/// Wrapper to send a raw shared-page pointer across threads.
///
/// # Safety
/// The shared page is backed by a memfd mmap that outlives all spawned tasks.
struct SendableSharedPage(*mut KvmApfSharedPage);
unsafe impl Send for SendableSharedPage {}

/// A deferred APF completion request.
struct ApfCompletion {
    shared_page: SendableSharedPage,
    complete_eventfd: RawFd,
    gpa: u64,
}
unsafe impl Send for ApfCompletion {}

/// Signal APF completion via the shared completion ring + eventfd.
///
/// Thread-safe: only touches a shared mmap and a raw fd write.
fn signal_apf_ready(shared_page: SendableSharedPage, complete_eventfd: RawFd, gpa: u64) {
    let ring = unsafe { &(*shared_page.0).complete };
    let head = ring.head.load(Ordering::Relaxed);
    let tail = ring.tail.load(Ordering::Acquire);
    let next = (head + 1) % KVM_APF_RING_SIZE as u32;
    if next == tail {
        return; // Ring full
    }
    let entry = KvmApfRingEntry { gpa, flags: 0 };
    unsafe {
        let slot = &raw const ring.entries[head as usize] as *mut KvmApfRingEntry;
        ptr::write(slot, entry);
    }
    ring.head.store(next, Ordering::Release);
    let val: u64 = 1;
    unsafe { libc::write(complete_eventfd, &val as *const u64 as *const c_void, 8) };
}

#[repr(C)]
struct uffdio_range {
    start: u64,
    len: u64,
}

pub fn uffd_continue(uffd: RawFd, fault_addr: u64, len: u64) -> std::io::Result<()> {
    let mut cont = uffdio_continue {
        range: uffdio_range {
            start: fault_addr,
            len,
        },
        mode: 0, // Normal continuation mode
        mapped: 0,
    };

    let ret = unsafe { ioctl_with_mut_ref(&uffd, UFFDIO_CONTINUE(), &mut cont) };

    if ret == -1 {
        return Err(std::io::Error::last_os_error());
    }

    Ok(())
}

// This is the same with the one used in src/vmm.
/// This describes the mapping between Firecracker base virtual address and offset in the
/// buffer or file backend for a guest memory region. It is used to tell an external
/// process/thread where to populate the guest memory data for this range.
///
/// E.g. Guest memory contents for a region of `size` bytes can be found in the backend
/// at `offset` bytes from the beginning, and should be copied/populated into `base_host_address`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GuestRegionUffdMapping {
    /// Base host virtual address where the guest memory contents for this region
    /// should be copied/populated.
    pub base_host_virt_addr: u64,
    /// Region size.
    pub size: usize,
    /// Offset in the backend file/buffer where the region contents are.
    pub offset: u64,
    /// Guest physical address start for this region.
    #[serde(default)]
    pub gpa_start: u64,
    /// The configured page size for this memory region.
    pub page_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, bitcode::Decode)]
pub struct FaultRequest {
    /// vCPU that encountered the fault
    pub vcpu: u32,
    /// Offset in guest_memfd where the fault occured
    pub offset: u64,
    /// Flags
    pub flags: u64,
    /// Async PF token
    pub gpa: Option<u64>,
}

impl FaultRequest {
    pub fn into_reply(self, len: u64) -> FaultReply {
        FaultReply {
            vcpu: Some(self.vcpu),
            offset: self.offset,
            len,
            flags: self.flags,
            gpa: self.gpa,
            zero: false,
        }
    }
}

/// FaultReply
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, bitcode::Encode)]
pub struct FaultReply {
    /// vCPU that encountered the fault, from `FaultRequest` (if present, otherwise 0)
    pub vcpu: Option<u32>,
    /// Offset in guest_memfd where population started
    pub offset: u64,
    /// Length of populated area
    pub len: u64,
    /// Flags, must be copied from `FaultRequest`, otherwise 0
    pub flags: u64,
    /// Async PF token, must be copied from `FaultRequest`, otherwise None
    pub gpa: Option<u64>,
    /// Whether the populated pages are zero pages
    pub zero: bool,
}

/// UffdMsgFromFirecracker
#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum UffdMsgFromFirecracker {
    /// Mappings
    Mappings(Vec<GuestRegionUffdMapping>),
    /// FaultReq
    FaultReq(FaultRequest),
}

/// UffdMsgToFirecracker
#[derive(Serialize, Deserialize, Debug, bitcode::Encode)]
#[serde(untagged)]
pub enum UffdMsgToFirecracker {
    /// FaultRep
    FaultRep(FaultReply),
}

impl GuestRegionUffdMapping {
    fn contains(&self, fault_page_addr: u64) -> bool {
        fault_page_addr >= self.base_host_virt_addr
            && fault_page_addr < self.base_host_virt_addr + self.size as u64
    }
}

#[derive(Debug)]
pub struct UffdHandler {
    pub mem_regions: Vec<GuestRegionUffdMapping>,
    pub page_size: usize,
    backing_buffer: *const u8,
    pub uffd: Uffd,
    removed_pages: HashSet<u64>,
    pub guest_memfd: Option<File>,
    pub guest_memfd_addr: Option<*mut u8>,
    pub userfault_bitmap: Option<UserfaultBitmap>,
}

impl UffdHandler {
    fn try_get_mappings_and_file(
        stream: &UnixStream,
    ) -> Result<(String, Option<File>), std::io::Error> {
        let mut message_buf = vec![0u8; 1024];
        let (bytes_read, file) = stream.recv_with_fd(&mut message_buf[..])?;
        message_buf.resize(bytes_read, 0);

        // We do not expect to receive non-UTF-8 data from Firecracker, so this is probably
        // an error we can't recover from. Just immediately abort
        let body = String::from_utf8(message_buf.clone()).unwrap_or_else(|_| {
            panic!(
                "Received body is not a utf-8 valid string. Raw bytes received: {message_buf:#?}"
            )
        });
        Ok((body, file))
    }

    fn get_mappings_and_file(stream: &UnixStream) -> (String, File) {
        // Sometimes, reading from the stream succeeds but we don't receive any
        // UFFD descriptor. We don't really have a good understanding why this is
        // happening, but let's try to be a bit more robust and retry a few times
        // before we declare defeat.
        for _ in 1..=5 {
            match Self::try_get_mappings_and_file(stream) {
                Ok((body, Some(file))) => {
                    return (body, file);
                }
                Ok((body, None)) => {
                    println!("Didn't receive UFFD over socket. We received: '{body}'. Retrying...");
                }
                Err(err) => {
                    println!("Could not get UFFD and mapping from Firecracker: {err}. Retrying...");
                }
            }
            std::thread::sleep(Duration::from_millis(100));
        }

        panic!("Could not get UFFD and mappings after 5 retries");
    }

    fn mmap_helper(len: libc::size_t, fd: libc::c_int) -> *mut libc::c_void {
        // SAFETY: `mmap` is a safe function to call with valid parameters.
        let ret = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };

        assert_ne!(ret, libc::MAP_FAILED);

        ret
    }

    pub fn from_mappings(
        mappings: Vec<GuestRegionUffdMapping>,
        uffd: File,
        guest_memfd: Option<File>,
        userfault_bitmap_memfd: Option<File>,
        backing_buffer: *const u8,
        size: usize,
    ) -> Self {
        let memsize: usize = mappings.iter().map(|r| r.size).sum();
        // Page size is the same for all memory regions, so just grab the first one
        let first_mapping = mappings.first().unwrap_or_else(|| {
            panic!(
                "Cannot get the first mapping. Mappings size is {}.",
                mappings.len()
            )
        });
        let page_size = first_mapping.page_size;

        // Make sure memory size matches backing data size.
        assert_eq!(memsize, size);
        assert!(page_size.is_power_of_two());

        let uffd = unsafe { Uffd::from_raw_fd(uffd.into_raw_fd()) };

        match (&guest_memfd, &userfault_bitmap_memfd) {
            (Some(guestmem_file), Some(bitmap_file)) => {
                let guest_memfd_addr =
                    Some(Self::mmap_helper(size, guestmem_file.as_raw_fd()).cast::<u8>());

                let bitmap_ptr =
                    Self::mmap_helper(size, bitmap_file.as_raw_fd()).cast::<AtomicU64>();

                // SAFETY: The bitmap pointer is valid and the size is correct.
                let userfault_bitmap = Some(unsafe {
                    UserfaultBitmap::new(bitmap_ptr, memsize, NonZero::new(page_size).unwrap())
                });

                Self {
                    mem_regions: mappings,
                    page_size,
                    backing_buffer,
                    uffd,
                    removed_pages: HashSet::new(),
                    guest_memfd,
                    guest_memfd_addr,
                    userfault_bitmap,
                }
            }
            (None, None) => Self {
                mem_regions: mappings,
                page_size,
                backing_buffer,
                uffd,
                removed_pages: HashSet::new(),
                guest_memfd: None,
                guest_memfd_addr: None,
                userfault_bitmap: None,
            },
            (_, _) => {
                panic!(
                    "Only both guest_memfd and userfault_bitmap_memfd can be set at the same time."
                );
            }
        }
    }

    pub fn read_event(&mut self) -> Result<Option<Event>, Error> {
        self.uffd.read_event()
    }

    pub fn mark_range_removed(&mut self, start: u64, end: u64) {
        let pfn_start = start / self.page_size as u64;
        let pfn_end = end / self.page_size as u64;

        for pfn in pfn_start..pfn_end {
            self.removed_pages.insert(pfn);
        }
    }

    /// Convert guest physical address to file offset
    pub fn gpa_to_offset(&self, gpa: u64) -> Option<u64> {
        for region in &self.mem_regions {
            if gpa >= region.gpa_start && gpa < region.gpa_start + region.size as u64 {
                return Some(gpa - region.gpa_start + region.offset);
            }
        }
        None
    }

    pub fn addr_to_offset(&self, addr: *mut u8) -> u64 {
        let addr = addr as u64;
        for region in &self.mem_regions {
            if region.contains(addr) {
                return addr - region.base_host_virt_addr + region.offset;
            }
        }

        panic!(
            "Could not find addr: {:#x} within guest region mappings.",
            addr
        );
    }

    pub fn serve_pf(&mut self, addr: *mut u8, len: usize) -> bool {
        // Find the start of the page that the current faulting address belongs to.
        let dst = (addr as usize & !(self.page_size - 1)) as *mut libc::c_void;
        let fault_page_addr = dst as u64;
        let fault_pfn = fault_page_addr / self.page_size as u64;

        if self.removed_pages.contains(&fault_pfn) {
            return self.zero_out(fault_page_addr);
        }

        for region in self.mem_regions.iter() {
            if region.contains(fault_page_addr) {
                return self.populate_from_file(&region.clone(), fault_page_addr, len);
            }
        }

        panic!(
            "Could not find addr: {:?} within guest region mappings.",
            addr
        );
    }

    pub fn size(&self) -> usize {
        self.mem_regions.iter().map(|r| r.size).sum()
    }

    pub fn populate_via_write(&mut self, offset: usize, len: usize) -> usize {
        // man 2 write:
        //
        //    On Linux, write() (and similar system calls) will transfer at most
        //    0x7ffff000 (2,147,479,552) bytes, returning the number of bytes
        //    actually transferred.  (This is true on both 32-bit and 64-bit
        //    systems.)
        const MAX_WRITE_LEN: usize = 2_147_479_552;

        assert!(
            offset.checked_add(len).unwrap() <= self.size(),
            "{} + {} >= {}",
            offset,
            len,
            self.size()
        );

        let mut total_written = 0;

        while total_written < len {
            let src = unsafe { self.backing_buffer.add(offset + total_written) };
            let len_to_write = (len - total_written).min(MAX_WRITE_LEN);
            let bytes_written = unsafe {
                libc::pwrite64(
                    self.guest_memfd.as_ref().unwrap().as_raw_fd(),
                    src.cast(),
                    len_to_write,
                    (offset + total_written) as libc::off64_t,
                )
            };

            let bytes_written = match bytes_written {
                -1 if vmm_sys_util::errno::Error::last().errno() == libc::ENOSPC => 0,
                written @ 0.. => written as usize,
                _ => panic!("{:?}", std::io::Error::last_os_error()),
            };

            self.userfault_bitmap
                .as_mut()
                .unwrap()
                .reset_addr_range(offset + total_written, bytes_written);

            total_written += bytes_written;

            if bytes_written != len_to_write {
                break;
            }
        }

        total_written
    }

    fn populate_via_uffdio_copy(&self, src: *const u8, dst: u64, len: usize) -> bool {
        unsafe {
            match self.uffd.copy(src.cast(), dst as *mut _, len, true) {
                // Make sure the UFFD copied some bytes.
                Ok(value) => assert!(value > 0),
                // Catch EAGAIN errors, which occur when a `remove` event lands in the UFFD
                // queue while we're processing `pagefault` events.
                // The weird cast is because the `bytes_copied` field is based on the
                // `uffdio_copy->copy` field, which is a signed 64 bit integer, and if something
                // goes wrong, it gets set to a -errno code. However, uffd-rs always casts this
                // value to an unsigned `usize`, which scrambled the errno.
                Err(Error::PartiallyCopied(bytes_copied))
                    if bytes_copied == 0 || bytes_copied == (-libc::EAGAIN) as usize =>
                {
                    return false;
                }
                Err(Error::CopyFailed(errno))
                    if std::io::Error::from(errno).raw_os_error().unwrap() == libc::EEXIST => {}
                Err(e) => {
                    panic!("Uffd copy failed: {e:?}");
                }
            }
        };

        true
    }

    fn populate_via_memcpy(&mut self, src: *const u8, dst: u64, offset: usize, len: usize) -> bool {
        let dst_memcpy = unsafe {
            self.guest_memfd_addr
                .expect("no guest_memfd addr")
                .add(offset)
        };

        unsafe {
            std::ptr::copy_nonoverlapping(src, dst_memcpy, len);
        }

        self.userfault_bitmap
            .as_mut()
            .unwrap()
            .reset_addr_range(offset, len);

        uffd_continue(self.uffd.as_raw_fd(), dst, len as u64).expect("uffd_continue");

        true
    }

    fn populate_from_file(
        &mut self,
        region: &GuestRegionUffdMapping,
        dst: u64,
        len: usize,
    ) -> bool {
        let offset = (region.offset + dst - region.base_host_virt_addr) as usize;
        let src = unsafe { self.backing_buffer.add(offset) };

        match self.guest_memfd {
            Some(_) => self.populate_via_memcpy(src, dst, offset, len),
            None => self.populate_via_uffdio_copy(src, dst, len),
        }
    }

    fn zero_out(&mut self, addr: u64) -> bool {
        match unsafe { self.uffd.zeropage(addr as *mut _, self.page_size, true) } {
            Ok(_) => true,
            Err(Error::ZeropageFailed(error)) if error as i32 == libc::EAGAIN => false,
            r => panic!("Unexpected zeropage result: {:?}", r),
        }
    }
}

struct UffdMsgIterBitcode {
    stream: Arc<Mutex<UnixStream>>,
    buffer: Vec<u8>,
    current_pos: usize,
}

impl UffdMsgIterBitcode {
    fn new(stream: Arc<Mutex<UnixStream>>) -> Self {
        Self {
            stream,
            buffer: vec![0u8; 4096],
            current_pos: 0,
        }
    }

    fn expected_size(&self) -> Option<u32> {
        if self.current_pos < 4 {
            return None
        }

        Some(u32::from_le_bytes(self.buffer[..4].try_into().unwrap()))
    }

    fn can_decode(&self) -> bool {
        let Some(expected_size) = self.expected_size() else {
            return false;
        };

        self.current_pos - 4 >= expected_size as usize
    }
}

impl Iterator for UffdMsgIterBitcode {
    type Item = FaultRequest;

    fn next(&mut self) -> Option<Self::Item> {
        match self.stream.lock().unwrap().read(&mut self.buffer[self.current_pos..]) {
            Ok(bytes_read) => self.current_pos += bytes_read,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Continue with existing buffer data
            }
            Err(e) => panic!("Failed to read from stream: {e}"),
        }

        if !self.can_decode() {
            return None;
        }

        let size = self.expected_size().unwrap() as usize;
        let found = bitcode::decode(&self.buffer[4..4 + size]).unwrap();

        self.buffer.copy_within(4 + size..self.current_pos, 0);
        self.current_pos -= 4 + size;

        Some(found)
    }
}

pub struct Runtime {
    stream: Arc<Mutex<UnixStream>>,
    apf_stream: Arc<Mutex<UnixStream>>,
    backing_file: File,
    backing_memory: *mut u8,
    backing_memory_size: usize,
    handler: UffdHandler,
    message_buffer: [u8; 4096],
    encode_buffer: bitcode::Buffer,
    /// Exitless APF contexts keyed by eventfd
    exitless_vcpus: HashMap<RawFd, ExitlessApfVcpu>,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("backing_memory_size", &self.backing_memory_size)
            .finish_non_exhaustive()
    }
}

impl Runtime {
    pub fn new(stream: UnixStream, backing_file: File, apf_stream: UnixStream) -> Self {
        let file_meta = backing_file
            .metadata()
            .expect("can not get backing file metadata");
        let backing_memory_size = file_meta.len() as usize;
        // # Safety:
        // File size and fd are valid
        let ret = unsafe {
            libc::mmap(
                ptr::null_mut(),
                backing_memory_size,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                backing_file.as_raw_fd(),
                0,
            )
        };
        if ret == libc::MAP_FAILED {
            panic!("mmap on backing file failed");
        }

        let stream = Arc::new(Mutex::new(stream));
        let apf_stream = Arc::new(Mutex::new(apf_stream));

        let handler = Runtime::construct_handler(stream.clone(), ret.cast(), backing_memory_size);

        Self {
            stream,
            apf_stream,
            backing_file,
            backing_memory: ret.cast(),
            backing_memory_size,
            handler,
            message_buffer: [0; 4096],
            encode_buffer: bitcode::Buffer::new(),
            exitless_vcpus: HashMap::new(),
        }
    }

    /// Receive exitless APF fds from Firecracker.
    /// Firecracker sends: "exitless_apf:{num_vcpus}" with 3 fds per vCPU:
    /// [notify_eventfd, complete_eventfd, shared_page_memfd] × num_vcpus
    pub fn try_receive_exitless_apf(&mut self) -> std::io::Result<bool> {
        let mut msg_buf = [0u8; 256];
        let mut fds = [0i32; 64];

        {
            let stream = self.stream.lock().unwrap();
            stream.set_nonblocking(false).expect("set nonblocking");
        }

        let (_bytes_read, fds_read) = {
            let stream = self.stream.lock().unwrap();
            let mut iovecs = [libc::iovec {
                iov_base: msg_buf.as_mut_ptr().cast(),
                iov_len: msg_buf.len(),
            }];
            match unsafe { stream.recv_with_fds(&mut iovecs, &mut fds) } {
                Ok(r) => r,
                Err(e) => {
                    stream.set_nonblocking(true).expect("Set nonblocking");
                    let errno = e.errno();
                    return Err(std::io::Error::from_raw_os_error(errno));
                }
            }
        };

        {
            let stream = self.stream.lock().unwrap();
            stream.set_nonblocking(true).expect("set nonblocking");
        }

        if fds_read == 0 || fds_read % 3 != 0 {
            return Ok(false);
        }

        let num_vcpus = fds_read / 3;
        for i in 0..num_vcpus {
            let base = i * 3;
            let ctx = ExitlessApfVcpu::from_fds(
                fds[base],     // notify_eventfd
                fds[base + 1], // complete_eventfd
                fds[base + 2], // shared_page_memfd
            )?;
            let eventfd = ctx.eventfd;
            self.exitless_vcpus.insert(eventfd, ctx);
        }

        println!("Exitless APF: {num_vcpus} vCPUs configured");
        Ok(!self.exitless_vcpus.is_empty())
    }

    fn peer_process_credentials(&self) -> libc::ucred {
        let mut creds: libc::ucred = libc::ucred {
            pid: 0,
            gid: 0,
            uid: 0,
        };
        let mut creds_size = size_of::<libc::ucred>() as u32;
        let ret = unsafe {
            libc::getsockopt(
                self.stream.lock().unwrap().as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                (&raw mut creds).cast::<c_void>(),
                &raw mut creds_size,
            )
        };
        if ret != 0 {
            panic!("Failed to get peer process credentials");
        }
        creds
    }

    pub fn install_panic_hook(&self) {
        let peer_creds = self.peer_process_credentials();

        let default_panic_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic_info| {
            let r = unsafe { libc::kill(peer_creds.pid, libc::SIGKILL) };

            if r != 0 {
                eprintln!("Failed to kill Firecracker process from panic hook");
            }

            default_panic_hook(panic_info);
        }));
    }

    pub fn send_reply(&mut self, fault_reply: FaultReply, sender: &Sender<Vec<u8>>) {
        let encoded = self.encode_buffer.encode(&fault_reply);
        let size = encoded.len() as u32;
        self.message_buffer[..4].copy_from_slice(&size.to_le_bytes());
        self.message_buffer[4..4 + encoded.len()].copy_from_slice(encoded);

        let data = Vec::from(&self.message_buffer[..4 + (size as usize)]);
        sender.send_blocking(data).unwrap();
    }

    pub fn construct_handler(
        stream: Arc<Mutex<UnixStream>>,
        backing_memory: *mut u8,
        backing_memory_size: usize,
    ) -> UffdHandler {
        let mut message_buf = vec![0u8; 1024];
        let mut iovecs = [libc::iovec {
            iov_base: message_buf.as_mut_ptr().cast::<libc::c_void>(),
            iov_len: message_buf.len(),
        }];
        let mut fds = [0; 3];
        let (bytes_read, fds_read) = unsafe {
            stream
                .lock().unwrap()
                .recv_with_fds(&mut iovecs, &mut fds)
                .expect("recv_with_fds failed")
        };
        message_buf.resize(bytes_read, 0);

        let (guest_memfd, userfault_bitmap_memfd) = if fds_read == 3 {
            (
                Some(unsafe { File::from_raw_fd(fds[1]) }),
                Some(unsafe { File::from_raw_fd(fds[2]) }),
            )
        } else {
            (None, None)
        };

        UffdHandler::from_mappings(
            serde_json::from_slice(message_buf.as_slice()).unwrap(),
            unsafe { File::from_raw_fd(fds[0]) },
            guest_memfd,
            userfault_bitmap_memfd,
            backing_memory,
            backing_memory_size,
        )
    }

    /// Polls the `UnixStream` and UFFD fds in a loop.
    /// When stream is polled, new uffd is retrieved.
    /// When uffd is polled, page fault is handled by
    /// calling `pf_event_dispatch` with corresponding
    /// uffd object passed in.
    pub fn run(
        &mut self,
        pf_event_dispatch: impl Fn(&mut UffdHandler),
        pf_vcpu_event_dispatch: impl Fn(&mut UffdHandler, usize),
    ) {
        let mut pollfds = vec![];

        // Poll the stream for incoming uffds
        pollfds.push(libc::pollfd {
            fd: self.stream.lock().unwrap().as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        });

        // Poll the stream for incoming async page faults
        pollfds.push(libc::pollfd {
            fd: self.apf_stream.lock().unwrap().as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        });

        pollfds.push(libc::pollfd {
            fd: self.handler.uffd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        });

        let mut uffd_msg_iter =
            UffdMsgIterBitcode::new(self.stream.clone());

        let mut apf_msg_iter =
            UffdMsgIterBitcode::new(self.apf_stream.clone());

        let r_stream = self.stream.clone();
        let (r_send, r_recv) = unbounded::<Vec<u8>>();
        spawn(move || {
            smol::block_on(async {
                while let Ok(message) = r_recv.recv().await {
                    r_stream.lock().unwrap().write_all(&message).unwrap();
                }
            });
        });

        let a_stream = self.apf_stream.clone();
        let (a_send, a_recv) = unbounded::<Vec<u8>>();
        spawn(move || {
            smol::block_on(async {
                while let Ok(message) = a_recv.recv().await {
                    a_stream.lock().unwrap().write_all(&message).unwrap();
                }
            });
        });

        // Exitless APF completion channel — signals completion with artificial delay.
        let (apf_complete_send, apf_complete_recv) = unbounded::<ApfCompletion>();
        spawn(move || {
            smol::block_on(async {
                while let Ok(req) = apf_complete_recv.recv().await {
                    smol::spawn(async move {
                        smol::Timer::after(Duration::from_micros(100)).await;
                        signal_apf_ready(req.shared_page, req.complete_eventfd, req.gpa);
                    }).detach();
                }
            });
        });

        let stream_fd = self.stream.lock().unwrap().as_raw_fd();
        let apf_fd = self.apf_stream.lock().unwrap().as_raw_fd();

        // Try to receive exitless APF fds from Firecracker
        if let Ok(true) = self.try_receive_exitless_apf() {
            for ctx in self.exitless_vcpus.values() {
                pollfds.push(libc::pollfd {
                    fd: ctx.eventfd,
                    events: libc::POLLIN,
                    revents: 0,
                });
            }
        }

        loop {
            let pollfd_ptr = pollfds.as_mut_ptr();
            let pollfd_size = pollfds.len() as u64;

            // # Safety:
            // Pollfds vector is valid
            let mut nready = unsafe { libc::poll(pollfd_ptr, pollfd_size, -1) };

            if nready == -1 {
                panic!("Could not poll for events!")
            }

            for fd in &pollfds {
                if nready == 0 {
                    break;
                }
                if fd.revents & libc::POLLIN != 0 {
                    nready -= 1;
                    if fd.fd == stream_fd {
                        for fault_request in uffd_msg_iter.by_ref() {
                            let page_size = self.handler.page_size;

                            assert!(
                                (fault_request.offset as usize) < self.handler.size(),
                                "received bogus offset from firecracker"
                            );
                            pf_vcpu_event_dispatch(
                                &mut self.handler,
                                fault_request.offset as usize,
                            );

                            self.send_reply(fault_request.into_reply(page_size as u64), &r_send);
                        }
                    } else if fd.fd == apf_fd {
                        for mut fault_request in apf_msg_iter.by_ref() {
                            let page_size = self.handler.page_size;

                            if let Some(gpa) = fault_request.gpa {
                                fault_request.offset = self.handler.gpa_to_offset(gpa)
                                    .expect("Failed to convert GPA to offset");
                            }

                            assert!(
                                (fault_request.offset as usize) < self.handler.size(),
                                "received bogus offset from firecracker"
                            );

                            pf_vcpu_event_dispatch(
                                &mut self.handler,
                                fault_request.offset as usize,
                            );
                            self.send_reply(fault_request.into_reply(page_size as u64), &a_send);
                        }
                    } else if let Some(ctx) = self.exitless_vcpus.get_mut(&fd.fd) {
                        ctx.drain_eventfd();

                        let complete_eventfd = ctx.complete_eventfd;
                        let shared_page = ctx.shared_page;

                        while let Some(entry) = ctx.notify_ring().pop() {
                            let gpa = entry.gpa;
                            if let Some(offset) = self.handler.gpa_to_offset(gpa) {
                                pf_vcpu_event_dispatch(&mut self.handler, offset as usize);
                                signal_apf_ready(SendableSharedPage(shared_page), complete_eventfd, gpa);

                                // apf_complete_send
                                //     .send_blocking(ApfCompletion {
                                //         shared_page: SendableSharedPage(shared_page),
                                //         complete_eventfd,
                                //         gpa,
                                //     })
                                //     .unwrap();
                            }
                        }
                    } else {
                        pf_event_dispatch(&mut self.handler);
                    }
                }
            }
            pollfds.retain(|pollfd| pollfd.revents & (libc::POLLRDHUP | libc::POLLHUP) == 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;
    use std::os::unix::net::UnixListener;

    use vmm_sys_util::tempdir::TempDir;
    use vmm_sys_util::tempfile::TempFile;

    use super::*;

    unsafe impl Send for Runtime {}

    #[test]
    fn test_runtime() {
        let tmp_dir = TempDir::new().unwrap();
        let dummy_socket_path = tmp_dir.as_path().join("dummy_socket");
        let dummy_socket_path_clone = dummy_socket_path.clone();

        let mut uninit_runtime = Box::new(MaybeUninit::<Runtime>::uninit());
        // We will use this pointer to bypass a bunch of Rust Safety
        // for the sake of convenience.
        let runtime_ptr = uninit_runtime.as_ptr().cast::<Runtime>();

        let runtime_thread = std::thread::spawn(move || {
            let tmp_file = TempFile::new().unwrap();
            tmp_file.as_file().set_len(0x1000).unwrap();
            let dummy_mem_path = tmp_file.as_path();

            let file = File::open(dummy_mem_path).expect("Cannot open memfile");
            let listener =
                UnixListener::bind(dummy_socket_path).expect("Cannot bind to socket path");
            let (stream, _) = listener.accept().expect("Cannot listen on UDS socket");
            // Update runtime with actual runtime
            let runtime = uninit_runtime.write(Runtime::new(stream, file));
            runtime.run(|_: &mut UffdHandler| {}, |_: &mut UffdHandler, _: usize| {});
        });

        // wait for runtime thread to initialize itself
        std::thread::sleep(std::time::Duration::from_millis(100));

        let stream =
            UnixStream::connect(dummy_socket_path_clone).expect("Cannot connect to the socket");

        #[allow(deprecated)]
        let dummy_memory_region = vec![GuestRegionUffdMapping {
            base_host_virt_addr: 0,
            size: 0x1000,
            offset: 0,
            page_size: 4096,
        }];
        let dummy_memory_region_json = serde_json::to_string(&dummy_memory_region).unwrap();

        // Send the mapping message to the runtime.
        // We expect for the runtime to create a corresponding UffdHandler
        let dummy_file = TempFile::new().unwrap();
        let dummy_fd = dummy_file.as_file().as_raw_fd();
        stream
            .send_with_fd(dummy_memory_region_json.as_bytes(), dummy_fd)
            .unwrap();
        // wait for the runtime thread to process message
        std::thread::sleep(std::time::Duration::from_millis(100));
        unsafe {
            assert_eq!(
                (*runtime_ptr).handler.mem_regions.len(),
                dummy_memory_region.len()
            );
        }

        // there is no way to properly stop runtime, so
        // we send a message with an incorrect memory region
        // to cause runtime thread to panic
        #[allow(deprecated)]
        let error_memory_region = vec![GuestRegionUffdMapping {
            base_host_virt_addr: 0,
            size: 0,
            offset: 0,
            page_size: 4096,
        }];
        let error_memory_region_json = serde_json::to_string(&error_memory_region).unwrap();
        stream
            .send_with_fd(error_memory_region_json.as_bytes(), dummy_fd)
            .unwrap();

        runtime_thread.join().unwrap_err();
    }
}
