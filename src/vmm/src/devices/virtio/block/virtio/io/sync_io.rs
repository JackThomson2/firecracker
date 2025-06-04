// Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::fs::File;

use vm_memory::mmap::MmapRegionBuilder;
use vm_memory::{FileOffset, GuestMemoryError, MmapRegion, VolatileMemory, VolatileMemoryError};

use crate::vstate::memory::{GuestAddress, GuestMemory, GuestMemoryMmap};

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum SyncIoError {
    /// Flush: {0}
    Flush(std::io::Error),
    /// Seek: {0}
    Seek(std::io::Error),
    /// SyncAll: {0}
    SyncAll(std::io::Error),
    /// Transfer: {0}
    Transfer(#[from] GuestMemoryError),
}

impl From<VolatileMemoryError> for SyncIoError {
    fn from(value: VolatileMemoryError) -> Self {
        SyncIoError::Transfer(GuestMemoryError::from(value))
    }
}

#[derive(Debug)]
pub struct SyncFileEngine {
    mapping: MmapRegion,
}

// SAFETY: `File` is send and ultimately a POD.
unsafe impl Send for SyncFileEngine {}

impl SyncFileEngine {
    pub fn from_file(file: File, is_disk_read_only: bool) -> SyncFileEngine {
        let prot = libc::PROT_READ
            | if is_disk_read_only {
                0
            } else {
                libc::PROT_WRITE
            };

        let file_size = file.metadata().unwrap().len();
        let mapping = MmapRegionBuilder::new(file_size as usize)
            .with_file_offset(FileOffset::new(file, 0))
            .with_mmap_prot(prot)
            .with_mmap_flags(libc::MAP_SHARED)
            .build()
            .unwrap();

        SyncFileEngine { mapping }
    }

    #[cfg(test)]
    pub fn file(&self) -> &File {
        self.mapping.file_offset().unwrap().file()
    }

    pub fn start_bouncing(&mut self) {}

    pub fn is_bouncing(&self) -> bool {
        false
    }

    /// Update the backing file of the engine
    pub fn update_file(&mut self, file: File) {
        *self = SyncFileEngine::from_file(file, self.mapping.prot() == libc::PROT_READ);
    }

    pub fn read(
        &mut self,
        offset: u64,
        mem: &GuestMemoryMmap,
        addr: GuestAddress,
        count: u32,
    ) -> Result<u32, SyncIoError> {
        let target = mem.get_slice(addr, count as usize)?;
        let source = self.mapping.get_slice(offset as usize, count as usize)?;
        source.copy_to_volatile_slice(target);

        Ok(count)
    }

    pub fn write(
        &mut self,
        offset: u64,
        mem: &GuestMemoryMmap,
        addr: GuestAddress,
        count: u32,
    ) -> Result<u32, SyncIoError> {
        let target = self.mapping.get_slice(offset as usize, count as usize)?;
        let source = mem.get_slice(addr, count as usize)?;
        source.copy_to_volatile_slice(target);

        Ok(count)
    }

    pub fn flush(&mut self) -> Result<(), SyncIoError> {
        // SAFETY: No invariants need to be upheld - the kernel validates that ptr + addr are
        // valid.
        let r = unsafe {
            libc::msync(
                self.mapping.as_ptr().cast(),
                self.mapping.len(),
                libc::MS_SYNC,
            )
        };

        if r < 0 {
            return Err(SyncIoError::Flush(std::io::Error::last_os_error()));
        }

        Ok(())
    }
}
