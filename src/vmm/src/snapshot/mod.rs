// Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provides serialization and deserialization facilities and implements a persistent storage
//! format for Firecracker state snapshots.
//!
//! Uses the [`FastSnapshot`] custom serializer which employs raw memcpy for `#[repr(C)]` KVM
//! structs and zero-overhead field-by-field encoding for everything else. Native-endian,
//! fixed-width integers, length-prefixed sequences.
//!
//! The snapshot format uses the following layout:
//!
//!  |-----------------------------|
//!  |       64 bit magic_id       |
//!  |-----------------------------|
//!  |       version string        |
//!  |-----------------------------|
//!  |            State            |
//!  |-----------------------------|
//!  |        optional CRC64       |
//!  |-----------------------------|
//!
//!
//! The snapshot format uses a version value in the form of `MAJOR.MINOR.PATCH`. The version is
//! provided by the library clients (it is not tied to this crate).
pub mod crc;
pub mod fast;
mod persist;
use std::fmt::Debug;
use std::io::{Read, Write};

use crc64::crc64;
use semver::Version;

use crate::persist::SNAPSHOT_VERSION;
use crate::snapshot::fast::FastSnapshot;
pub use crate::snapshot::persist::Persist;

#[cfg(target_arch = "x86_64")]
const SNAPSHOT_MAGIC_ID: u64 = 0x0710_1984_8664_0000u64;

#[cfg(target_arch = "aarch64")]
const SNAPSHOT_MAGIC_ID: u64 = 0x0710_1984_AAAA_0000u64;

/// Maximum size in bytes for snapshot deserialization to prevent DOS attacks.
/// Snapshots contain VM state which can be large, but we set a reasonable upper bound.
/// This limit is 10MB which should be sufficient for any legitimate snapshot.
const SNAPSHOT_DESERIALIZATION_BYTES_LIMIT: usize = 10_000_000;

/// Error definitions for the Snapshot API.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum SnapshotError {
    /// CRC64 validation failed
    Crc64,
    /// Invalid data version: {0}
    InvalidFormatVersion(Version),
    /// Magic value does not match arch: {0}
    InvalidMagic(u64),
    /// IO Error: {0}
    Io(#[from] std::io::Error),
    /// Snapshot size exceeds limit of {0} bytes
    SizeLimitExceeded(usize),
    /// Snapshot decode error: {0}
    Decode(#[from] fast::DecodeError),
    /// Snapshot contained {trailing} trailing bytes after the decoded state
    TrailingBytes {
        /// Number of unconsumed bytes between the end of the encoded state
        /// and the start of the CRC.
        trailing: usize,
    },
}

/// Firecracker snapshot header
#[derive(Debug)]
pub(crate) struct SnapshotHdr {
    /// magic value
    pub(crate) magic: u64,
    /// Snapshot data version
    pub(crate) version: Version,
}

/// Assumes the raw bytes stream read from the given [`Read`] instance is a snapshot file,
/// and returns the version of it.
pub fn get_format_version<R: Read>(reader: &mut R) -> Result<Version, SnapshotError> {
    let mut buf = Vec::new();
    let bytes_read = reader
        .take((SNAPSHOT_DESERIALIZATION_BYTES_LIMIT + 1) as u64)
        .read_to_end(&mut buf)?;

    if bytes_read > SNAPSHOT_DESERIALIZATION_BYTES_LIMIT {
        return Err(SnapshotError::SizeLimitExceeded(
            SNAPSHOT_DESERIALIZATION_BYTES_LIMIT,
        ));
    }

    if buf.len() < 8 {
        return Err(SnapshotError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "File too short to contain CRC",
        )));
    }

    let (data_buf, _crc_buf) = buf.split_at(buf.len() - 8);

    use crate::persist::MicrovmState;
    let mut offset = 0;
    match Snapshot::<MicrovmState>::decode(data_buf, &mut offset) {
        Ok(snapshot) => {
            if offset != data_buf.len() {
                return Err(SnapshotError::TrailingBytes {
                    trailing: data_buf.len() - offset,
                });
            }
            Ok(snapshot.header.version)
        }
        Err(e) => Err(SnapshotError::Decode(e)),
    }
}

/// Firecracker snapshot type
///
/// A type used to store and load Firecracker snapshots of a particular version
#[derive(Debug)]
pub struct Snapshot<Data> {
    pub(crate) header: SnapshotHdr,
    /// The data stored in this [`Snapshot`]
    pub data: Data,
}

impl<Data> Snapshot<Data> {
    /// Constructs a new snapshot with the given `data`.
    pub fn new(data: Data) -> Self {
        Self {
            header: SnapshotHdr {
                magic: SNAPSHOT_MAGIC_ID,
                version: SNAPSHOT_VERSION.clone(),
            },
            data,
        }
    }

    /// Gets the version of this snapshot
    pub fn version(&self) -> &Version {
        &self.header.version
    }
}

impl Snapshot<crate::persist::MicrovmState> {
    fn load_without_crc_check(buf: &[u8]) -> Result<Self, SnapshotError> {
        if buf.len() > SNAPSHOT_DESERIALIZATION_BYTES_LIMIT {
            return Err(SnapshotError::SizeLimitExceeded(
                SNAPSHOT_DESERIALIZATION_BYTES_LIMIT,
            ));
        }

        let mut offset = 0;
        let snapshot = Self::decode(buf, &mut offset)?;

        if offset != buf.len() {
            return Err(SnapshotError::TrailingBytes {
                trailing: buf.len() - offset,
            });
        }

        if snapshot.header.magic != SNAPSHOT_MAGIC_ID {
            return Err(SnapshotError::InvalidMagic(snapshot.header.magic));
        }

        if snapshot.header.version.major != SNAPSHOT_VERSION.major
            || snapshot.header.version.minor > SNAPSHOT_VERSION.minor
        {
            return Err(SnapshotError::InvalidFormatVersion(
                snapshot.header.version.clone(),
            ));
        }

        Ok(snapshot)
    }

    /// Loads a snapshot from the given [`Read`] instance, performing all validations
    /// (CRC, snapshot magic value, snapshot version).
    pub fn load<R: Read>(reader: &mut R) -> Result<Self, SnapshotError> {
        let mut buf = Vec::new();
        let bytes_read = reader
            .take((SNAPSHOT_DESERIALIZATION_BYTES_LIMIT + 1) as u64)
            .read_to_end(&mut buf)?;

        if bytes_read > SNAPSHOT_DESERIALIZATION_BYTES_LIMIT {
            return Err(SnapshotError::SizeLimitExceeded(
                SNAPSHOT_DESERIALIZATION_BYTES_LIMIT,
            ));
        }

        if buf.len() < 8 {
            return Err(SnapshotError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "File too short to contain CRC",
            )));
        }

        let (data_buf, _crc_buf) = buf.split_at(buf.len() - 8);
        let snapshot = Self::load_without_crc_check(data_buf)?;

        let computed_checksum = crc64(0, buf.as_slice());
        if computed_checksum != 0 {
            return Err(SnapshotError::Crc64);
        }
        Ok(snapshot)
    }

    /// Saves a `MicrovmState` snapshot by reference (avoids cloning).
    ///
    /// Encodes `header + data` into a single pre-sized buffer, writes it,
    /// then appends the CRC64 of the buffer. The buffer is also what
    /// [`load`](Self::load) validates the CRC against.
    ///
    /// Rejects the save with [`SnapshotError::SizeLimitExceeded`] if the
    /// encoded state is larger than [`SNAPSHOT_DESERIALIZATION_BYTES_LIMIT`],
    /// so that we never produce a snapshot we cannot read back.
    pub(crate) fn save_ref<W: Write>(
        header: &SnapshotHdr,
        data: &crate::persist::MicrovmState,
        writer: &mut W,
    ) -> Result<(), SnapshotError> {
        let size = header.encoded_size() + data.encoded_size();
        let mut buf = Vec::with_capacity(size);
        header.encode(&mut buf);
        data.encode(&mut buf);
        if buf.len() > SNAPSHOT_DESERIALIZATION_BYTES_LIMIT {
            return Err(SnapshotError::SizeLimitExceeded(
                SNAPSHOT_DESERIALIZATION_BYTES_LIMIT,
            ));
        }
        writer.write_all(&buf).map_err(SnapshotError::Io)?;
        let checksum = crc64(0, &buf);
        writer
            .write_all(&checksum.to_le_bytes())
            .map_err(SnapshotError::Io)
    }

    /// Saves this snapshot to the given writer, computing and appending CRC64.
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<(), SnapshotError> {
        Self::save_ref(&self.header, &self.data, writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist::MicrovmState;

    #[test]
    fn test_snapshot_restore() {
        let state = MicrovmState::default();
        let mut buf = Vec::new();

        Snapshot::save_ref(&Snapshot::new(()).header, &state, &mut buf).unwrap();
        Snapshot::<MicrovmState>::load(&mut buf.as_slice()).unwrap();
    }

    #[test]
    fn test_parse_version_from_file() {
        let state = MicrovmState::default();

        let mut snapshot_data = Vec::new();
        Snapshot::save_ref(&Snapshot::new(()).header, &state, &mut snapshot_data).unwrap();

        assert_eq!(
            get_format_version(&mut std::io::Cursor::new(&snapshot_data)).unwrap(),
            SNAPSHOT_VERSION
        );
    }

    #[test]
    fn test_bad_reader() {
        #[derive(Debug)]
        struct BadReader;

        impl Read for BadReader {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::InvalidInput.into())
            }
        }

        let mut reader = BadReader {};

        assert!(
            matches!(Snapshot::<MicrovmState>::load(&mut reader), Err(SnapshotError::Io(inner)) if inner.kind() == std::io::ErrorKind::InvalidInput)
        );
    }

    #[test]
    fn test_bad_magic() {
        let state = MicrovmState::default();
        let mut snapshot = Snapshot::new(&state);
        snapshot.header.magic = 0xDEADBEEF;

        let mut buf = Vec::new();
        // Encode with corrupted magic
        snapshot.header.encode(&mut buf);
        state.encode(&mut buf);

        assert!(matches!(
            Snapshot::<MicrovmState>::load_without_crc_check(&buf),
            Err(SnapshotError::InvalidMagic(_))
        ));
    }

    #[test]
    fn test_trailing_bytes_rejected() {
        let state = MicrovmState::default();
        let mut buf = Vec::new();
        let hdr = Snapshot::new(()).header;
        hdr.encode(&mut buf);
        state.encode(&mut buf);
        // Stuff a few extra bytes into the encoded state — they should be
        // rejected even though decode can parse the prefix.
        buf.extend_from_slice(&[0xDEu8, 0xAD, 0xBE, 0xEF]);
        assert!(matches!(
            Snapshot::<MicrovmState>::load_without_crc_check(&buf),
            Err(SnapshotError::TrailingBytes { trailing: 4 })
        ));
    }

    #[test]
    fn test_bad_crc() {
        let state = MicrovmState::default();

        let mut valid_data = Vec::new();
        Snapshot::save_ref(&Snapshot::new(()).header, &state, &mut valid_data).unwrap();

        // Corrupt the CRC
        if valid_data.len() >= 8 {
            for i in (valid_data.len() - 8)..valid_data.len() {
                valid_data[i] ^= 0xFF;
            }
        }

        assert!(matches!(
            Snapshot::<MicrovmState>::load(&mut std::io::Cursor::new(&valid_data)),
            Err(SnapshotError::Crc64)
        ));
    }

    #[test]
    fn test_bad_version() {
        let state = MicrovmState::default();

        // Different major version: shouldn't work
        let mut snapshot = Snapshot::new(&state);
        snapshot.header.version.major = SNAPSHOT_VERSION.major + 1;
        let mut buf = Vec::new();
        snapshot.header.encode(&mut buf);
        state.encode(&mut buf);
        assert!(matches!(
            Snapshot::<MicrovmState>::load_without_crc_check(&buf),
            Err(SnapshotError::InvalidFormatVersion(v)) if v.major == SNAPSHOT_VERSION.major + 1
        ));

        // minor > SNAPSHOT_VERSION.minor: shouldn't work
        let mut snapshot = Snapshot::new(&state);
        snapshot.header.version.minor = SNAPSHOT_VERSION.minor + 1;
        let mut buf = Vec::new();
        snapshot.header.encode(&mut buf);
        state.encode(&mut buf);
        assert!(matches!(
            Snapshot::<MicrovmState>::load_without_crc_check(&buf),
            Err(SnapshotError::InvalidFormatVersion(v)) if v.minor == SNAPSHOT_VERSION.minor + 1
        ));

        // Same version should work
        let snapshot = Snapshot::new(&state);
        let mut buf = Vec::new();
        snapshot.header.encode(&mut buf);
        state.encode(&mut buf);
        Snapshot::<MicrovmState>::load_without_crc_check(&buf).unwrap();

        // Smaller minor version should work
        if SNAPSHOT_VERSION.minor != 0 {
            let mut snapshot = Snapshot::new(&state);
            snapshot.header.version.minor = SNAPSHOT_VERSION.minor - 1;
            let mut buf = Vec::new();
            snapshot.header.encode(&mut buf);
            state.encode(&mut buf);
            Snapshot::<MicrovmState>::load_without_crc_check(&buf).unwrap();
        }

        // Any patch version should work
        let mut snapshot = Snapshot::new(&state);
        snapshot.header.version.patch = 1024;
        let mut buf = Vec::new();
        snapshot.header.encode(&mut buf);
        state.encode(&mut buf);
        Snapshot::<MicrovmState>::load_without_crc_check(&buf).unwrap();
    }
}
