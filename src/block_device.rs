/*
 * Copyright (C) 2026 Arael David Espinosa Pérez
 *
 * This program is free software: you can redistribute and/or modify it
 * under the terms of the GNU General Public License published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 */

//! Abstract block device interface for BAFS.
//!
//! This module defines the `BlockDevice` trait that all BAFS I/O goes through.
//! The trait mirrors the `BlockDevice` trait defined in the Bazzulto kernel's
//! `kernel/src/hal/disk.rs` so that kernel-compiled code can pass a reference
//! to a real disk driver without any conversion.
//!
//! # Why a separate trait instead of depending on the kernel directly?
//!
//! BAFS is a standalone crate.  It must compile in three contexts:
//!
//! 1. **Kernel** (`feature = "kernel"`): linked into the Bazzulto kernel image.
//!    Here `BlockDevice` objects are real virtio-blk or NVMe drivers.
//! 2. **Userspace tools** (`feature = "userspace"`): `bafs-mkfs`, `bafs-fsck`.
//!    Here `BlockDevice` implementations wrap a regular file or device node.
//! 3. **Integration tests** (no feature flag, `cargo test`): the test suite
//!    provides a `MemoryDisk` backed by a `Vec<u8>`.
//!
//! Defining the trait inside the crate avoids a circular dependency on the
//! kernel crate and keeps the crate self-contained.
//!
//! # Impact on the rest of the system
//!
//! Every module that performs disk I/O (`superblock`, `btree`, `journal`,
//! `volume`, etc.) receives a `&dyn BlockDevice` reference.  The `kernel`
//! module provides a zero-overhead adapter so that the kernel's own
//! `BlockDevice` objects satisfy this trait without copying any data.

/// A raw block storage device that can be read and written in 512-byte sectors.
///
/// This trait is the only I/O abstraction used by BAFS.  All higher-level
/// operations (B-tree traversal, extent allocation, journal replay) ultimately
/// call `read_sectors` and `write_sectors` through this interface.
///
/// # Sector size
///
/// The Bazzulto kernel currently uses 512-byte sectors exclusively.  BAFS
/// groups sectors into 4 KiB blocks (8 sectors per block) for all metadata
/// and data I/O.
///
/// # Thread safety
///
/// The trait requires `Send + Sync` so that a `Arc<dyn BlockDevice>` can be
/// shared across kernel threads once SMP is introduced.
pub trait BlockDevice: Send + Sync {
    /// Read `sector_count` consecutive 512-byte sectors starting at logical
    /// block address `start_lba` into `destination_buffer`.
    ///
    /// `destination_buffer` must be at least `sector_count * 512` bytes long.
    ///
    /// Returns `true` on success, `false` if the device reported an error.
    fn read_sectors(
        &self,
        start_lba: u64,
        sector_count: u32,
        destination_buffer: &mut [u8],
    ) -> bool;

    /// Write `sector_count` consecutive 512-byte sectors from `source_buffer`
    /// to the device starting at logical block address `start_lba`.
    ///
    /// `source_buffer` must be at least `sector_count * 512` bytes long.
    ///
    /// Returns `true` on success, `false` if the device reported an error.
    fn write_sectors(
        &self,
        start_lba: u64,
        sector_count: u32,
        source_buffer: &[u8],
    ) -> bool;

    /// Total capacity of the device in 512-byte sectors.
    fn total_sector_count(&self) -> u64;

    /// Sector size in bytes.  Always 512 in the current Bazzulto kernel.
    fn sector_size_in_bytes(&self) -> u32 {
        512
    }

    /// Human-readable device name for log messages and error reports.
    fn device_name(&self) -> &str;
}
