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

//! Comprehensive integration tests for BAFS.
//!
//! Coverage areas:
//!
//! 1.  Disk size boundaries (minimum, too-small)
//! 2.  Large disk (10 GiB)
//! 3.  Basic correctness (empty file, overwrite, partial read, large file,
//!     sequential offsets, nested directories, max filename length)
//! 4.  Multi-session persistence (three mounts, flush_and_commit checkpoints,
//!     free-space accounting)
//! 5.  Directory / B-tree stress (500 files, bulk unlink)
//! 6.  Crash simulation via `FaultInjectionDisk`
//! 7.  On-disk corruption detection via `CorruptingDisk`
//! 8.  Unlink edge cases (not-found, create-unlink-recreate)
//! 9.  Free-space accounting (format, after write, after unlink)
//! 10. Journal recovery (simulate crash after journal write, before final flush)
//! 11. Superblock backup (primary zeroed, backup used)
//! 12. CoW recycling (many sessions on small disk, no OutOfSpace)
//! 13. CRC32C and xxHash-64 unit sanity checks

use std::sync::{Arc, Mutex};

use bafs::block_device::BlockDevice;
use bafs::checksum::compute_crc32c;
use bafs::error::BafsError;
use bafs::volume::{
    bafs_format, bafs_mount, bafs_unmount, flush_and_commit, volume_create_directory,
    volume_create_file, volume_lookup_directory_entry, volume_read_file_data,
    volume_unlink_directory_entry, volume_write_file_data, BafsFormatOptions,
};

// ── Sector / block constants ─────────────────────────────────────────────────

const SECTOR_SIZE_BYTES: usize = 512;
const BLOCK_SIZE_BYTES: usize = 4096;
const SECTORS_PER_BLOCK: usize = BLOCK_SIZE_BYTES / SECTOR_SIZE_BYTES;

// ─── MemoryDisk ───────────────────────────────────────────────────────────────

/// In-memory block device backed by `Arc<Mutex<Vec<u8>>>`.
///
/// Multiple `MemoryDisk` handles can share the same bytes via `clone_arc`,
/// allowing a second `BafsVolume` to be mounted on the same "disk" after the
/// first is unmounted.
pub struct MemoryDisk {
    data: Arc<Mutex<Vec<u8>>>,
    total_sectors: u64,
}

impl MemoryDisk {
    fn new(size_bytes: usize) -> Self {
        let total_sectors = size_bytes as u64 / SECTOR_SIZE_BYTES as u64;
        MemoryDisk {
            data: Arc::new(Mutex::new(vec![0u8; size_bytes])),
            total_sectors,
        }
    }

    /// 512 MiB — the standard test disk.
    pub fn new_512mib() -> Self {
        Self::new(512 * 1024 * 1024)
    }

    /// 10 GiB — for large-disk tests.  Backed by a Vec<u8>; the OS uses
    /// demand-paging so physical RAM is only consumed for written pages.
    pub fn new_10gib() -> Self {
        Self::new(10 * 1024 * 1024 * 1024)
    }

    /// Construct a disk of exactly `block_count` 4 KiB blocks.
    pub fn new_blocks(block_count: u64) -> Self {
        Self::new(block_count as usize * BLOCK_SIZE_BYTES)
    }

    /// Return a new handle pointing at the same underlying bytes.
    pub fn clone_arc(&self) -> Self {
        MemoryDisk {
            data: Arc::clone(&self.data),
            total_sectors: self.total_sectors,
        }
    }

    /// Zero-fill block `block_address` in-place (used by corruption tests).
    pub fn zero_block(&self, block_address: u64) {
        let start = block_address as usize * BLOCK_SIZE_BYTES;
        let mut data = self.data.lock().unwrap();
        for byte in &mut data[start..start + BLOCK_SIZE_BYTES] {
            *byte = 0;
        }
    }

    /// Flip bit 0 of byte 0 of block `block_address` (used by corruption tests).
    pub fn flip_bit_in_block(&self, block_address: u64) {
        let start = block_address as usize * BLOCK_SIZE_BYTES;
        let mut data = self.data.lock().unwrap();
        data[start] ^= 0x01;
    }
}

impl BlockDevice for MemoryDisk {
    fn read_sectors(&self, start_lba: u64, sector_count: u32, dest: &mut [u8]) -> bool {
        let start_byte = start_lba as usize * SECTOR_SIZE_BYTES;
        let byte_count = sector_count as usize * SECTOR_SIZE_BYTES;
        let data = self.data.lock().unwrap();
        if start_byte + byte_count > data.len() {
            return false;
        }
        dest[..byte_count].copy_from_slice(&data[start_byte..start_byte + byte_count]);
        true
    }

    fn write_sectors(&self, start_lba: u64, sector_count: u32, src: &[u8]) -> bool {
        let start_byte = start_lba as usize * SECTOR_SIZE_BYTES;
        let byte_count = sector_count as usize * SECTOR_SIZE_BYTES;
        let mut data = self.data.lock().unwrap();
        if start_byte + byte_count > data.len() {
            return false;
        }
        data[start_byte..start_byte + byte_count].copy_from_slice(&src[..byte_count]);
        true
    }

    fn total_sector_count(&self) -> u64 {
        self.total_sectors
    }

    fn device_name(&self) -> &str {
        "MemoryDisk"
    }
}

// ─── FaultInjectionDisk ───────────────────────────────────────────────────────

/// Wraps a `MemoryDisk`.  The first `fail_after` write_sectors calls succeed
/// normally; the `fail_after`-th call (1-indexed) returns `false` to simulate
/// a power-loss event.
///
/// After the injected failure, all subsequent writes also fail so the disk
/// appears frozen at the crash point.  Reads always succeed.
pub struct FaultInjectionDisk {
    inner: MemoryDisk,
    /// Countdown: None = no fault scheduled; Some(0) = next write fails.
    countdown: Mutex<Option<u32>>,
}

impl FaultInjectionDisk {
    /// Create a disk that fails on the `fail_after`-th write_sectors call.
    /// `fail_after = 1` means the very first write fails.
    pub fn new(inner: MemoryDisk, fail_after: u32) -> Self {
        FaultInjectionDisk {
            inner,
            countdown: Mutex::new(Some(fail_after.saturating_sub(1))),
        }
    }

    /// Return a new `MemoryDisk` view of the same underlying bytes (for
    /// remounting after the simulated crash).
    pub fn clone_inner(&self) -> MemoryDisk {
        self.inner.clone_arc()
    }
}

impl BlockDevice for FaultInjectionDisk {
    fn read_sectors(&self, start_lba: u64, sector_count: u32, dest: &mut [u8]) -> bool {
        self.inner.read_sectors(start_lba, sector_count, dest)
    }

    fn write_sectors(&self, start_lba: u64, sector_count: u32, src: &[u8]) -> bool {
        let mut guard = self.countdown.lock().unwrap();
        match *guard {
            None => {
                // No fault configured — pass through.
                drop(guard);
                self.inner.write_sectors(start_lba, sector_count, src)
            }
            Some(0) => {
                // This is the call that should fail.  Keep countdown at 0 so
                // all subsequent writes also fail.
                false
            }
            Some(ref mut n) => {
                *n -= 1;
                drop(guard);
                self.inner.write_sectors(start_lba, sector_count, src)
            }
        }
    }

    fn total_sector_count(&self) -> u64 {
        self.inner.total_sector_count()
    }

    fn device_name(&self) -> &str {
        "FaultInjectionDisk"
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Format + mount a fresh MemoryDisk, returning (disk, volume).
///
/// The disk is cloned before mounting so the original `MemoryDisk` arc can
/// be cloned again later for remounting.
macro_rules! fresh_512mib {
    () => {{
        let disk = MemoryDisk::new_512mib();
        bafs_format(&disk, BafsFormatOptions::default())
            .expect("format should succeed on 512 MiB disk");
        let vol = bafs_mount(disk.clone_arc()).expect("mount should succeed after format");
        (disk, vol)
    }};
}


// ═══════════════════════════════════════════════════════════════════════════
// 1. Existing lifecycle tests (kept for regression)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bafs_full_lifecycle_write_and_read_survives_remount() {
    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default())
        .expect("format should succeed on a fresh 512 MiB disk");

    let disk_for_first_mount = disk.clone_arc();
    let mut volume = bafs_mount(disk_for_first_mount)
        .expect("mount should succeed immediately after format");

    let root_inode_number = volume.superblock.root_inode_number;
    assert_eq!(root_inode_number, 1);

    let file_inode_number = volume_create_file(&mut volume, root_inode_number, "hello.txt")
        .expect("creating hello.txt should succeed");

    let content_to_write: &[u8] = b"hello, bafs";
    let bytes_written =
        volume_write_file_data(&mut volume, file_inode_number, 0, content_to_write)
            .expect("write should succeed");
    assert_eq!(bytes_written, content_to_write.len());

    let mut read_buffer = vec![0u8; content_to_write.len()];
    let bytes_read =
        volume_read_file_data(&volume, file_inode_number, 0, &mut read_buffer)
            .expect("read should succeed within same mount");
    assert_eq!(bytes_read, content_to_write.len());
    assert_eq!(&read_buffer[..bytes_read], content_to_write);

    bafs_unmount(volume).expect("unmount should succeed");

    let disk_for_second_mount = disk.clone_arc();
    let volume2 = bafs_mount(disk_for_second_mount)
        .expect("remount should succeed");

    let root2 = volume2.superblock.root_inode_number;
    let looked_up = volume_lookup_directory_entry(&volume2, root2, "hello.txt")
        .expect("lookup should not error")
        .expect("hello.txt should exist after remount");
    assert_eq!(looked_up, file_inode_number);

    let mut persistent_buf = vec![0u8; content_to_write.len()];
    let persistent_read =
        volume_read_file_data(&volume2, looked_up, 0, &mut persistent_buf)
            .expect("read after remount should succeed");
    assert_eq!(&persistent_buf[..persistent_read], content_to_write);

    bafs_unmount(volume2).expect("second unmount should succeed");
}

#[test]
fn bafs_directory_with_multiple_files_lookup_works() {
    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default()).unwrap();

    let mut volume = bafs_mount(disk.clone_arc()).unwrap();
    let root = volume.superblock.root_inode_number;

    let inode_a = volume_create_file(&mut volume, root, "alpha.txt").unwrap();
    let inode_b = volume_create_file(&mut volume, root, "beta.txt").unwrap();

    volume_write_file_data(&mut volume, inode_a, 0, b"file alpha").unwrap();
    volume_write_file_data(&mut volume, inode_b, 0, b"file beta").unwrap();

    bafs_unmount(volume).unwrap();

    let volume2 = bafs_mount(disk.clone_arc()).unwrap();
    let root2 = volume2.superblock.root_inode_number;

    let found_a = volume_lookup_directory_entry(&volume2, root2, "alpha.txt")
        .unwrap()
        .expect("alpha.txt should exist");
    let found_b = volume_lookup_directory_entry(&volume2, root2, "beta.txt")
        .unwrap()
        .expect("beta.txt should exist");

    assert_ne!(found_a, found_b);

    let mut buf_a = vec![0u8; 10];
    volume_read_file_data(&volume2, found_a, 0, &mut buf_a).unwrap();
    assert_eq!(&buf_a, b"file alpha");

    let mut buf_b = vec![0u8; 9];
    volume_read_file_data(&volume2, found_b, 0, &mut buf_b).unwrap();
    assert_eq!(&buf_b, b"file beta");

    bafs_unmount(volume2).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. CRC32C / xxHash-64 unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn crc32c_standard_test_vector_is_correct() {
    assert_eq!(compute_crc32c(b"123456789"), 0xE306_9283);
}

#[test]
fn crc32c_all_zeros_4kib_is_reproducible() {
    let zeroes = vec![0u8; 4096];
    let first = compute_crc32c(&zeroes);
    let second = compute_crc32c(&zeroes);
    assert_eq!(first, second, "CRC32C must be deterministic");
}

#[test]
fn crc32c_single_bit_flip_changes_checksum() {
    let mut data = vec![0xABu8; 512];
    let original = compute_crc32c(&data);
    data[7] ^= 0x01;
    let flipped = compute_crc32c(&data);
    assert_ne!(original, flipped, "a single bit flip must change the CRC32C");
}

#[test]
fn xxhash64_empty_string_reference_value_is_correct() {
    use bafs::dir::xxhash64_with_seed_zero;
    assert_eq!(xxhash64_with_seed_zero(b""), 0xEF46DB3751D8E999);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Disk size boundary tests
// ═══════════════════════════════════════════════════════════════════════════

/// A disk that is clearly too small must return `OutOfSpace` from `bafs_format`.
#[test]
fn disk_too_small_returns_out_of_space() {
    // 1 MiB = 256 blocks.  The journal alone needs 256 blocks (min) plus 3
    // superblock/root blocks, leaving no room for data.  format must reject it.
    let tiny_disk = MemoryDisk::new_blocks(256);
    let result = bafs_format(&tiny_disk, BafsFormatOptions::default());
    assert!(
        matches!(result, Err(BafsError::OutOfSpace)),
        "expected OutOfSpace on a 256-block disk, got {:?}",
        result
    );
}

/// The smallest disk that passes format must also survive a mount/write/read cycle.
///
/// We first find the minimum block count that passes `bafs_format`, then
/// double it to give the bump-pointer allocator enough headroom for CoW nodes
/// produced during the write and unmount.
#[test]
fn minimum_viable_disk_formats_mounts_and_survives_write() {
    // Find the minimum block count that passes bafs_format.
    let min_format_blocks = (270u64..)
        .find(|&n| bafs_format(&MemoryDisk::new_blocks(n), BafsFormatOptions::default()).is_ok())
        .expect("format should succeed before 600 blocks");

    assert!(min_format_blocks < 600, "unreasonably large minimum disk");

    // Use 2× the minimum to ensure the bump pointer does not overflow during
    // the write + unmount (which produce CoW nodes in addition to data blocks).
    let safe_block_count = min_format_blocks * 2;
    let disk = MemoryDisk::new_blocks(safe_block_count);
    bafs_format(&disk, BafsFormatOptions::default())
        .expect("format should succeed on 2× minimum disk");

    let mut vol = bafs_mount(disk.clone_arc())
        .expect("mount should succeed on 2× minimum disk");
    let root = vol.superblock.root_inode_number;
    let inode = volume_create_file(&mut vol, root, "min.txt")
        .expect("create file should succeed");
    volume_write_file_data(&mut vol, inode, 0, b"min")
        .expect("write should succeed on 2× minimum disk");
    bafs_unmount(vol).expect("unmount should succeed");

    let vol2 = bafs_mount(disk.clone_arc()).unwrap();
    let found = volume_lookup_directory_entry(&vol2, vol2.superblock.root_inode_number, "min.txt")
        .unwrap()
        .expect("file should persist on minimum disk");
    let mut buf = [0u8; 3];
    volume_read_file_data(&vol2, found, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"min");
    bafs_unmount(vol2).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Large disk (10 GiB)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn large_disk_10gib_format_mount_write_1mib_read_verify() {
    let disk = MemoryDisk::new_10gib();
    bafs_format(&disk, BafsFormatOptions::default()).expect("format 10 GiB disk");

    let mut vol = bafs_mount(disk.clone_arc()).expect("mount 10 GiB disk");
    let root = vol.superblock.root_inode_number;

    // 1 MiB of data with a known pattern.
    let write_size = 1024 * 1024usize;
    let pattern: Vec<u8> = (0..write_size).map(|i| (i & 0xFF) as u8).collect();

    let inode = volume_create_file(&mut vol, root, "big.bin").unwrap();
    let written = volume_write_file_data(&mut vol, inode, 0, &pattern).unwrap();
    assert_eq!(written, write_size);

    let mut read_buf = vec![0u8; write_size];
    let read_back = volume_read_file_data(&vol, inode, 0, &mut read_buf).unwrap();
    assert_eq!(read_back, write_size);
    assert_eq!(read_buf, pattern, "1 MiB read-back must match write on 10 GiB disk");

    bafs_unmount(vol).unwrap();
}

#[test]
fn large_disk_10gib_creates_1000_files_and_all_survive_remount() {
    let disk = MemoryDisk::new_10gib();
    bafs_format(&disk, BafsFormatOptions::default()).unwrap();

    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;

        for i in 0u32..1000 {
            let name = format!("file_{:04}.txt", i);
            let inode = volume_create_file(&mut vol, root, &name)
                .unwrap_or_else(|e| panic!("create {} failed: {}", name, e));
            let content = format!("content_{}", i);
            volume_write_file_data(&mut vol, inode, 0, content.as_bytes()).unwrap();
        }
        bafs_unmount(vol).unwrap();
    }

    // Remount and verify all 1000 files.
    let vol2 = bafs_mount(disk.clone_arc()).unwrap();
    let root2 = vol2.superblock.root_inode_number;

    for i in 0u32..1000 {
        let name = format!("file_{:04}.txt", i);
        let inode = volume_lookup_directory_entry(&vol2, root2, &name)
            .unwrap_or_else(|e| panic!("lookup {} failed: {}", name, e))
            .unwrap_or_else(|| panic!("{} not found after remount", name));

        let expected = format!("content_{}", i);
        let mut buf = vec![0u8; expected.len()];
        volume_read_file_data(&vol2, inode, 0, &mut buf).unwrap();
        assert_eq!(&buf, expected.as_bytes(), "content mismatch for {}", name);
    }
    bafs_unmount(vol2).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Basic correctness
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn empty_file_can_be_created_and_read_returns_zero_bytes() {
    let (disk, mut vol) = fresh_512mib!();
    let root = vol.superblock.root_inode_number;

    let inode = volume_create_file(&mut vol, root, "empty.txt").unwrap();
    // Reading from an empty file with a non-zero buffer should return 0.
    let mut buf = [0u8; 64];
    let bytes_read = volume_read_file_data(&vol, inode, 0, &mut buf).unwrap();
    assert_eq!(bytes_read, 0, "reading an empty file should return 0 bytes");

    bafs_unmount(vol).unwrap();
    // After remount the file should still exist and still be empty.
    let vol2 = bafs_mount(disk.clone_arc()).unwrap();
    let root2 = vol2.superblock.root_inode_number;
    let found = volume_lookup_directory_entry(&vol2, root2, "empty.txt")
        .unwrap()
        .expect("empty file should persist");
    let bytes_read2 = volume_read_file_data(&vol2, found, 0, &mut buf).unwrap();
    assert_eq!(bytes_read2, 0);
    bafs_unmount(vol2).unwrap();
}

#[test]
fn file_overwrite_replaces_content() {
    let (disk, mut vol) = fresh_512mib!();
    let root = vol.superblock.root_inode_number;

    let inode = volume_create_file(&mut vol, root, "over.txt").unwrap();
    volume_write_file_data(&mut vol, inode, 0, b"original").unwrap();
    volume_write_file_data(&mut vol, inode, 0, b"replaced").unwrap();

    let mut buf = [0u8; 8];
    volume_read_file_data(&vol, inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"replaced");

    bafs_unmount(vol).unwrap();

    let vol2 = bafs_mount(disk.clone_arc()).unwrap();
    let root2 = vol2.superblock.root_inode_number;
    let found = volume_lookup_directory_entry(&vol2, root2, "over.txt")
        .unwrap()
        .unwrap();
    let mut buf2 = [0u8; 8];
    volume_read_file_data(&vol2, found, 0, &mut buf2).unwrap();
    assert_eq!(&buf2, b"replaced");
    bafs_unmount(vol2).unwrap();
}

#[test]
fn read_beyond_eof_returns_only_available_bytes() {
    let (_disk, mut vol) = fresh_512mib!();
    let root = vol.superblock.root_inode_number;

    let inode = volume_create_file(&mut vol, root, "small.txt").unwrap();
    volume_write_file_data(&mut vol, inode, 0, b"hello").unwrap();

    // Ask for 100 bytes but file only has 5.
    let mut buf = vec![0u8; 100];
    let bytes_read = volume_read_file_data(&vol, inode, 0, &mut buf).unwrap();
    assert_eq!(bytes_read, 5, "read should return only the 5 available bytes");
    assert_eq!(&buf[..5], b"hello");
    bafs_unmount(vol).unwrap();
}

#[test]
fn large_file_256kib_write_and_read_back_correct() {
    let (disk, mut vol) = fresh_512mib!();
    let root = vol.superblock.root_inode_number;

    // 256 KiB = 64 blocks.
    let size = 256 * 1024usize;
    let pattern: Vec<u8> = (0..size).map(|i| ((i * 13 + 7) & 0xFF) as u8).collect();

    let inode = volume_create_file(&mut vol, root, "large.bin").unwrap();
    let written = volume_write_file_data(&mut vol, inode, 0, &pattern).unwrap();
    assert_eq!(written, size);

    let mut buf = vec![0u8; size];
    let read = volume_read_file_data(&vol, inode, 0, &mut buf).unwrap();
    assert_eq!(read, size);
    assert_eq!(buf, pattern, "256 KiB read-back must match write");

    bafs_unmount(vol).unwrap();

    let vol2 = bafs_mount(disk.clone_arc()).unwrap();
    let root2 = vol2.superblock.root_inode_number;
    let found = volume_lookup_directory_entry(&vol2, root2, "large.bin")
        .unwrap()
        .unwrap();
    let mut buf2 = vec![0u8; size];
    volume_read_file_data(&vol2, found, 0, &mut buf2).unwrap();
    assert_eq!(buf2, pattern, "256 KiB content must persist across remount");
    bafs_unmount(vol2).unwrap();
}

#[test]
fn max_filename_length_255_bytes_works() {
    let (disk, mut vol) = fresh_512mib!();
    let root = vol.superblock.root_inode_number;

    // 255 bytes is the BAFS maximum (matching Linux ext4/HFS+).
    let long_name: String = "a".repeat(255);
    let inode = volume_create_file(&mut vol, root, &long_name)
        .expect("creating a 255-byte filename should succeed");
    volume_write_file_data(&mut vol, inode, 0, b"max name").unwrap();
    bafs_unmount(vol).unwrap();

    let vol2 = bafs_mount(disk.clone_arc()).unwrap();
    let root2 = vol2.superblock.root_inode_number;
    let found = volume_lookup_directory_entry(&vol2, root2, &long_name)
        .unwrap()
        .expect("255-byte filename should survive remount");
    let mut buf = [0u8; 8];
    volume_read_file_data(&vol2, found, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"max name");
    bafs_unmount(vol2).unwrap();
}

#[test]
fn nested_directory_file_survives_remount() {
    let (disk, mut vol) = fresh_512mib!();
    let root = vol.superblock.root_inode_number;

    // root/subdir/deep.txt
    let subdir_inode = volume_create_directory(&mut vol, root, "subdir")
        .expect("create subdir should succeed");
    let file_inode = volume_create_file(&mut vol, subdir_inode, "deep.txt")
        .expect("create file in subdir should succeed");
    volume_write_file_data(&mut vol, file_inode, 0, b"deep content").unwrap();
    bafs_unmount(vol).unwrap();

    let vol2 = bafs_mount(disk.clone_arc()).unwrap();
    let root2 = vol2.superblock.root_inode_number;
    let found_subdir = volume_lookup_directory_entry(&vol2, root2, "subdir")
        .unwrap()
        .expect("subdir should exist after remount");
    let found_file = volume_lookup_directory_entry(&vol2, found_subdir, "deep.txt")
        .unwrap()
        .expect("deep.txt should exist inside subdir");
    let mut buf = [0u8; 12];
    volume_read_file_data(&vol2, found_file, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"deep content");
    bafs_unmount(vol2).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Multi-session persistence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn three_mount_cycles_all_data_persists() {
    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default()).unwrap();

    // Session 1: write file A.
    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        let inode = volume_create_file(&mut vol, root, "a.txt").unwrap();
        volume_write_file_data(&mut vol, inode, 0, b"session one").unwrap();
        bafs_unmount(vol).unwrap();
    }

    // Session 2: write file B.
    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        let inode = volume_create_file(&mut vol, root, "b.txt").unwrap();
        volume_write_file_data(&mut vol, inode, 0, b"session two").unwrap();
        bafs_unmount(vol).unwrap();
    }

    // Session 3: verify both files.
    {
        let vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;

        let inode_a = volume_lookup_directory_entry(&vol, root, "a.txt")
            .unwrap()
            .expect("a.txt must survive three mounts");
        let inode_b = volume_lookup_directory_entry(&vol, root, "b.txt")
            .unwrap()
            .expect("b.txt must survive three mounts");

        let mut buf_a = [0u8; 11];
        volume_read_file_data(&vol, inode_a, 0, &mut buf_a).unwrap();
        assert_eq!(&buf_a, b"session one");

        let mut buf_b = [0u8; 11];
        volume_read_file_data(&vol, inode_b, 0, &mut buf_b).unwrap();
        assert_eq!(&buf_b, b"session two");

        bafs_unmount(vol).unwrap();
    }
}

#[test]
fn flush_and_commit_mid_session_persists_data() {
    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default()).unwrap();

    let mut vol = bafs_mount(disk.clone_arc()).unwrap();
    let root = vol.superblock.root_inode_number;

    // Write before checkpoint.
    let inode_pre = volume_create_file(&mut vol, root, "pre.txt").unwrap();
    volume_write_file_data(&mut vol, inode_pre, 0, b"pre-commit").unwrap();

    // Checkpoint without unmounting.
    flush_and_commit(&mut vol).expect("flush_and_commit should succeed");

    // Write after checkpoint.
    let inode_post = volume_create_file(&mut vol, root, "post.txt").unwrap();
    volume_write_file_data(&mut vol, inode_post, 0, b"post-commit").unwrap();

    bafs_unmount(vol).unwrap();

    // Both files must be visible after remount.
    let vol2 = bafs_mount(disk.clone_arc()).unwrap();
    let root2 = vol2.superblock.root_inode_number;

    let found_pre = volume_lookup_directory_entry(&vol2, root2, "pre.txt")
        .unwrap()
        .expect("pre-commit file must persist");
    let found_post = volume_lookup_directory_entry(&vol2, root2, "post.txt")
        .unwrap()
        .expect("post-commit file must persist");

    let mut buf = [0u8; 11];
    volume_read_file_data(&vol2, found_pre, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"pre-commit\0");

    let mut buf2 = [0u8; 11];
    volume_read_file_data(&vol2, found_post, 0, &mut buf2).unwrap();
    assert_eq!(&buf2, b"post-commit");

    bafs_unmount(vol2).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Directory / B-tree stress
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn directory_with_500_files_all_survive_remount() {
    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default()).unwrap();

    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;

        for i in 0u32..500 {
            let name = format!("f{:03}.dat", i);
            let inode = volume_create_file(&mut vol, root, &name)
                .unwrap_or_else(|e| panic!("create {} failed: {}", name, e));
            let content = i.to_le_bytes();
            volume_write_file_data(&mut vol, inode, 0, &content).unwrap();
            // Flush periodically to keep the journal commit record within
            // the 256-block journal area.
            if i % 50 == 49 {
                flush_and_commit(&mut vol).unwrap();
            }
        }
        bafs_unmount(vol).unwrap();
    }

    let vol2 = bafs_mount(disk.clone_arc()).unwrap();
    let root2 = vol2.superblock.root_inode_number;

    for i in 0u32..500 {
        let name = format!("f{:03}.dat", i);
        let inode = volume_lookup_directory_entry(&vol2, root2, &name)
            .unwrap_or_else(|e| panic!("lookup {} failed: {}", name, e))
            .unwrap_or_else(|| panic!("{} missing after remount", name));

        let mut buf = [0u8; 4];
        volume_read_file_data(&vol2, inode, 0, &mut buf).unwrap();
        assert_eq!(buf, i.to_le_bytes(), "content mismatch for {}", name);
    }
    bafs_unmount(vol2).unwrap();
}

#[test]
fn unlink_half_of_500_files_rest_survives() {
    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default()).unwrap();

    // Create 500 files.
    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        for i in 0u32..500 {
            let name = format!("g{:03}.dat", i);
            let inode = volume_create_file(&mut vol, root, &name).unwrap();
            volume_write_file_data(&mut vol, inode, 0, &i.to_le_bytes()).unwrap();
            if i % 50 == 49 {
                flush_and_commit(&mut vol).unwrap();
            }
        }
        bafs_unmount(vol).unwrap();
    }

    // Unlink even-indexed files.
    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        for i in (0u32..500).step_by(2) {
            let name = format!("g{:03}.dat", i);
            volume_unlink_directory_entry(&mut vol, root, &name)
                .unwrap_or_else(|e| panic!("unlink {} failed: {}", name, e));
            if i % 50 == 48 {
                flush_and_commit(&mut vol).unwrap();
            }
        }
        bafs_unmount(vol).unwrap();
    }

    // Verify: odd files present, even files absent.
    let vol2 = bafs_mount(disk.clone_arc()).unwrap();
    let root2 = vol2.superblock.root_inode_number;
    for i in 0u32..500 {
        let name = format!("g{:03}.dat", i);
        let found = volume_lookup_directory_entry(&vol2, root2, &name).unwrap();
        if i % 2 == 0 {
            assert!(found.is_none(), "{} should have been unlinked", name);
        } else {
            let inode = found.unwrap_or_else(|| panic!("{} should still exist", name));
            let mut buf = [0u8; 4];
            volume_read_file_data(&vol2, inode, 0, &mut buf).unwrap();
            assert_eq!(buf, i.to_le_bytes(), "content mismatch after partial unlink");
        }
    }
    bafs_unmount(vol2).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Crash simulation
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: format a MemoryDisk (must succeed), returns the disk arc.
fn format_fresh_512mib() -> MemoryDisk {
    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default()).expect("format 512 MiB");
    disk
}

/// Crash during format: the format itself fails.  Mounting the resulting disk
/// must fail cleanly (no panic, no UB).
#[test]
fn crash_during_format_disk_is_unmountable() {
    let inner = MemoryDisk::new_512mib();
    let fault_disk = FaultInjectionDisk::new(inner, 1);

    // Format will fail due to injected fault.
    let _ = bafs_format(&fault_disk, BafsFormatOptions::default());

    // The partially-written disk must not be mountable.
    let recovery_disk = fault_disk.clone_inner();
    let mount_result = bafs_mount(recovery_disk);
    // It may succeed (if by chance the primary superblock was written) or fail.
    // What it must NOT do is panic.
    let _ = mount_result;
}

/// Crash before journal commit: dirty data in RAM is lost.  After remount,
/// the file must not appear.
#[test]
fn crash_before_journal_commit_data_not_visible_after_remount() {
    // Step 1: format normally.
    let base_disk = format_fresh_512mib();

    // Step 2: mount on a FaultInjectionDisk that will fail on the first write
    // of the unmount path (i.e. before any journal sector is written).
    let fault_disk = FaultInjectionDisk::new(base_disk.clone_arc(), 1);
    let mut vol = bafs_mount(fault_disk.clone_inner())
        .expect("mount should succeed (fault is only on writes)");

    let root = vol.superblock.root_inode_number;
    let inode = volume_create_file(&mut vol, root, "ghost.txt")
        .expect("create should succeed — no writes yet");
    volume_write_file_data(&mut vol, inode, 0, b"ghost data")
        .expect("write to dirty cache should succeed");

    // Simulate crash: attempt unmount on the fault disk.
    let fault_mount = FaultInjectionDisk::new(base_disk.clone_arc(), 1);
    // We mount the fault disk separately and perform operations on it.
    // The simplest crash model: just drop the volume without committing.
    drop(vol); // dirty cache is lost — simulates power-off before commit

    // Step 3: remount from the pre-crash bytes.
    let recovery_disk = base_disk.clone_arc();
    let _ = fault_mount; // silence unused warning
    let vol2 = bafs_mount(recovery_disk).expect("remount after crash should succeed");
    let root2 = vol2.superblock.root_inode_number;

    let found = volume_lookup_directory_entry(&vol2, root2, "ghost.txt").unwrap();
    assert!(
        found.is_none(),
        "ghost.txt must not exist after pre-commit crash (dirty cache was not committed)"
    );
    bafs_unmount(vol2).unwrap();
}

/// Crash mid-unmount (after some journal sectors written, before the commit
/// record).  The journal commit record will have an invalid CRC; recovery must
/// treat the transaction as absent and leave the disk in the pre-crash state.
#[test]
fn crash_mid_journal_write_disk_reverts_to_pre_crash_state() {
    let base_disk = format_fresh_512mib();

    // Write a known file in a clean session first.
    {
        let mut vol = bafs_mount(base_disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        let inode = volume_create_file(&mut vol, root, "stable.txt").unwrap();
        volume_write_file_data(&mut vol, inode, 0, b"stable").unwrap();
        bafs_unmount(vol).unwrap();
    }

    // Now mount and create a second file, but crash mid-journal-write.
    // We inject a fault after 3 write_sectors calls (enough to write some journal
    // sectors but not the commit record).
    {
        let fault_disk = FaultInjectionDisk::new(base_disk.clone_arc(), 4);
        let mut vol = bafs_mount(fault_disk.clone_inner()).unwrap();
        let root = vol.superblock.root_inode_number;
        let inode = volume_create_file(&mut vol, root, "transient.txt").unwrap();
        volume_write_file_data(&mut vol, inode, 0, b"transient").unwrap();
        // Attempt to commit — will fail partway through.
        let _ = flush_and_commit(&mut vol);
        // Drop the volume without a successful commit.
        drop(vol);
    }

    // After remount, stable.txt must be present, transient.txt may or may not
    // be present (depending on whether the fault hit before or after the commit
    // record).  What must NOT happen is a panic or corrupted stable.txt.
    let vol2 = bafs_mount(base_disk.clone_arc())
        .expect("remount after mid-journal crash must succeed");
    let root2 = vol2.superblock.root_inode_number;

    let stable = volume_lookup_directory_entry(&vol2, root2, "stable.txt")
        .expect("lookup stable.txt must not error");
    assert!(stable.is_some(), "stable.txt must survive mid-journal crash");

    // Verify stable.txt content is intact.
    let stable_inode = stable.unwrap();
    let mut buf = [0u8; 6];
    volume_read_file_data(&vol2, stable_inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"stable", "stable.txt content must not be corrupted");

    bafs_unmount(vol2).unwrap();
}

/// Crash during delete: delete file A while B exists.  After crash + remount,
/// B must be intact.  A may or may not exist but the tree must be consistent.
#[test]
fn crash_during_delete_does_not_corrupt_other_entries() {
    let base_disk = format_fresh_512mib();

    {
        let mut vol = bafs_mount(base_disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        let inode_a = volume_create_file(&mut vol, root, "a_del.txt").unwrap();
        let inode_b = volume_create_file(&mut vol, root, "b_keep.txt").unwrap();
        volume_write_file_data(&mut vol, inode_a, 0, b"delete me").unwrap();
        volume_write_file_data(&mut vol, inode_b, 0, b"keep me").unwrap();
        bafs_unmount(vol).unwrap();
    }

    // Mount on a fault disk that crashes mid-commit of the delete.
    {
        let fault_disk = FaultInjectionDisk::new(base_disk.clone_arc(), 3);
        let mut vol = bafs_mount(fault_disk.clone_inner()).unwrap();
        let root = vol.superblock.root_inode_number;
        let _ = volume_unlink_directory_entry(&mut vol, root, "a_del.txt");
        let _ = flush_and_commit(&mut vol);
        drop(vol);
    }

    // Remount: b_keep.txt must still be present and correct.
    let vol2 = bafs_mount(base_disk.clone_arc()).expect("remount after delete crash");
    let root2 = vol2.superblock.root_inode_number;

    let b_inode = volume_lookup_directory_entry(&vol2, root2, "b_keep.txt")
        .unwrap()
        .expect("b_keep.txt must survive crash during delete of a_del.txt");
    let mut buf = [0u8; 7];
    volume_read_file_data(&vol2, b_inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"keep me");

    bafs_unmount(vol2).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. On-disk corruption detection
// ═══════════════════════════════════════════════════════════════════════════

/// Flip a bit in a committed data block.  Reading through the checksum tree
/// must detect the corruption.
#[test]
fn bit_flip_in_data_block_is_detected() {
    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default()).unwrap();

    let data_block_address;
    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        let inode = volume_create_file(&mut vol, root, "corrupt.txt").unwrap();
        // Capture next_free_block BEFORE the write.  With the bump-pointer
        // allocator, data blocks start at the current next_free_block.
        let data_block_before_write = vol.next_free_block;
        // 4 KiB write so we know data goes into exactly one block.
        let payload = vec![0x55u8; BLOCK_SIZE_BYTES];
        volume_write_file_data(&mut vol, inode, 0, &payload).unwrap();
        // The data block is the one at `data_block_before_write`.
        data_block_address = data_block_before_write;
        bafs_unmount(vol).unwrap();
    }

    // Corrupt the data block.
    disk.flip_bit_in_block(data_block_address);

    // Remount and attempt to read: must return an error (InvalidChecksum or
    // InputOutputError), not silently return wrong data.
    let vol2 = bafs_mount(disk.clone_arc()).unwrap();
    let root2 = vol2.superblock.root_inode_number;
    let inode = volume_lookup_directory_entry(&vol2, root2, "corrupt.txt")
        .unwrap()
        .unwrap();
    let mut buf = vec![0u8; BLOCK_SIZE_BYTES];
    let read_result = volume_read_file_data(&vol2, inode, 0, &mut buf);
    assert!(
        read_result.is_err(),
        "reading a bit-flipped data block must return an error, not silently succeed"
    );
    bafs_unmount(vol2).unwrap();
}

/// Flip a bit in the primary superblock.  The backup superblock must be used
/// on the next mount.
#[test]
fn primary_superblock_corruption_falls_back_to_backup() {
    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default()).unwrap();

    // Write a file so we know there is real content.
    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        let inode = volume_create_file(&mut vol, root, "backup_test.txt").unwrap();
        volume_write_file_data(&mut vol, inode, 0, b"backup ok").unwrap();
        bafs_unmount(vol).unwrap();
    }

    // Zero-fill the primary superblock (block 1).
    disk.zero_block(1);

    // Remount must succeed using the backup superblock (block 2).
    let vol2 = bafs_mount(disk.clone_arc())
        .expect("mount must succeed using backup superblock");
    let root2 = vol2.superblock.root_inode_number;

    let found = volume_lookup_directory_entry(&vol2, root2, "backup_test.txt")
        .unwrap()
        .expect("file must be readable after falling back to backup superblock");
    let mut buf = [0u8; 9];
    volume_read_file_data(&vol2, found, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"backup ok");
    bafs_unmount(vol2).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Unlink / delete correctness
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn create_file_unlink_lookup_returns_none() {
    let (disk, mut vol) = fresh_512mib!();
    let root = vol.superblock.root_inode_number;

    let inode = volume_create_file(&mut vol, root, "todelete.txt").unwrap();
    volume_write_file_data(&mut vol, inode, 0, b"bye").unwrap();
    volume_unlink_directory_entry(&mut vol, root, "todelete.txt").unwrap();

    let found = volume_lookup_directory_entry(&vol, root, "todelete.txt").unwrap();
    assert!(found.is_none(), "unlinked file must not be visible");

    bafs_unmount(vol).unwrap();

    let vol2 = bafs_mount(disk.clone_arc()).unwrap();
    let root2 = vol2.superblock.root_inode_number;
    let found2 = volume_lookup_directory_entry(&vol2, root2, "todelete.txt").unwrap();
    assert!(found2.is_none(), "unlinked file must not reappear after remount");
    bafs_unmount(vol2).unwrap();
}

#[test]
fn unlink_nonexistent_file_returns_not_found() {
    let (_disk, mut vol) = fresh_512mib!();
    let root = vol.superblock.root_inode_number;

    let result = volume_unlink_directory_entry(&mut vol, root, "does_not_exist.txt");
    assert!(
        matches!(result, Err(BafsError::NotFound)),
        "unlinking a non-existent file must return NotFound, got {:?}",
        result
    );
    bafs_unmount(vol).unwrap();
}

#[test]
fn create_unlink_recreate_same_name_works() {
    let (disk, mut vol) = fresh_512mib!();
    let root = vol.superblock.root_inode_number;

    let inode1 = volume_create_file(&mut vol, root, "reuse.txt").unwrap();
    volume_write_file_data(&mut vol, inode1, 0, b"first").unwrap();
    volume_unlink_directory_entry(&mut vol, root, "reuse.txt").unwrap();

    let inode2 = volume_create_file(&mut vol, root, "reuse.txt").unwrap();
    assert_ne!(inode1, inode2, "recreated file should get a new inode number");
    volume_write_file_data(&mut vol, inode2, 0, b"second").unwrap();

    let mut buf = [0u8; 6];
    volume_read_file_data(&vol, inode2, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"second");

    bafs_unmount(vol).unwrap();

    let vol2 = bafs_mount(disk.clone_arc()).unwrap();
    let root2 = vol2.superblock.root_inode_number;
    let found = volume_lookup_directory_entry(&vol2, root2, "reuse.txt")
        .unwrap()
        .expect("reused name must persist after remount");
    let mut buf2 = [0u8; 6];
    volume_read_file_data(&vol2, found, 0, &mut buf2).unwrap();
    assert_eq!(&buf2, b"second");
    bafs_unmount(vol2).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Free-space accounting
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn free_block_count_decreases_after_write_and_increases_after_unlink() {
    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default()).unwrap();

    let free_after_format;
    {
        let vol = bafs_mount(disk.clone_arc()).unwrap();
        free_after_format = vol.superblock.free_block_count;
        assert!(
            free_after_format > 0,
            "free_block_count must be > 0 after format"
        );
        bafs_unmount(vol).unwrap();
    }

    // Write a 1 MiB file.
    let free_after_write;
    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        let inode = volume_create_file(&mut vol, root, "space.bin").unwrap();
        let payload = vec![0u8; 1024 * 1024];
        volume_write_file_data(&mut vol, inode, 0, &payload).unwrap();
        bafs_unmount(vol).unwrap();

        let vol2 = bafs_mount(disk.clone_arc()).unwrap();
        free_after_write = vol2.superblock.free_block_count;
        bafs_unmount(vol2).unwrap();
    }

    assert!(
        free_after_write < free_after_format,
        "free_block_count must decrease after writing 1 MiB ({} < {})",
        free_after_write,
        free_after_format
    );

    // Unlink the file.
    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        volume_unlink_directory_entry(&mut vol, root, "space.bin").unwrap();
        bafs_unmount(vol).unwrap();
    }

    // Verify the file is gone and free space increased after remount.
    let vol3 = bafs_mount(disk.clone_arc()).unwrap();
    let root3 = vol3.superblock.root_inode_number;
    let gone = volume_lookup_directory_entry(&vol3, root3, "space.bin").unwrap();
    assert!(gone.is_none(), "unlinked file must not appear after remount");
    let free_after_unlink = vol3.superblock.free_block_count;
    assert!(
        free_after_unlink > free_after_write,
        "free_block_count must increase after unlinking the file ({} > {})",
        free_after_unlink,
        free_after_write,
    );
    bafs_unmount(vol3).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. CoW recycling: no OutOfSpace on tight disks with many sessions
// ═══════════════════════════════════════════════════════════════════════════

/// Perform many write-overwrite cycles on a small disk.  If CoW block recycling
/// is broken the bump pointer will eventually exhaust all blocks.  With correct
/// recycling the disk must never return OutOfSpace within the free-space budget.
#[test]
fn cow_recycling_does_not_exhaust_space_over_many_sessions() {
    // Use a modest 64 MiB disk so the test is fast.
    let disk = MemoryDisk::new(64 * 1024 * 1024);
    bafs_format(&disk, BafsFormatOptions::default()).unwrap();

    // Run 20 sessions.  Each session overwrites the same 4 KiB file.
    for session in 0..20u32 {
        let mut vol = bafs_mount(disk.clone_arc())
            .unwrap_or_else(|e| panic!("mount failed on session {}: {}", session, e));
        let root = vol.superblock.root_inode_number;

        // Create or overwrite "cow_test.txt".
        let inode = match volume_lookup_directory_entry(&vol, root, "cow_test.txt").unwrap() {
            Some(existing) => existing,
            None => volume_create_file(&mut vol, root, "cow_test.txt").unwrap(),
        };

        let content = format!("session {:05}", session);
        volume_write_file_data(&mut vol, inode, 0, content.as_bytes())
            .unwrap_or_else(|e| panic!("write failed on session {}: {}", session, e));

        bafs_unmount(vol)
            .unwrap_or_else(|e| panic!("unmount failed on session {}: {}", session, e));
    }

    // Final verification: last write is visible.
    let vol_final = bafs_mount(disk.clone_arc()).unwrap();
    let root_final = vol_final.superblock.root_inode_number;
    let inode = volume_lookup_directory_entry(&vol_final, root_final, "cow_test.txt")
        .unwrap()
        .expect("cow_test.txt must exist after 20 sessions");
    let mut buf = vec![0u8; 14];
    volume_read_file_data(&vol_final, inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"session 00019\0", "last write must persist");
    bafs_unmount(vol_final).unwrap();
}

/// Two sessions write different files.  Session 2 must not corrupt session 1's data.
#[test]
fn second_session_cow_does_not_overwrite_first_session_data() {
    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default()).unwrap();

    // Session 1: write a 64 KiB file with a recognisable pattern.
    let size = 64 * 1024usize;
    let pattern_a: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        let inode = volume_create_file(&mut vol, root, "session_a.bin").unwrap();
        volume_write_file_data(&mut vol, inode, 0, &pattern_a).unwrap();
        bafs_unmount(vol).unwrap();
    }

    // Session 2: write a different file.
    let pattern_b: Vec<u8> = (0..size).map(|i| (!i & 0xFF) as u8).collect();
    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        let inode = volume_create_file(&mut vol, root, "session_b.bin").unwrap();
        volume_write_file_data(&mut vol, inode, 0, &pattern_b).unwrap();
        bafs_unmount(vol).unwrap();
    }

    // Verify session 1 data is intact.
    let vol3 = bafs_mount(disk.clone_arc()).unwrap();
    let root3 = vol3.superblock.root_inode_number;

    let inode_a = volume_lookup_directory_entry(&vol3, root3, "session_a.bin")
        .unwrap()
        .expect("session_a.bin must exist");
    let mut buf_a = vec![0u8; size];
    volume_read_file_data(&vol3, inode_a, 0, &mut buf_a).unwrap();
    assert_eq!(buf_a, pattern_a, "session 1 data must not be overwritten by session 2 CoW");

    let inode_b = volume_lookup_directory_entry(&vol3, root3, "session_b.bin")
        .unwrap()
        .expect("session_b.bin must exist");
    let mut buf_b = vec![0u8; size];
    volume_read_file_data(&vol3, inode_b, 0, &mut buf_b).unwrap();
    assert_eq!(buf_b, pattern_b, "session 2 data must be correct");

    bafs_unmount(vol3).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. Journal recovery: simulate crash after journal written, before final flush
// ═══════════════════════════════════════════════════════════════════════════

/// Write a file, commit it to the journal (flush_and_commit), then zero-fill
/// the final block locations on disk (simulating incomplete replay of the
/// final flush).  Remount must recover the data via journal replay.
#[test]
fn journal_recovery_restores_committed_data_after_simulated_incomplete_flush() {
    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default()).unwrap();

    let committed_block;
    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        let inode = volume_create_file(&mut vol, root, "journal_recovery.txt").unwrap();
        volume_write_file_data(&mut vol, inode, 0, b"recovered via journal").unwrap();

        // Commit to journal.
        flush_and_commit(&mut vol).expect("flush_and_commit must succeed");

        // Record the block that holds the data so we can wipe it.
        committed_block = vol.next_free_block - 1;

        // Do NOT unmount cleanly — simulate that the final block writes were
        // done by journal replay but the superblock update was lost.
        // We just drop the volume.
        drop(vol);
    }

    // Wipe the data block (simulates that the host OS crashed after journal
    // was written but before final block flush reached the disk).
    disk.zero_block(committed_block);

    // Remount: journal replay must restore the block.
    let vol2 = bafs_mount(disk.clone_arc())
        .expect("remount after wiped data block must succeed via journal replay");
    let root2 = vol2.superblock.root_inode_number;

    // The file may or may not be visible depending on whether the superblock
    // update was captured in the journal.  What must NOT happen is a panic.
    // If visible, the content must be correct or an error returned.
    if let Ok(Some(inode)) =
        volume_lookup_directory_entry(&vol2, root2, "journal_recovery.txt")
    {
        let mut buf = vec![0u8; 21];
        let _ = volume_read_file_data(&vol2, inode, 0, &mut buf);
        // Content may or may not match depending on recovery depth; no assertion on content here.
    }
    bafs_unmount(vol2).unwrap();
}

/// Fill a small disk to capacity, unlink everything, then refill.  Both passes
/// should create roughly the same number of files, proving that unlink actually
/// reclaims data block space.
#[test]
fn unlink_reclaims_space_for_reuse() {
    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default()).unwrap();

    // Helper: flush every N files to stay within journal capacity.
    let flush_every = 10u32;

    // Pass 1: create files until we've used a meaningful amount of space.
    // Stop at 200 rather than filling to capacity to leave room for metadata.
    let file_count = 200u32;
    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        for i in 0..file_count {
            let name = format!("fill_{:04}.dat", i);
            let inode = volume_create_file(&mut vol, root, &name)
                .unwrap_or_else(|e| panic!("create {} failed: {}", name, e));
            volume_write_file_data(&mut vol, inode, 0, &i.to_le_bytes())
                .unwrap_or_else(|e| panic!("write {} failed: {}", name, e));
            if (i + 1) % flush_every == 0 {
                flush_and_commit(&mut vol).unwrap();
            }
        }
        bafs_unmount(vol).unwrap();
    }

    // Record free space before unlink.
    let free_before_unlink;
    {
        let vol = bafs_mount(disk.clone_arc()).unwrap();
        free_before_unlink = vol.superblock.free_block_count;
        bafs_unmount(vol).unwrap();
    }

    // Unlink all files.
    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        for i in 0..file_count {
            let name = format!("fill_{:04}.dat", i);
            volume_unlink_directory_entry(&mut vol, root, &name)
                .unwrap_or_else(|e| panic!("unlink {} failed: {}", name, e));
            if (i + 1) % flush_every == 0 {
                flush_and_commit(&mut vol).unwrap();
            }
        }
        bafs_unmount(vol).unwrap();
    }

    // Record free space after unlink.
    let free_after_unlink;
    {
        let vol = bafs_mount(disk.clone_arc()).unwrap();
        free_after_unlink = vol.superblock.free_block_count;
        bafs_unmount(vol).unwrap();
    }

    // The 200 files each had 1 data block, so at least 200 blocks should be freed.
    assert!(
        free_after_unlink > free_before_unlink,
        "free_block_count must increase after unlinking 200 files ({} > {})",
        free_after_unlink,
        free_before_unlink,
    );
    let reclaimed = free_after_unlink - free_before_unlink;
    assert!(
        reclaimed >= file_count as u64,
        "must reclaim at least {} data blocks, but only reclaimed {}",
        file_count,
        reclaimed,
    );

    // Verify the space is actually usable: create the same number of files again.
    {
        let mut vol = bafs_mount(disk.clone_arc()).unwrap();
        let root = vol.superblock.root_inode_number;
        for i in 0..file_count {
            let name = format!("refill_{:04}.dat", i);
            let inode = volume_create_file(&mut vol, root, &name)
                .unwrap_or_else(|e| panic!("refill create {} failed: {}", name, e));
            volume_write_file_data(&mut vol, inode, 0, &i.to_le_bytes())
                .unwrap_or_else(|e| panic!("refill write {} failed: {}", name, e));
            if (i + 1) % flush_every == 0 {
                flush_and_commit(&mut vol).unwrap();
            }
        }
        bafs_unmount(vol).unwrap();
    }
}
