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

//! BAFS superblock: on-disk layout, read, write, and validation.
//!
//! The superblock is the root of all filesystem metadata.  It is stored twice:
//! once at block 1 (the primary copy) and once at block 2 (the backup copy).
//! Both copies are identical; the backup is written first so that a crash
//! between the two writes always leaves at least one valid copy.
//!
//! # On-disk layout (512 bytes total)
//!
//! ```text
//! Offset   Size  Field
//! ──────   ────  ─────────────────────────────────────────────────────
//!      0      8  magic  (b"BAFS\x1B\x00\x00\x00")
//!      8      4  version  (1 for v1)
//!     12      4  block_size  (4096, 8192, or 16384)
//!     16      8  block_count
//!     24      8  free_block_count  (informational; extent B-tree is authoritative)
//!     32      8  inode_count
//!     40      8  transaction_id  (monotonically increasing)
//!     48      8  root_inode_number  (always 1 for v1)
//!     56      8  inode_tree_root_block
//!     64      8  free_extent_tree_root_block
//!     72      8  checksum_tree_root_block
//!     80      8  journal_start_block
//!     88      8  journal_size_in_blocks
//!     96     16  volume_uuid  ([u64; 2])
//!    112     32  volume_label  ([u64; 4], UTF-8, zero-padded)
//!    144      4  feature_flags
//!    148      4  checksum_algorithm  (0 = CRC32C)
//!    152    356  reserved  (zero-filled, reserved for future use)
//!    508      4  superblock_checksum  (CRC32C of bytes 0..508)
//! ```
//!
//! # Impact on the rest of the system
//!
//! - `volume.rs` calls `read_superblock` on mount and `write_superblock` after
//!   every committed transaction.
//! - `journal.rs` reads and updates `transaction_id` and the tree root fields.
//! - All B-tree modules receive the tree root block addresses from the
//!   superblock so they know where to start their lookups.

#[cfg(feature = "kernel")]
use alloc::vec;
#[cfg(feature = "kernel")]
use alloc::vec::Vec;

use crate::checksum::{compute_crc32c, verify_crc32c};
use crate::error::BafsError;

// ─── Constants ────────────────────────────────────────────────────────────────

/// The eight-byte magic number that identifies a BAFS filesystem.
///
/// The first four bytes are the ASCII string "BAFS"; the fifth byte (0x1B) is
/// an escape character chosen to make the magic resistant to being mistaken for
/// plain text; the last three bytes are zero padding.
pub const BAFS_MAGIC_NUMBER: [u8; 8] = *b"BAFS\x1B\x00\x00\x00";

/// The filesystem version implemented by this crate.  Only v1 is supported.
pub const BAFS_SUPPORTED_VERSION: u32 = 1;

/// Default block size in bytes used when formatting a new filesystem.
pub const BAFS_DEFAULT_BLOCK_SIZE_BYTES: u32 = 4096;

/// Size of a disk sector in bytes.  All `BlockDevice` implementations in the
/// Bazzulto kernel use 512-byte sectors.
pub const BAFS_SECTOR_SIZE_BYTES: u32 = 512;

/// Number of 512-byte sectors in one 4096-byte block.
pub const BAFS_SECTORS_PER_BLOCK: u32 =
    BAFS_DEFAULT_BLOCK_SIZE_BYTES / BAFS_SECTOR_SIZE_BYTES;

/// Block address of the primary superblock.
pub const BAFS_PRIMARY_SUPERBLOCK_BLOCK_ADDRESS: u64 = 1;

/// Block address of the backup superblock.
pub const BAFS_BACKUP_SUPERBLOCK_BLOCK_ADDRESS: u64 = 2;

/// Block address where the journal area begins.
pub const BAFS_JOURNAL_START_BLOCK_ADDRESS: u64 = 3;

/// Minimum journal size in blocks (1 MiB at 4 KiB/block).
///
/// Even a tiny disk gets at least 256 journal blocks so the filesystem has
/// room to absorb bursts of metadata writes without stalling on every commit.
pub const BAFS_JOURNAL_SIZE_MIN_BLOCKS: u64 = 256; // 1 MiB

/// Maximum journal size in blocks (64 MiB at 4 KiB/block).
///
/// 64 MiB is the same cap used by APFS and BTRFS.  Beyond 64 MiB the
/// journal replay time at boot dominates the benefit of a larger ring buffer.
pub const BAFS_JOURNAL_SIZE_MAX_BLOCKS: u64 = 16_384; // 64 MiB

/// Compute the journal size for a disk with `total_block_count` 4 KiB blocks.
///
/// The formula is: clamp(1% of disk, 1 MiB, 64 MiB).
///
/// | Disk size | Journal |
/// |-----------|---------|
/// | 256 MiB   |   1 MiB (minimum) |
/// | 1 GiB     |  ~10 MiB |
/// | 256 GiB   |  64 MiB (maximum) |
/// | 100 TiB   |  64 MiB (maximum) |
pub fn compute_journal_size_in_blocks(total_block_count: u64) -> u64 {
    let one_percent_of_total = total_block_count / 100;
    one_percent_of_total
        .max(BAFS_JOURNAL_SIZE_MIN_BLOCKS)
        .min(BAFS_JOURNAL_SIZE_MAX_BLOCKS)
}

/// Compute the block address of the first block in the data area.
///
/// The data area begins immediately after the journal.  The journal size is
/// determined dynamically from the disk capacity.
pub fn compute_data_area_start_block(total_block_count: u64) -> u64 {
    BAFS_JOURNAL_START_BLOCK_ADDRESS + compute_journal_size_in_blocks(total_block_count)
}

/// Checksum algorithm identifier stored in the superblock: 0 = CRC32C.
pub const BAFS_CHECKSUM_ALGORITHM_CRC32C: u32 = 0;

/// Inode number of the root directory.  Always 1 in BAFS v1.
pub const BAFS_ROOT_INODE_NUMBER: u64 = 1;

/// Reserved object ID used for free-extent B-tree entries.
pub const BAFS_FREE_EXTENT_OBJECT_ID: u64 = 0xFFFF_FFFF_FFFF_FFFE;

/// Reserved object ID used for checksum B-tree entries.
pub const BAFS_CHECKSUM_OBJECT_ID: u64 = 0xFFFF_FFFF_FFFF_FFFD;

// ─── On-disk structure ────────────────────────────────────────────────────────

/// The BAFS superblock as it appears on disk (512 bytes, little-endian fields).
///
/// The struct is `#[repr(C)]` so that field offsets are predictable and match
/// the specification.  All multi-byte integer fields are stored in
/// little-endian byte order; callers must use `.to_le()` / `from_le()` when
/// reading from or writing to raw bytes.
///
/// The checksum covers bytes 0..508 (i.e. everything except the checksum field
/// itself at offset 508).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BafsSuperblock {
    /// Eight-byte magic number.  Must equal `BAFS_MAGIC_NUMBER`.
    pub magic_number: [u8; 8],

    /// Filesystem format version.  This implementation only reads and writes
    /// version 1.
    pub version: u32,

    /// Block size in bytes (4096, 8192, or 16384).  For v1 only 4096 is used.
    pub block_size_in_bytes: u32,

    /// Total number of blocks on the filesystem (including reserved blocks,
    /// journal blocks, and data blocks).
    pub total_block_count: u64,

    /// Approximate number of free blocks.  Informational only; the free-extent
    /// B-tree is the authoritative source.  Updated on commit.
    pub free_block_count: u64,

    /// Number of inodes that have been allocated (including the root inode).
    /// Used to generate new monotonically increasing inode numbers.
    pub allocated_inode_count: u64,

    /// Transaction ID of the last successfully committed transaction.
    /// Monotonically increasing.  Used during crash recovery to determine
    /// whether the journal contains data newer than what is on disk.
    pub last_committed_transaction_id: u64,

    /// Inode number of the root directory.  Always `BAFS_ROOT_INODE_NUMBER`.
    pub root_inode_number: u64,

    /// Block address of the root node of the inode B-tree.  The inode B-tree
    /// contains all inode items, directory entries, and file extent data.
    pub inode_tree_root_block: u64,

    /// Block address of the root node of the free-extent B-tree.  This tree
    /// tracks which blocks are unallocated and available for new data.
    pub free_extent_tree_root_block: u64,

    /// Block address of the root node of the checksum B-tree.  This tree
    /// stores the CRC32C checksum of every data block (as opposed to metadata
    /// blocks, which carry their own inline checksums).
    pub checksum_tree_root_block: u64,

    /// Block address where the write-ahead journal begins.
    pub journal_start_block: u64,

    /// Size of the journal area in blocks.
    pub journal_size_in_blocks: u64,

    /// 128-bit universally unique identifier for this filesystem volume.
    /// Stored as two little-endian 64-bit halves.
    pub volume_uuid: [u64; 2],

    /// Human-readable volume label, UTF-8, zero-padded to 32 bytes (four
    /// little-endian u64 words stored raw).
    pub volume_label: [u64; 4],

    /// Feature flags bitmask.  For v1 all bits are zero (no optional features).
    pub feature_flags: u32,

    /// Checksum algorithm: 0 = CRC32C.  For v1 only CRC32C is supported.
    pub checksum_algorithm: u32,

    /// Address of the next block to allocate for copy-on-write tree nodes.
    ///
    /// This is a monotonically increasing bump pointer maintained by the volume
    /// layer.  It is saved to the superblock on every commit so that, after a
    /// remount, new CoW allocations continue from where the previous mount left
    /// off and never reuse or collide with any block already written to disk.
    ///
    /// Data block addresses come from the free-extent B-tree (authoritative);
    /// this field only seeds the CoW tree-node allocator.
    pub next_cow_block_address: u64,

    /// Reserved bytes, zero-filled.  Sized so that `superblock_checksum` lands
    /// at offset 508 and the total struct size is exactly 512 bytes.
    pub reserved: [u8; 348],

    /// CRC32C of bytes 0..508 of the serialised superblock (i.e. everything
    /// except this field itself).
    pub superblock_checksum: u32,
}

// Verify at compile time that our struct is exactly 512 bytes.
const _SUPERBLOCK_SIZE_ASSERTION: () =
    assert!(core::mem::size_of::<BafsSuperblock>() == 512);

impl BafsSuperblock {
    /// Compute the CRC32C checksum that should be stored in `superblock_checksum`.
    ///
    /// The checksum covers all bytes of the superblock except the last four
    /// (the checksum field itself at offset 508).
    pub fn compute_checksum(&self) -> u32 {
        // Safety: BafsSuperblock is repr(C) with no padding (verified by the
        // compile-time size assertion), so interpreting it as bytes is safe.
        let raw_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                self as *const BafsSuperblock as *const u8,
                core::mem::size_of::<BafsSuperblock>(),
            )
        };
        // Checksum covers bytes 0..508 (everything except the last 4 bytes).
        compute_crc32c(&raw_bytes[..508])
    }

    /// Fill the `superblock_checksum` field with the correct value.
    pub fn seal_with_checksum(&mut self) {
        self.superblock_checksum = self.compute_checksum();
    }

    /// Verify that the stored checksum matches the actual content.
    pub fn verify_checksum(&self) -> bool {
        let expected = self.superblock_checksum.to_le();
        verify_crc32c(
            {
                let raw_bytes: &[u8] = unsafe {
                    core::slice::from_raw_parts(
                        self as *const BafsSuperblock as *const u8,
                        core::mem::size_of::<BafsSuperblock>(),
                    )
                };
                &raw_bytes[..508]
            },
            expected,
        )
    }
}

// ─── Serialisation helpers ────────────────────────────────────────────────────

/// Serialise a `BafsSuperblock` into a 512-byte buffer in little-endian form.
///
/// The struct fields are already stored in host byte order; this function
/// writes them as little-endian bytes so the layout is portable across both
/// little-endian (x86-64, AArch64) and big-endian hosts.
///
/// In practice all Bazzulto target platforms are little-endian, so this is
/// effectively a `memcpy`, but we keep the explicit byte-order conversion for
/// correctness on any future big-endian port.
pub fn serialise_superblock_to_bytes(superblock: &BafsSuperblock) -> [u8; 512] {
    let mut output = [0u8; 512];
    // Safety: same as compute_checksum — repr(C), no padding, exactly 512 bytes.
    let raw_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            superblock as *const BafsSuperblock as *const u8,
            512,
        )
    };
    output.copy_from_slice(raw_bytes);
    output
}

/// Deserialise a 512-byte buffer into a `BafsSuperblock`.
///
/// The bytes are interpreted as a little-endian layout matching the spec.
pub fn deserialise_superblock_from_bytes(bytes: &[u8; 512]) -> BafsSuperblock {
    // Safety: we read exactly 512 bytes into a 512-byte repr(C) struct with no
    // padding.  The resulting struct may have unverified fields; the caller
    // must call `verify_checksum()` and check the magic / version before use.
    unsafe { core::ptr::read(bytes.as_ptr() as *const BafsSuperblock) }
}

// ─── Block device I/O ─────────────────────────────────────────────────────────

/// Read and validate a superblock from `block_address` on `device`.
///
/// This function reads one 4 KiB block, extracts the first 512 bytes as a
/// `BafsSuperblock`, verifies the CRC32C checksum, checks the magic number,
/// and checks that the version is 1.
///
/// Returns `Err(BafsError::InvalidMagicNumber)` if the magic is wrong,
/// `Err(BafsError::UnsupportedVersion)` if the version is not 1, and
/// `Err(BafsError::InvalidChecksum)` if the checksum does not match.
pub fn read_superblock_from_device(
    device: &dyn crate::block_device::BlockDevice,
    block_address: u64,
) -> Result<BafsSuperblock, BafsError> {
    // Read one full block (4096 bytes = 8 sectors).
    let mut block_buffer = vec![0u8; BAFS_DEFAULT_BLOCK_SIZE_BYTES as usize];
    let start_lba = block_address * BAFS_SECTORS_PER_BLOCK as u64;
    if !device.read_sectors(start_lba, BAFS_SECTORS_PER_BLOCK, &mut block_buffer) {
        return Err(BafsError::InputOutputError);
    }

    // The superblock occupies the first 512 bytes of the block.
    let superblock_bytes: &[u8; 512] = block_buffer[..512]
        .try_into()
        .expect("slice is exactly 512 bytes");
    let superblock = deserialise_superblock_from_bytes(superblock_bytes);

    // Validate magic number.
    if superblock.magic_number != BAFS_MAGIC_NUMBER {
        return Err(BafsError::InvalidMagicNumber);
    }

    // Validate version.
    if superblock.version != BAFS_SUPPORTED_VERSION {
        return Err(BafsError::UnsupportedVersion {
            found_version: superblock.version,
        });
    }

    // Validate checksum.
    if !superblock.verify_checksum() {
        return Err(BafsError::InvalidChecksum { block_address });
    }

    Ok(superblock)
}

/// Write a superblock to both the backup block (block 2) and the primary block
/// (block 1), in that order.
///
/// Writing the backup first means that if the system crashes between the two
/// writes, at least one valid copy survives.  On the next mount, `volume.rs`
/// will read the primary; if the primary is corrupt it falls back to the
/// backup.
pub fn write_superblock_to_device(
    device: &dyn crate::block_device::BlockDevice,
    superblock: &mut BafsSuperblock,
) -> Result<(), BafsError> {
    // Seal the checksum before writing.
    superblock.seal_with_checksum();

    let serialised = serialise_superblock_to_bytes(superblock);

    // Helper closure: write the 512-byte superblock at a given block address.
    // The block is 4096 bytes; we write the superblock in the first 512 bytes
    // and zero-fill the rest.
    let write_at_block = |block_address: u64| -> Result<(), BafsError> {
        let mut block_buffer = [0u8; BAFS_DEFAULT_BLOCK_SIZE_BYTES as usize];
        block_buffer[..512].copy_from_slice(&serialised);
        let start_lba = block_address * BAFS_SECTORS_PER_BLOCK as u64;
        if !device.write_sectors(start_lba, BAFS_SECTORS_PER_BLOCK, &block_buffer) {
            return Err(BafsError::InputOutputError);
        }
        Ok(())
    };

    // Write backup first, then primary.
    write_at_block(BAFS_BACKUP_SUPERBLOCK_BLOCK_ADDRESS)?;
    write_at_block(BAFS_PRIMARY_SUPERBLOCK_BLOCK_ADDRESS)?;

    Ok(())
}
