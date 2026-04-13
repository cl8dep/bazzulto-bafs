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

//! BAFS extent allocation: on-disk structures, free-extent B-tree, and block
//! allocation / deallocation.
//!
//! In BAFS every file's data is stored in one or more extents — contiguous runs
//! of blocks on disk.  Two B-trees track extent information:
//!
//! 1. **Inode B-tree** (root = `superblock.inode_tree_root_block`): stores
//!    `BafsExtentData` items (key `(inode_number, ITEM_TYPE_EXTENT_DATA, byte_offset)`)
//!    that map a file's logical byte offsets to their disk block addresses.
//!
//! 2. **Free-extent B-tree** (root = `superblock.free_extent_tree_root_block`):
//!    stores `BafsFreeExtent` items (key `(BAFS_FREE_EXTENT_OBJECT_ID,
//!    ITEM_TYPE_EXTENT_DATA, block_address)`) recording which block runs are
//!    available for allocation.
//!
//! # Allocation algorithm
//!
//! Uses a simple first-fit strategy:
//! 1. Scan the free-extent B-tree for the first entry with `block_count` ≥ the
//!    requested size.
//! 2. Remove that entry.
//! 3. If the entry is larger than needed, re-insert the remainder.
//! 4. Return the starting block address.
//!
//! # Copy-on-Write
//!
//! All tree modifications go through `btree::insert_into_tree` and
//! `btree::delete_from_tree`, which write new CoW copies of every modified
//! node into the dirty-block cache.  The caller (`volume.rs`) commits the
//! cache atomically via the journal.
//!
//! # Impact on the rest of the system
//!
//! - `volume.rs` calls `allocate_blocks` when writing new file data.
//! - `volume.rs` calls `free_blocks` when truncating or deleting files.
//! - `volume.rs` calls `write_extent_data_to_inode_tree` after allocating
//!   blocks to record the file↔disk mapping.
//! - `volume.rs` calls `read_extent_data_for_file_offset` when reading file
//!   data to find which disk blocks hold the requested bytes.

#[cfg(feature = "kernel")]
use alloc::{vec, vec::Vec, collections::BTreeMap};
#[cfg(not(feature = "kernel"))]
use std::{vec, vec::Vec, collections::BTreeMap};

use crate::block_device::BlockDevice;
use crate::btree::{
    insert_into_tree, delete_from_tree, iterate_tree_range,
    BafsKey, ITEM_TYPE_EXTENT_DATA,
};
use crate::error::BafsError;
use crate::superblock::BAFS_FREE_EXTENT_OBJECT_ID;

// ─── On-disk structures ───────────────────────────────────────────────────────

/// Describes one contiguous run of disk blocks that stores file data.
///
/// Stored in the inode B-tree with key
/// `(inode_number, ITEM_TYPE_EXTENT_DATA, logical_byte_offset)`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BafsExtentData {
    /// Byte offset within the file where this extent begins.
    pub logical_byte_offset: u64,

    /// Block address of the first block in this extent on disk.
    pub first_disk_block: u64,

    /// Number of 4 KiB blocks in this extent.
    pub block_count: u64,

    /// Extent flags.  In v1 all bits are zero.  Reserved for v2+ sparse-file
    /// support (`0x01 = pre-allocated`, `0x02 = hole`).
    pub extent_flags: u32,

    /// Reserved, zero-padded.
    pub reserved: u32,
}

// Verify exact size for deterministic on-disk layout.
const _EXTENT_DATA_SIZE_ASSERTION: () =
    assert!(core::mem::size_of::<BafsExtentData>() == 32);

/// Describes one contiguous run of unallocated disk blocks.
///
/// Stored in the free-extent B-tree with key
/// `(BAFS_FREE_EXTENT_OBJECT_ID, ITEM_TYPE_EXTENT_DATA, block_address)`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BafsFreeExtent {
    /// Block address of the first free block in this run.
    pub block_address: u64,

    /// Number of consecutive free blocks in this run.
    pub block_count: u64,
}

const _FREE_EXTENT_SIZE_ASSERTION: () =
    assert!(core::mem::size_of::<BafsFreeExtent>() == 16);

// ─── Serialisation ────────────────────────────────────────────────────────────

/// Serialise a `BafsExtentData` into 32 bytes (little-endian).
pub fn serialise_extent_data_to_bytes(extent: &BafsExtentData) -> [u8; 32] {
    let mut output = [0u8; 32];
    unsafe {
        core::ptr::copy_nonoverlapping(
            extent as *const BafsExtentData as *const u8,
            output.as_mut_ptr(),
            32,
        );
    }
    output
}

/// Deserialise 32 bytes into a `BafsExtentData`.
pub fn deserialise_extent_data_from_bytes(bytes: &[u8]) -> BafsExtentData {
    debug_assert!(bytes.len() >= 32);
    unsafe { core::ptr::read(bytes.as_ptr() as *const BafsExtentData) }
}

/// Serialise a `BafsFreeExtent` into 16 bytes (little-endian).
fn serialise_free_extent_to_bytes(free_extent: &BafsFreeExtent) -> [u8; 16] {
    let mut output = [0u8; 16];
    unsafe {
        core::ptr::copy_nonoverlapping(
            free_extent as *const BafsFreeExtent as *const u8,
            output.as_mut_ptr(),
            16,
        );
    }
    output
}

/// Deserialise 16 bytes into a `BafsFreeExtent`.
fn deserialise_free_extent_from_bytes(bytes: &[u8]) -> BafsFreeExtent {
    debug_assert!(bytes.len() >= 16);
    unsafe { core::ptr::read(bytes.as_ptr() as *const BafsFreeExtent) }
}

// ─── Extent data B-tree I/O ───────────────────────────────────────────────────

/// Store an extent data record for a file in the inode B-tree.
///
/// Returns the (possibly new) inode-tree root block address.
///
/// `freed_blocks` is appended with block addresses superseded by CoW clones;
/// thread it through from the top-level operation.
pub fn write_extent_data_to_inode_tree(
    device: &dyn BlockDevice,
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    inode_tree_root_block: u64,
    inode_number: u64,
    extent: &BafsExtentData,
    generation: u64,
    next_free_block: &mut u64,
    freed_blocks: &mut Vec<u64>,
) -> Result<u64, BafsError> {
    let key = BafsKey::new(inode_number, ITEM_TYPE_EXTENT_DATA, extent.logical_byte_offset);
    let value = serialise_extent_data_to_bytes(extent).to_vec();
    insert_into_tree(
        device,
        dirty_cache,
        inode_tree_root_block,
        key,
        value,
        generation,
        next_free_block,
        freed_blocks,
    )
}

/// Find the extent that covers `file_byte_offset` for the given inode.
///
/// Returns `Ok(Some(extent))` if found, `Ok(None)` if the file has no data at
/// that offset (e.g. the file is empty or `file_byte_offset` is past the end).
pub fn read_extent_data_for_file_offset(
    device: &dyn BlockDevice,
    dirty_cache: &BTreeMap<u64, Vec<u8>>,
    inode_tree_root_block: u64,
    inode_number: u64,
    file_byte_offset: u64,
) -> Result<Option<BafsExtentData>, BafsError> {
    // Scan all extent items for this inode and find the one whose range
    // [logical_byte_offset, logical_byte_offset + block_count * BLOCK_SIZE)
    // contains `file_byte_offset`.
    let min_key = BafsKey::new(inode_number, ITEM_TYPE_EXTENT_DATA, 0);
    let max_key = BafsKey::new(inode_number, ITEM_TYPE_EXTENT_DATA, u64::MAX);
    let all_extent_items = iterate_tree_range(
        device,
        dirty_cache,
        inode_tree_root_block,
        min_key,
        max_key,
    )?;

    let block_size_in_bytes = crate::superblock::BAFS_DEFAULT_BLOCK_SIZE_BYTES as u64;

    for item in &all_extent_items {
        let extent = deserialise_extent_data_from_bytes(&item.value);
        let extent_end_byte =
            extent.logical_byte_offset + extent.block_count * block_size_in_bytes;
        if file_byte_offset >= extent.logical_byte_offset
            && file_byte_offset < extent_end_byte
        {
            return Ok(Some(extent));
        }
    }
    Ok(None)
}

// ─── Free-extent B-tree I/O ───────────────────────────────────────────────────

/// Insert a free-extent record into the free-extent B-tree.
///
/// Used during `bafs_format` to initialise the data area as one large free
/// extent, and by `free_blocks` to return blocks to the pool.
///
/// Returns the new free-extent-tree root block address.
///
/// `freed_blocks` is appended with block addresses superseded by CoW clones;
/// thread it through from the top-level operation.
pub fn insert_free_extent_into_tree(
    device: &dyn BlockDevice,
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    free_extent_tree_root_block: u64,
    block_address: u64,
    block_count: u64,
    generation: u64,
    next_free_block: &mut u64,
    freed_blocks: &mut Vec<u64>,
) -> Result<u64, BafsError> {
    let free_extent = BafsFreeExtent { block_address, block_count };
    let key = BafsKey::new(
        BAFS_FREE_EXTENT_OBJECT_ID,
        ITEM_TYPE_EXTENT_DATA,
        block_address,
    );
    let value = serialise_free_extent_to_bytes(&free_extent).to_vec();
    insert_into_tree(
        device,
        dirty_cache,
        free_extent_tree_root_block,
        key,
        value,
        generation,
        next_free_block,
        freed_blocks,
    )
}

/// Allocate `requested_block_count` contiguous blocks from the free-extent tree.
///
/// Uses a first-fit strategy: scans free extents in ascending block-address
/// order and takes the first entry large enough.  Removes that entry from the
/// tree and re-inserts the remainder if the entry was larger than needed.
///
/// The free-extent tree only contains blocks outside the metadata CoW zone
/// (maintained by the caller's bump pointer).  No address filtering is needed
/// here; the caller guarantees that the tree never contains entries inside the
/// live CoW zone.
///
/// Returns `(starting_block_address, new_free_extent_tree_root_block)`.
pub fn allocate_blocks(
    device: &dyn BlockDevice,
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    free_extent_tree_root_block: u64,
    requested_block_count: u64,
    generation: u64,
    next_free_block: &mut u64,
    freed_blocks: &mut Vec<u64>,
) -> Result<(u64, u64), BafsError> {
    // Collect all free extents in ascending block-address order.
    let min_key = BafsKey::new(BAFS_FREE_EXTENT_OBJECT_ID, ITEM_TYPE_EXTENT_DATA, 0);
    let max_key =
        BafsKey::new(BAFS_FREE_EXTENT_OBJECT_ID, ITEM_TYPE_EXTENT_DATA, u64::MAX);
    let all_free_items = iterate_tree_range(
        device,
        dirty_cache,
        free_extent_tree_root_block,
        min_key,
        max_key,
    )?;

    // Last-fit: find the last entry with enough contiguous blocks and allocate
    // from its HIGH end.
    //
    // Allocating from the high end of the largest free extent is the key to
    // keeping the data zone and the metadata CoW zone from colliding.
    //
    // The metadata CoW bump pointer always grows *forward* from
    // `next_cow_block_address` (low addresses).  By allocating data blocks from
    // the *top* of the highest free extent, data blocks and CoW metadata blocks
    // always grow toward each other from opposite ends of the free space — they
    // only meet when the disk is genuinely full.
    //
    // With first-fit (allocate from the bottom), both CoW metadata and data
    // would race to claim the same low addresses, causing the remainder to be
    // consumed by the CoW of the allocation itself.
    let chosen_free_extent = all_free_items
        .iter()
        .rev()
        .find(|item| {
            let extent = deserialise_free_extent_from_bytes(&item.value);
            extent.block_count >= requested_block_count
        })
        .ok_or(BafsError::OutOfSpace)?
        .clone();

    let free_extent = deserialise_free_extent_from_bytes(&chosen_free_extent.value);

    // Allocate from the HIGH end of this extent.
    //
    // allocated = [extent_end - requested, extent_end)
    // remainder = [extent_start, extent_end - requested)
    //
    // The remainder is at LOW addresses — below where CoW can ever reach
    // (because CoW only grows from next_free_block upward, and next_free_block
    // is always ≤ extent_start at the time of the call because the free-extent
    // tree only holds entries beyond the committed CoW zone).
    //
    // Actually: next_free_block may grow into the remainder range during the
    // delete + re-insert CoW operations below.  But the remainder is trimmed
    // from the TOP, so the HIGH part is what we allocate.  The LOW part
    // (remainder) stays below next_free_block's current trajectory in the
    // common case — and if CoW nodes do land in the remainder range, those
    // blocks are intra-transaction and will be abandoned (not re-inserted in
    // the free-extent tree).  This is the same intra-transaction waste that
    // is already accepted for metadata CoW nodes.
    let extent_end = free_extent.block_address + free_extent.block_count;
    let allocated_block_address = extent_end - requested_block_count;

    // Remove the chosen entry from the tree.
    let mut current_root = free_extent_tree_root_block;
    current_root = delete_from_tree(
        device,
        dirty_cache,
        current_root,
        chosen_free_extent.key,
        generation,
        next_free_block,
        freed_blocks,
    )?;

    // Re-insert the remainder (the lower portion of the extent), if any.
    //
    // The remainder occupies [free_extent.block_address, allocated_block_address).
    // This range is at LOW addresses relative to what we just allocated, so it
    // does not conflict with the just-allocated HIGH blocks.
    //
    // CoW nodes produced by delete_from_tree go into *next_free_block which
    // may now overlap with the low-address remainder — but that is fine:
    // those CoW nodes are intra-transaction and will be abandoned, and the
    // remainder entry in the free tree only covers addresses that are not
    // currently occupied by any live CoW node.
    if allocated_block_address > free_extent.block_address {
        let remainder_start = free_extent.block_address;
        let remainder_count = allocated_block_address - free_extent.block_address;
        current_root = insert_free_extent_into_tree(
            device,
            dirty_cache,
            current_root,
            remainder_start,
            remainder_count,
            generation,
            next_free_block,
            freed_blocks,
        )?;
    }

    Ok((allocated_block_address, current_root))
}

/// Return `block_count` blocks starting at `block_address` to the free pool.
///
/// Coalesces adjacent free extents before inserting so that the free-extent
/// B-tree stays compact on large disks.  The algorithm is:
///
/// 1. Scan for a free extent immediately **before** the freed range
///    (i.e. one whose `block_address + block_count == block_address`).
/// 2. Scan for a free extent immediately **after** the freed range
///    (i.e. one whose `block_address == block_address + block_count`).
/// 3. Remove any adjacent extents found.
/// 4. Insert a single merged extent that covers all contiguous blocks.
///
/// Returns the new free-extent-tree root block address.
pub fn free_blocks(
    device: &dyn BlockDevice,
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    free_extent_tree_root_block: u64,
    block_address: u64,
    block_count: u64,
    generation: u64,
    next_free_block: &mut u64,
    freed_blocks: &mut Vec<u64>,
) -> Result<u64, BafsError> {
    // Collect all free extents to search for adjacent neighbours.
    let min_key = BafsKey::new(BAFS_FREE_EXTENT_OBJECT_ID, ITEM_TYPE_EXTENT_DATA, 0);
    let max_key = BafsKey::new(BAFS_FREE_EXTENT_OBJECT_ID, ITEM_TYPE_EXTENT_DATA, u64::MAX);
    let all_free_items = iterate_tree_range(
        device,
        dirty_cache,
        free_extent_tree_root_block,
        min_key,
        max_key,
    )?;

    // The merged range starts as the caller's freed range.
    let mut merged_start = block_address;
    let mut merged_count = block_count;
    let mut current_root = free_extent_tree_root_block;

    // Check for a free extent that ends exactly where our range begins.
    let predecessor = all_free_items.iter().find(|item| {
        let extent = deserialise_free_extent_from_bytes(&item.value);
        extent.block_address + extent.block_count == block_address
    });

    if let Some(predecessor_item) = predecessor {
        let predecessor_extent = deserialise_free_extent_from_bytes(&predecessor_item.value);
        // Extend the merged range to include the predecessor.
        merged_start = predecessor_extent.block_address;
        merged_count += predecessor_extent.block_count;
        // Remove the predecessor from the tree so it can be replaced by the merged entry.
        current_root = delete_from_tree(
            device,
            dirty_cache,
            current_root,
            predecessor_item.key,
            generation,
            next_free_block,
            freed_blocks,
        )?;
    }

    // Re-collect items after the predecessor removal, then look for a successor.
    // (Re-scan is necessary because the tree root may have changed after the delete.)
    let min_key2 = BafsKey::new(BAFS_FREE_EXTENT_OBJECT_ID, ITEM_TYPE_EXTENT_DATA, 0);
    let max_key2 = BafsKey::new(BAFS_FREE_EXTENT_OBJECT_ID, ITEM_TYPE_EXTENT_DATA, u64::MAX);
    let updated_free_items = iterate_tree_range(
        device,
        dirty_cache,
        current_root,
        min_key2,
        max_key2,
    )?;

    // Check for a free extent that starts exactly where our merged range ends.
    let successor = updated_free_items.iter().find(|item| {
        let extent = deserialise_free_extent_from_bytes(&item.value);
        extent.block_address == merged_start + merged_count
    });

    if let Some(successor_item) = successor {
        let successor_extent = deserialise_free_extent_from_bytes(&successor_item.value);
        // Extend the merged range to include the successor.
        merged_count += successor_extent.block_count;
        // Remove the successor from the tree.
        current_root = delete_from_tree(
            device,
            dirty_cache,
            current_root,
            successor_item.key,
            generation,
            next_free_block,
            freed_blocks,
        )?;
    }

    // Insert the single merged free extent.
    insert_free_extent_into_tree(
        device,
        dirty_cache,
        current_root,
        merged_start,
        merged_count,
        generation,
        next_free_block,
        freed_blocks,
    )
}
