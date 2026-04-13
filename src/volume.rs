//! BAFS volume: format, mount, unmount, and high-level file/directory operations.
//!
//! `BafsVolume` is the top-level handle to a mounted BAFS filesystem.  Every
//! public operation that the kernel VFS layer or userspace tools need goes
//! through this type.  Internally it:
//!
//! - Holds the current superblock (in memory, with updates staged until commit).
//! - Owns the dirty-block cache (`BTreeMap<block_address, block_data>`), which
//!   acts as the write buffer for the current transaction.
//! - Manages the `next_free_block` counter that feeds new CoW allocations.
//! - Calls the journal module to commit all dirty blocks atomically.
//!
//! # Concurrency model (v1)
//!
//! v1 assumes a single-threaded kernel with no SMP.  All operations on the
//! volume are called serially; there are no locks.  When SMP is introduced
//! (v2+), callers must wrap the volume in a mutex.
//!
//! # Impact on the rest of the system
//!
//! - `kernel.rs` holds an `Arc<UnsafeCell<BafsVolume>>` and implements the
//!   kernel `Inode` trait by delegating to these functions.
//! - `tests/integration_test.rs` calls these functions directly through a
//!   `MemoryDisk` adapter.

#[cfg(feature = "kernel")]
use alloc::{collections::BTreeMap, vec, vec::Vec};
#[cfg(not(feature = "kernel"))]
use std::{collections::BTreeMap, vec, vec::Vec};

use crate::block_device::BlockDevice;
use crate::btree::{create_empty_tree, delete_from_tree, iterate_tree_range, write_block_to_cache, BafsKey, ITEM_TYPE_EXTENT_DATA, ITEM_TYPE_INODE};
use crate::checksum_tree::{store_data_block_checksum, verify_data_block_checksum};
use crate::dir::{
    directory_create_entry, directory_iterate, directory_list_all_entries,
    directory_lookup_entry, directory_unlink_entry, BafsDirEntry, CHILD_TYPE_DIRECTORY,
    CHILD_TYPE_REGULAR_FILE,
};
use crate::error::BafsError;
use crate::extent::{
    allocate_blocks, deserialise_extent_data_from_bytes, free_blocks,
    insert_free_extent_into_tree, read_extent_data_for_file_offset,
    write_extent_data_to_inode_tree, BafsExtentData,
};
use crate::inode::{
    allocate_next_inode_number, read_inode_from_tree, write_inode_to_tree, BafsInode,
    INODE_MODE_DIRECTORY,
};
use crate::journal::{begin_transaction, commit_transaction, recover_journal_on_mount};
use crate::superblock::{
    compute_data_area_start_block, compute_journal_size_in_blocks,
    read_superblock_from_device, write_superblock_to_device, BafsSuperblock,
    BAFS_BACKUP_SUPERBLOCK_BLOCK_ADDRESS, BAFS_CHECKSUM_ALGORITHM_CRC32C,
    BAFS_DEFAULT_BLOCK_SIZE_BYTES, BAFS_JOURNAL_START_BLOCK_ADDRESS,
    BAFS_FREE_EXTENT_OBJECT_ID, BAFS_MAGIC_NUMBER, BAFS_PRIMARY_SUPERBLOCK_BLOCK_ADDRESS,
    BAFS_ROOT_INODE_NUMBER, BAFS_SECTORS_PER_BLOCK, BAFS_SUPPORTED_VERSION,
};

// ─── Format options ───────────────────────────────────────────────────────────

/// Options for `bafs_format`.  All fields have sensible defaults via
/// `BafsFormatOptions::default()`.
pub struct BafsFormatOptions {
    /// Volume UUID lower half (high half is always 0 in default).
    pub volume_uuid_lower: u64,

    /// Volume label as a UTF-8 string up to 32 bytes.  Truncated silently if
    /// longer.
    pub volume_label: [u8; 32],
}

impl Default for BafsFormatOptions {
    fn default() -> Self {
        BafsFormatOptions {
            volume_uuid_lower: 0xBAF5_0001_0000_0001,
            volume_label: *b"BAFS Volume\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
        }
    }
}

// ─── Block addresses reserved at format time ──────────────────────────────────
//
// These are NOT compile-time constants because the journal size — and therefore
// the data-area start — depends on the total disk capacity.  All format-time
// block addresses are computed at runtime inside `bafs_format`.

// ─── CoW block allocation model ───────────────────────────────────────────────
//
// BAFS uses two completely separate allocators — one for metadata, one for
// file data — so that neither can interfere with the other.  This mirrors the
// design of APFS, which keeps a separate node-pool allocator for its B-tree
// metadata distinct from the container free-list that manages file data.
//
// METADATA (B-tree CoW nodes)
//   Allocated by a simple linear bump pointer stored in
//   `superblock.next_cow_block_address`.  On mount this pointer is loaded into
//   `BafsVolume::next_free_block`.  Every B-tree CoW clone takes the next
//   address from this pointer and increments it.  The pointer only moves
//   forward; freed CoW nodes are simply abandoned (old snapshots of a path
//   after a CoW are not reclaimed in v1).  The final value of the pointer is
//   persisted to the superblock on every commit.
//
// FILE DATA (extents)
//   Allocated from the free-extent B-tree via `allocate_blocks`.  The
//   free-extent tree tracks ranges of blocks that are available for user data.
//   It is completely independent of the metadata bump pointer.
//
// Because CoW nodes always receive real disk addresses (not virtual
// placeholders) there is NO address-translation step at commit time.  Commit
// simply journals all dirty blocks — both CoW metadata nodes and data blocks —
// exactly as they are.
//   3. Builds a translation map: virtual CoW address → real disk address.
//   4. Rewrites every reference inside dirty blocks (in node headers and
//      internal-node child pointers) from virtual to real addresses.
//   5. Writes the translated dirty cache to the journal and then to disk.
//

// ─── Volume handle ────────────────────────────────────────────────────────────

/// A mounted BAFS filesystem volume.
///
/// The generic parameter `D` must implement `BlockDevice`.  In the kernel build
/// `D` is typically `Arc<dyn BlockDevice>`; in tests it is the `MemoryDisk`
/// struct defined in `tests/integration_test.rs`.
pub struct BafsVolume<D: BlockDevice> {
    /// The block device backing this volume.
    pub device: D,

    /// In-memory copy of the superblock.  Written to disk on every commit.
    pub superblock: BafsSuperblock,

    /// Dirty-block cache for the current in-progress transaction.  Maps block
    /// address → 4 KiB block data.  Flushed to disk and cleared on commit.
    pub dirty_block_cache: BTreeMap<u64, Vec<u8>>,

    /// Counter for the next transaction ID.  Incremented after each commit.
    pub next_transaction_id: u64,

    /// Pointer to the next unallocated block.  Kept in sync with the
    /// free-extent B-tree; updated after every `allocate_blocks` call.
    pub next_free_block: u64,
}

// ─── Format ───────────────────────────────────────────────────────────────────

/// Format a block device as a new, empty BAFS v1 filesystem.
///
/// This function writes:
/// 1. An empty inode B-tree root at block `FORMAT_INODE_TREE_ROOT_BLOCK`.
/// 2. An empty free-extent B-tree root.
/// 3. An empty checksum B-tree root.
/// 4. The root directory inode (inode number 1) inside the inode B-tree.
/// 5. One free-extent entry covering all blocks not consumed by steps 1–4.
/// 6. The primary and backup superblocks.
/// 7. Zero-fills the journal area.
///
/// After this function returns, `bafs_mount` can be called on the same device.
pub fn bafs_format(device: &dyn BlockDevice, options: BafsFormatOptions) -> Result<(), BafsError> {
    let block_size = BAFS_DEFAULT_BLOCK_SIZE_BYTES as usize;
    let sectors_per_block = BAFS_SECTORS_PER_BLOCK;

    // Compute total block count from device size.
    let total_sector_count = device.total_sector_count();
    let total_block_count = total_sector_count / sectors_per_block as u64;

    // Compute the journal size and data-area layout dynamically from disk capacity.
    // This scales from 1 MiB on tiny disks up to 64 MiB on large disks (≥ 256 GiB).
    let journal_size_in_blocks = compute_journal_size_in_blocks(total_block_count);
    let data_area_start_block = compute_data_area_start_block(total_block_count);

    // Block addresses for the three initial B-tree roots and first allocatable block.
    let format_inode_tree_root_block: u64 = data_area_start_block;
    let format_free_extent_tree_root_block: u64 = data_area_start_block + 1;
    let format_checksum_tree_root_block: u64 = data_area_start_block + 2;
    let format_first_allocatable_block: u64 = data_area_start_block + 3;

    if total_block_count < format_first_allocatable_block + 10 {
        // Device is too small to hold even the bare minimum structures.
        return Err(BafsError::OutOfSpace);
    }

    // ── Build format-time structures in a dirty-block cache ──────────────────

    let mut dirty_cache: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
    // During format, CoW nodes are allocated from format_first_allocatable_block
    // upward using a real bump pointer.  These blocks are NOT registered in the
    // free-extent tree; instead, the free-extent tree's initial entry is adjusted
    // to start after however many format-time CoW nodes were consumed.
    let mut next_free_block = format_first_allocatable_block;
    let format_generation: u64 = 1;

    // ── (1) Empty inode B-tree root ───────────────────────────────────────────
    create_empty_tree(
        &mut dirty_cache,
        format_inode_tree_root_block,
        format_generation,
    );

    // ── (2) Empty free-extent B-tree root ────────────────────────────────────
    //
    // The free-extent entry is NOT inserted yet.  We insert it last, after all
    // other CoW tree operations have consumed blocks from format_first_allocatable_block
    // upward.  At that point next_free_block tells us where the real free data
    // starts, and we insert exactly one entry: (next_free_block, remaining_count).
    create_empty_tree(
        &mut dirty_cache,
        format_free_extent_tree_root_block,
        format_generation,
    );
    let mut free_extent_root = format_free_extent_tree_root_block;

    // ── (3) Empty checksum B-tree root ────────────────────────────────────────
    create_empty_tree(
        &mut dirty_cache,
        format_checksum_tree_root_block,
        format_generation,
    );
    let mut checksum_root = format_checksum_tree_root_block;
    let mut inode_root = format_inode_tree_root_block;

    // ── (4) Root directory inode ──────────────────────────────────────────────

    let root_inode = BafsInode::new_directory(
        BAFS_ROOT_INODE_NUMBER,
        format_generation,
        0, // timestamp: zero for a freshly formatted volume
    );

    let mut format_freed_blocks: Vec<u64> = Vec::new();
    inode_root = write_inode_to_tree(
        device,
        &mut dirty_cache,
        inode_root,
        &root_inode,
        format_generation,
        &mut next_free_block,
        &mut format_freed_blocks,
    )?;

    // ── (5) Build the superblock ──────────────────────────────────────────────

    let mut label_words = [0u64; 4];
    {
        let label_bytes: &[u8; 32] = unsafe {
            core::slice::from_raw_parts(
                label_words.as_ptr() as *const u8,
                32,
            )
            .try_into()
            .unwrap_or(&[0u8; 32])
        };
        // Copy the label bytes into the word array via raw bytes.
        let _ = label_bytes; // suppress unused warning; write directly below
    }
    // Write label bytes directly.
    let label_bytes_ref: &mut [u8; 32] = unsafe {
        core::slice::from_raw_parts_mut(label_words.as_mut_ptr() as *mut u8, 32)
            .try_into()
            .expect("32-byte slice is 32 bytes")
    };
    label_bytes_ref.copy_from_slice(&options.volume_label);

    // ── (5b) Insert the free-extent tree entry last ───────────────────────────
    //
    // All preceding CoW tree operations (inode tree, checksum tree) used
    // next_free_block as a real bump pointer starting at format_first_allocatable_block.
    // Those blocks are now permanently consumed by tree nodes.  The free extent
    // must start at the current value of next_free_block, not at
    // format_first_allocatable_block.
    //
    // The insert itself may produce at most one additional CoW node (a root
    // split on an empty tree never occurs, so the tree stays height-0 for 1
    // item).  If next_free_block advances during the insert, we delete the
    // entry we just inserted and re-insert with the updated start.  This loop
    // converges in at most 2 iterations.

    {
        let free_start_first_attempt = next_free_block;
        let free_count_first_attempt = total_block_count
            .saturating_sub(free_start_first_attempt);
        if free_count_first_attempt == 0 {
            return Err(BafsError::OutOfSpace);
        }
        free_extent_root = insert_free_extent_into_tree(
            device,
            &mut dirty_cache,
            free_extent_root,
            free_start_first_attempt,
            free_count_first_attempt,
            format_generation,
            &mut next_free_block,
            &mut format_freed_blocks,
        )?;

        // If next_free_block advanced, the entry we just inserted is too broad
        // (it claims blocks that were just consumed as CoW nodes).  Fix it.
        if next_free_block > free_start_first_attempt {
            let old_key = BafsKey::new(
                BAFS_FREE_EXTENT_OBJECT_ID,
                ITEM_TYPE_EXTENT_DATA,
                free_start_first_attempt,
            );
            free_extent_root = delete_from_tree(
                device,
                &mut dirty_cache,
                free_extent_root,
                old_key,
                format_generation,
                &mut next_free_block,
                &mut format_freed_blocks,
            )?;

            let free_start_final = next_free_block;
            let free_count_final = total_block_count.saturating_sub(free_start_final);
            if free_count_final > 0 {
                free_extent_root = insert_free_extent_into_tree(
                    device,
                    &mut dirty_cache,
                    free_extent_root,
                    free_start_final,
                    free_count_final,
                    format_generation,
                    &mut next_free_block,
                    &mut format_freed_blocks,
                )?;
            }
        }
    }

    let final_free_block_count = total_block_count.saturating_sub(next_free_block);

    let mut superblock = BafsSuperblock {
        magic_number: BAFS_MAGIC_NUMBER,
        version: BAFS_SUPPORTED_VERSION,
        block_size_in_bytes: BAFS_DEFAULT_BLOCK_SIZE_BYTES,
        total_block_count,
        free_block_count: final_free_block_count,
        allocated_inode_count: 1, // root inode
        last_committed_transaction_id: 1,
        root_inode_number: BAFS_ROOT_INODE_NUMBER,
        inode_tree_root_block: inode_root,
        free_extent_tree_root_block: free_extent_root,
        checksum_tree_root_block: checksum_root,
        journal_start_block: BAFS_JOURNAL_START_BLOCK_ADDRESS,
        journal_size_in_blocks,
        volume_uuid: [options.volume_uuid_lower, 0],
        volume_label: label_words,
        feature_flags: 0,
        checksum_algorithm: BAFS_CHECKSUM_ALGORITHM_CRC32C,
        next_cow_block_address: next_free_block,
        reserved: [0u8; 348],
        superblock_checksum: 0, // filled by write_superblock_to_device
    };

    // ── (6) Zero the journal area ─────────────────────────────────────────────

    let zeroed_block = vec![0u8; block_size];
    for journal_block_index in 0..journal_size_in_blocks {
        let lba = (BAFS_JOURNAL_START_BLOCK_ADDRESS + journal_block_index)
            * sectors_per_block as u64;
        if !device.write_sectors(lba, sectors_per_block, &zeroed_block) {
            return Err(BafsError::InputOutputError);
        }
    }

    // ── (8) Write all dirty blocks to their final locations ───────────────────

    for (&block_address, block_data) in &dirty_cache {
        let lba = block_address * sectors_per_block as u64;
        if !device.write_sectors(lba, sectors_per_block, block_data) {
            return Err(BafsError::InputOutputError);
        }
    }

    // ── (9) Write the superblock (backup first, then primary) ─────────────────

    write_superblock_to_device(device, &mut superblock)?;

    Ok(())
}

// ─── Mount ────────────────────────────────────────────────────────────────────

/// Mount a BAFS filesystem from `device` and return a `BafsVolume` handle.
///
/// Reads and validates the superblock, then runs journal recovery to handle
/// any crash that occurred during the previous mount.  After this function
/// returns, all file and directory operations are available.
pub fn bafs_mount<D: BlockDevice>(device: D) -> Result<BafsVolume<D>, BafsError> {
    // Read the primary superblock.
    let mut superblock =
        match read_superblock_from_device(&device, BAFS_PRIMARY_SUPERBLOCK_BLOCK_ADDRESS) {
            Ok(sb) => sb,
            Err(_primary_error) => {
                // Primary is corrupt; try the backup.
                read_superblock_from_device(
                    &device,
                    BAFS_BACKUP_SUPERBLOCK_BLOCK_ADDRESS,
                )?
            }
        };

    // Run journal recovery before any other I/O.
    recover_journal_on_mount(&device, &mut superblock)?;

    // The metadata bump pointer is persisted in the superblock so that across
    // mounts CoW nodes always advance into fresh territory and never overwrite
    // previously written blocks.
    let next_free_block = superblock.next_cow_block_address;
    let next_transaction_id = superblock.last_committed_transaction_id + 1;

    Ok(BafsVolume {
        device,
        superblock,
        dirty_block_cache: BTreeMap::new(),
        next_transaction_id,
        next_free_block,
    })
}

// ─── Unmount / flush ──────────────────────────────────────────────────────────

/// Flush all pending writes and unmount the volume.
///
/// Commits the current dirty-block cache to the journal, writes all dirty
/// blocks to their final locations, and updates the superblock.  After this
/// function returns the `BafsVolume` is consumed and must not be used again.
pub fn bafs_unmount<D: BlockDevice>(mut volume: BafsVolume<D>) -> Result<(), BafsError> {
    flush_and_commit(&mut volume)
}

/// Commit the current dirty-block cache to disk without consuming the volume.
///
/// This is called by `bafs_unmount` and can also be called explicitly to
/// create a checkpoint mid-session.
///
/// All dirty blocks — both CoW metadata nodes and file-data blocks — carry
/// real disk addresses from the moment they are created, so no address
/// translation is needed at commit time.
pub fn flush_and_commit<D: BlockDevice>(volume: &mut BafsVolume<D>) -> Result<(), BafsError> {
    // Persist the metadata bump-pointer so the next mount resumes from the
    // correct position and never reuses blocks written in previous sessions.
    volume.superblock.next_cow_block_address = volume.next_free_block;

    // Commit all dirty blocks (metadata CoW nodes + file-data blocks) to the
    // journal and then to their final on-disk locations.  No address translation
    // is needed because every block already carries a real disk address.
    let transaction = begin_transaction(volume.next_transaction_id);
    commit_transaction(
        &volume.device,
        &transaction,
        &volume.dirty_block_cache,
        &mut volume.superblock,
    )?;

    volume.dirty_block_cache.clear();
    volume.next_transaction_id += 1;

    Ok(())
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Return CoW-obsolete block addresses to the free-extent tree for reuse.
///
/// Every B-tree insert or delete clones the modified path (CoW), making the
/// old nodes obsolete.  This function sorts those obsolete blocks into two
/// categories:
///
/// - **Intra-transaction** (address ≥ `superblock.next_cow_block_address`):
///   these nodes were allocated during the *current* session's bump-pointer
///   run and never reached disk.  They are simply ignored — the bump pointer
///   already moved past them and no future allocation will reuse those
///   addresses.  Critically, they are NOT evicted from the dirty cache,
///   because the same address could legitimately have been reused for a
///   different purpose within the same transaction.
///
/// - **Cross-session** (address < `superblock.next_cow_block_address`):
///   these nodes were committed to disk in a previous session and then
///   superseded by a CoW clone in this session.  They are genuine free space
///   and are inserted back into the free-extent tree so future allocations
///   (both data and metadata) can reuse them.
///
/// The free-extent tree modifications themselves produce CoW obsolete blocks;
/// those are always intra-transaction and handled by the recursion terminating
/// immediately on the second call.
fn return_freed_cow_blocks_to_free_extent_tree<D: BlockDevice>(
    volume: &mut BafsVolume<D>,
    freed_blocks: Vec<u64>,
) -> Result<(), BafsError> {
    if freed_blocks.is_empty() {
        return Ok(());
    }
    let committed_boundary = volume.superblock.next_cow_block_address;
    let generation = volume.next_transaction_id;
    let mut secondary_freed: Vec<u64> = Vec::new();

    for block_address in freed_blocks {
        if block_address >= committed_boundary {
            // Intra-transaction: never hit disk.  Leave the dirty cache alone —
            // do NOT remove, because the address may have been reused within
            // this transaction for a different node.
            continue;
        }
        // Cross-session: block is on disk and no pointer references it anymore.
        // Return it to the free-extent tree.
        let new_root = free_blocks(
            &volume.device,
            &mut volume.dirty_block_cache,
            volume.superblock.free_extent_tree_root_block,
            block_address,
            1,
            generation,
            &mut volume.next_free_block,
            &mut secondary_freed,
        )?;
        volume.superblock.free_extent_tree_root_block = new_root;
        volume.superblock.free_block_count += 1;
    }

    // secondary_freed contains CoW nodes produced by the free_blocks calls above.
    // All of them are intra-transaction (>= committed_boundary), so the recursion
    // returns immediately without any further free_blocks calls.
    return_freed_cow_blocks_to_free_extent_tree(volume, secondary_freed)
}

/// Allocate `requested_block_count` blocks and write `data` to them.
///
/// Returns the block address of the first allocated block.  Updates the
/// free-extent and checksum trees.
fn allocate_and_write_data_blocks<D: BlockDevice>(
    volume: &mut BafsVolume<D>,
    data: &[u8],
) -> Result<u64, BafsError> {
    let block_size = BAFS_DEFAULT_BLOCK_SIZE_BYTES as usize;
    let generation = volume.next_transaction_id;

    // How many blocks do we need?
    let block_count_needed = data.len().div_ceil(block_size) as u64;
    if block_count_needed == 0 {
        return Err(BafsError::InvalidArgument);
    }

    // Allocate data blocks from the bump pointer.
    //
    // Both metadata CoW nodes and file data are allocated from the same linear
    // bump pointer (`next_free_block`).  This is the only design that avoids
    // the CoW-of-the-allocator paradox: any attempt to allocate data from a
    // separate free-extent B-tree causes the CoW nodes of that very B-tree
    // operation to collide with the data blocks being allocated, because both
    // live in the same address space.
    //
    // The free-extent tree is retained for recycling blocks freed by
    // `volume_unlink_directory_entry` (cross-session blocks returned by
    // `return_freed_cow_blocks_to_free_extent_tree`), but new allocations
    // always come from the bump pointer first.
    //
    // Out-of-space check: compare current bump pointer against total blocks.
    let total_blocks = volume.superblock.total_block_count;
    if volume.next_free_block + block_count_needed > total_blocks {
        return Err(BafsError::OutOfSpace);
    }

    let mut freed_blocks: Vec<u64> = Vec::new();
    let allocated_block_address = volume.next_free_block;
    volume.next_free_block += block_count_needed;
    volume.superblock.free_block_count = volume.superblock.free_block_count
        .saturating_sub(block_count_needed);

    // Write the data blocks to the dirty cache and record their checksums.
    let mut bytes_written = 0usize;
    for block_index in 0..block_count_needed as usize {
        let chunk_start = block_index * block_size;
        let chunk_end = (chunk_start + block_size).min(data.len());

        let mut block_buffer = vec![0u8; block_size];
        let chunk = &data[chunk_start..chunk_end];
        block_buffer[..chunk.len()].copy_from_slice(chunk);

        let this_block_address = allocated_block_address + block_index as u64;

        // Write to dirty cache.
        write_block_to_cache(
            &mut volume.dirty_block_cache,
            this_block_address,
            block_buffer.clone(),
        );

        // Record checksum.
        let new_checksum_root = store_data_block_checksum(
            &volume.device,
            &mut volume.dirty_block_cache,
            volume.superblock.checksum_tree_root_block,
            this_block_address,
            &block_buffer,
            generation,
            &mut volume.next_free_block,
            &mut freed_blocks,
        )?;
        volume.superblock.checksum_tree_root_block = new_checksum_root;

        bytes_written += chunk.len();
    }
    let _ = bytes_written;

    // Return CoW-obsolete metadata blocks to the free-extent tree.
    return_freed_cow_blocks_to_free_extent_tree(volume, freed_blocks)?;

    Ok(allocated_block_address)
}

// ─── Inode operations ─────────────────────────────────────────────────────────

/// Read an inode by number from the volume.
pub fn volume_read_inode<D: BlockDevice>(
    volume: &BafsVolume<D>,
    inode_number: u64,
) -> Result<BafsInode, BafsError> {
    read_inode_from_tree(
        &volume.device,
        &volume.dirty_block_cache,
        volume.superblock.inode_tree_root_block,
        inode_number,
    )
}

/// Write (insert or update) an inode into the volume's inode tree.
///
/// Returns nothing; updates `volume.superblock.inode_tree_root_block`
/// internally.
fn volume_write_inode<D: BlockDevice>(
    volume: &mut BafsVolume<D>,
    inode: &BafsInode,
) -> Result<(), BafsError> {
    let generation = volume.next_transaction_id;
    let mut freed_blocks: Vec<u64> = Vec::new();
    let new_root = write_inode_to_tree(
        &volume.device,
        &mut volume.dirty_block_cache,
        volume.superblock.inode_tree_root_block,
        inode,
        generation,
        &mut volume.next_free_block,
        &mut freed_blocks,
    )?;
    volume.superblock.inode_tree_root_block = new_root;
    return_freed_cow_blocks_to_free_extent_tree(volume, freed_blocks)
}

/// Allocate a new inode number and increment the superblock counter.
fn volume_allocate_inode_number<D: BlockDevice>(volume: &mut BafsVolume<D>) -> u64 {
    let new_inode_number =
        allocate_next_inode_number(volume.superblock.allocated_inode_count);
    volume.superblock.allocated_inode_count = new_inode_number;
    new_inode_number
}

// ─── File creation ────────────────────────────────────────────────────────────

/// Create a new regular file named `filename` inside directory `parent_inode_number`.
///
/// Returns the new file's inode number.  The caller should then call
/// `volume_write_file_data` to populate the file with content.
pub fn volume_create_file<D: BlockDevice>(
    volume: &mut BafsVolume<D>,
    parent_inode_number: u64,
    filename: &str,
) -> Result<u64, BafsError> {
    let generation = volume.next_transaction_id;

    // Ensure the parent inode exists and is a directory.
    let parent_inode = volume_read_inode(volume, parent_inode_number)?;
    if !parent_inode.is_directory() {
        return Err(BafsError::NotADirectory);
    }

    // Allocate a new inode number.
    let new_inode_number = volume_allocate_inode_number(volume);

    // Create the file inode.
    let file_inode = BafsInode::new_regular_file(new_inode_number, generation, 0);
    volume_write_inode(volume, &file_inode)?;

    // Add the directory entry in the parent directory.
    let mut freed_blocks: Vec<u64> = Vec::new();
    let new_inode_root = directory_create_entry(
        &volume.device,
        &mut volume.dirty_block_cache,
        volume.superblock.inode_tree_root_block,
        parent_inode_number,
        filename,
        new_inode_number,
        CHILD_TYPE_REGULAR_FILE,
        generation,
        &mut volume.next_free_block,
        &mut freed_blocks,
    )?;
    volume.superblock.inode_tree_root_block = new_inode_root;
    return_freed_cow_blocks_to_free_extent_tree(volume, freed_blocks)?;

    Ok(new_inode_number)
}

/// Create a new directory named `dirname` inside directory `parent_inode_number`.
///
/// Returns the new directory's inode number.
pub fn volume_create_directory<D: BlockDevice>(
    volume: &mut BafsVolume<D>,
    parent_inode_number: u64,
    dirname: &str,
) -> Result<u64, BafsError> {
    let generation = volume.next_transaction_id;

    // Ensure the parent inode exists and is a directory.
    let parent_inode = volume_read_inode(volume, parent_inode_number)?;
    if !parent_inode.is_directory() {
        return Err(BafsError::NotADirectory);
    }

    // Allocate a new inode number.
    let new_inode_number = volume_allocate_inode_number(volume);

    // Create the directory inode.
    let dir_inode = BafsInode::new_directory(new_inode_number, generation, 0);
    volume_write_inode(volume, &dir_inode)?;

    // Add the directory entry in the parent directory.
    let mut freed_blocks: Vec<u64> = Vec::new();
    let new_inode_root = directory_create_entry(
        &volume.device,
        &mut volume.dirty_block_cache,
        volume.superblock.inode_tree_root_block,
        parent_inode_number,
        dirname,
        new_inode_number,
        CHILD_TYPE_DIRECTORY,
        generation,
        &mut volume.next_free_block,
        &mut freed_blocks,
    )?;
    volume.superblock.inode_tree_root_block = new_inode_root;
    return_freed_cow_blocks_to_free_extent_tree(volume, freed_blocks)?;

    Ok(new_inode_number)
}

// ─── File I/O ──────────────────────────────────────────────────────────────────

/// Write `data` to the file identified by `file_inode_number` at byte offset
/// `write_offset_bytes`.
///
/// For v1, only writing at offset 0 (new files) is fully supported.  Appending
/// (offset = current file size) and overwrites update the inode's
/// `file_size_in_bytes` correctly.
///
/// Returns the number of bytes written.
pub fn volume_write_file_data<D: BlockDevice>(
    volume: &mut BafsVolume<D>,
    file_inode_number: u64,
    write_offset_bytes: u64,
    data: &[u8],
) -> Result<usize, BafsError> {
    if data.is_empty() {
        return Ok(0);
    }

    let generation = volume.next_transaction_id;
    let block_size = BAFS_DEFAULT_BLOCK_SIZE_BYTES as u64;

    // Read the current inode.
    let mut file_inode = volume_read_inode(volume, file_inode_number)?;
    if !file_inode.is_regular_file() {
        return Err(BafsError::NotARegularFile);
    }

    // Allocate blocks and write the data.
    let first_allocated_block = allocate_and_write_data_blocks(volume, data)?;

    // Record the extent in the inode B-tree.
    let block_count_allocated =
        (data.len() as u64 + block_size - 1) / block_size;
    let extent = BafsExtentData {
        logical_byte_offset: write_offset_bytes,
        first_disk_block: first_allocated_block,
        block_count: block_count_allocated,
        extent_flags: 0,
        reserved: 0,
    };

    let mut freed_blocks: Vec<u64> = Vec::new();
    let new_inode_root = write_extent_data_to_inode_tree(
        &volume.device,
        &mut volume.dirty_block_cache,
        volume.superblock.inode_tree_root_block,
        file_inode_number,
        &extent,
        generation,
        &mut volume.next_free_block,
        &mut freed_blocks,
    )?;
    volume.superblock.inode_tree_root_block = new_inode_root;
    return_freed_cow_blocks_to_free_extent_tree(volume, freed_blocks)?;

    // Update inode metadata.
    let new_size = (write_offset_bytes + data.len() as u64)
        .max(file_inode.file_size_in_bytes);
    file_inode.file_size_in_bytes = new_size;
    file_inode.allocated_block_count += block_count_allocated;
    file_inode.generation = generation;
    volume_write_inode(volume, &file_inode)?;

    Ok(data.len())
}

/// Read up to `destination_buffer.len()` bytes from `file_inode_number`
/// starting at byte offset `read_offset_bytes`.
///
/// Returns the number of bytes actually read (may be less than
/// `destination_buffer.len()` if the file is shorter than the requested range).
pub fn volume_read_file_data<D: BlockDevice>(
    volume: &BafsVolume<D>,
    file_inode_number: u64,
    read_offset_bytes: u64,
    destination_buffer: &mut [u8],
) -> Result<usize, BafsError> {
    if destination_buffer.is_empty() {
        return Ok(0);
    }

    let block_size = BAFS_DEFAULT_BLOCK_SIZE_BYTES as u64;

    // Read the inode to get the file size.
    let file_inode = volume_read_inode(volume, file_inode_number)?;
    if !file_inode.is_regular_file() {
        return Err(BafsError::NotARegularFile);
    }

    let file_size = file_inode.file_size_in_bytes;
    if read_offset_bytes >= file_size {
        return Ok(0); // Reading past the end of file.
    }

    let bytes_available = file_size - read_offset_bytes;
    let bytes_to_read = (destination_buffer.len() as u64).min(bytes_available) as usize;
    let mut bytes_read_so_far = 0usize;

    while bytes_read_so_far < bytes_to_read {
        let current_offset = read_offset_bytes + bytes_read_so_far as u64;

        // Find the extent that covers `current_offset`.
        let extent = read_extent_data_for_file_offset(
            &volume.device,
            &volume.dirty_block_cache,
            volume.superblock.inode_tree_root_block,
            file_inode_number,
            current_offset,
        )?
        .ok_or(BafsError::CorruptedStructure)?;

        // How far into the first block of this extent are we?
        let offset_within_extent = current_offset - extent.logical_byte_offset;
        let block_index_within_extent = offset_within_extent / block_size;
        let offset_within_block = (offset_within_extent % block_size) as usize;

        // Read the block.
        let block_address = extent.first_disk_block + block_index_within_extent;
        let block_data =
            crate::btree::read_block(&volume.device, &volume.dirty_block_cache, block_address)?;

        // Verify the data block's checksum.
        verify_data_block_checksum(
            &volume.device,
            &volume.dirty_block_cache,
            volume.superblock.checksum_tree_root_block,
            block_address,
            &block_data,
        )?;

        // Copy bytes from this block into the destination buffer.
        let bytes_available_in_block = block_size as usize - offset_within_block;
        let bytes_to_copy_from_block =
            bytes_available_in_block.min(bytes_to_read - bytes_read_so_far);

        destination_buffer
            [bytes_read_so_far..bytes_read_so_far + bytes_to_copy_from_block]
            .copy_from_slice(
                &block_data[offset_within_block..offset_within_block + bytes_to_copy_from_block],
            );

        bytes_read_so_far += bytes_to_copy_from_block;
    }

    Ok(bytes_read_so_far)
}

// ─── Directory operations ─────────────────────────────────────────────────────

/// Look up `filename` in the directory identified by `directory_inode_number`.
///
/// Returns `Ok(Some(inode_number))` if the entry is found, `Ok(None)` if not.
pub fn volume_lookup_directory_entry<D: BlockDevice>(
    volume: &BafsVolume<D>,
    directory_inode_number: u64,
    filename: &str,
) -> Result<Option<u64>, BafsError> {
    // Verify the inode is a directory.
    let dir_inode = volume_read_inode(volume, directory_inode_number)?;
    if !dir_inode.is_directory() {
        return Err(BafsError::NotADirectory);
    }

    match directory_lookup_entry(
        &volume.device,
        &volume.dirty_block_cache,
        volume.superblock.inode_tree_root_block,
        directory_inode_number,
        filename,
    )? {
        Some(entry) => Ok(Some(entry.child_inode_number)),
        None => Ok(None),
    }
}

/// Return the `index`-th directory entry from `directory_inode_number`.
///
/// Returns `Ok(Some((filename, inode_number, child_type)))` or `Ok(None)` if
/// `index` is past the last entry.  Used by the kernel's `readdir` syscall.
pub fn volume_read_directory_entry_at_index<D: BlockDevice>(
    volume: &BafsVolume<D>,
    directory_inode_number: u64,
    index: usize,
) -> Result<Option<BafsDirEntry>, BafsError> {
    directory_iterate(
        &volume.device,
        &volume.dirty_block_cache,
        volume.superblock.inode_tree_root_block,
        directory_inode_number,
        index,
    )
}

/// Remove the entry named `filename` from directory `directory_inode_number`.
///
/// This decrements the child inode's `hard_link_count`.  When that count
/// reaches zero the inode's data extents are freed.  For directories the
/// directory must be empty first.
pub fn volume_unlink_directory_entry<D: BlockDevice>(
    volume: &mut BafsVolume<D>,
    directory_inode_number: u64,
    filename: &str,
) -> Result<(), BafsError> {
    let generation = volume.next_transaction_id;

    // Find the child entry.
    let child_entry = directory_lookup_entry(
        &volume.device,
        &volume.dirty_block_cache,
        volume.superblock.inode_tree_root_block,
        directory_inode_number,
        filename,
    )?
    .ok_or(BafsError::NotFound)?;

    // Remove the directory entry.
    let mut unlink_freed_blocks: Vec<u64> = Vec::new();
    let new_inode_root = directory_unlink_entry(
        &volume.device,
        &mut volume.dirty_block_cache,
        volume.superblock.inode_tree_root_block,
        directory_inode_number,
        filename,
        generation,
        &mut volume.next_free_block,
        &mut unlink_freed_blocks,
    )?;
    volume.superblock.inode_tree_root_block = new_inode_root;

    // Decrement the child's hard_link_count.
    let mut child_inode =
        volume_read_inode(volume, child_entry.child_inode_number)?;
    child_inode.hard_link_count = child_inode.hard_link_count.saturating_sub(1);

    if child_inode.hard_link_count == 0 {
        // Collect all data extents for this inode.
        let min_key = BafsKey::new(child_entry.child_inode_number, ITEM_TYPE_EXTENT_DATA, 0);
        let max_key = BafsKey::new(child_entry.child_inode_number, ITEM_TYPE_EXTENT_DATA, u64::MAX);
        let extent_items = iterate_tree_range(
            &volume.device,
            &volume.dirty_block_cache,
            volume.superblock.inode_tree_root_block,
            min_key,
            max_key,
        )?;

        let mut inode_root = volume.superblock.inode_tree_root_block;
        let mut free_extent_root = volume.superblock.free_extent_tree_root_block;

        for extent_item in &extent_items {
            let extent = deserialise_extent_data_from_bytes(&extent_item.value);

            // Return the data blocks to the free pool.
            free_extent_root = free_blocks(
                &volume.device,
                &mut volume.dirty_block_cache,
                free_extent_root,
                extent.first_disk_block,
                extent.block_count,
                generation,
                &mut volume.next_free_block,
                &mut unlink_freed_blocks,
            )?;

            // Remove the extent record from the inode tree.
            inode_root = delete_from_tree(
                &volume.device,
                &mut volume.dirty_block_cache,
                inode_root,
                extent_item.key,
                generation,
                &mut volume.next_free_block,
                &mut unlink_freed_blocks,
            )?;
        }

        // Remove the inode itself from the tree.
        inode_root = delete_from_tree(
            &volume.device,
            &mut volume.dirty_block_cache,
            inode_root,
            BafsKey::new(child_entry.child_inode_number, ITEM_TYPE_INODE, 0),
            generation,
            &mut volume.next_free_block,
            &mut unlink_freed_blocks,
        )?;

        volume.superblock.inode_tree_root_block = inode_root;
        volume.superblock.free_extent_tree_root_block = free_extent_root;
        volume.superblock.free_block_count = volume.superblock.free_block_count
            .saturating_add(child_inode.allocated_block_count);
    } else {
        volume_write_inode(volume, &child_inode)?;
    }

    return_freed_cow_blocks_to_free_extent_tree(volume, unlink_freed_blocks)?;
    Ok(())
}
