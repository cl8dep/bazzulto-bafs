//! BAFS write-ahead journal (WAL): transaction lifecycle and crash recovery.
//!
//! The journal guarantees that every multi-step write operation either
//! completes fully or leaves the filesystem in its previous consistent state.
//! It is a sequential write-ahead log stored in the dedicated journal area
//! (blocks 3..259 by default).
//!
//! # Transaction lifecycle
//!
//! ```text
//! 1. begin_transaction()
//!    └── Returns a BafsTransaction with a new transaction_id.
//!        All subsequent writes go into the volume's dirty-block cache.
//!
//! 2. commit_transaction()
//!    ├── (a) Serialise all dirty blocks into a commit record.
//!    ├── (b) Write the commit record to the journal area on disk.
//!    ├── (c) Write each dirty block to its final disk location.
//!    ├── (d) Write the updated superblock (backup first, then primary).
//!    └── (e) Mark the dirty-block cache as clean.
//!
//! 3. recover_journal_on_mount()
//!    ├── Read the newest commit record from the journal area.
//!    ├── If its transaction_id > superblock.last_committed_transaction_id:
//!    │   the previous mount crashed between steps (b) and (d).
//!    │   Replay: write all blocks from the commit record to their final
//!    │   locations, then write the superblock.
//!    └── If no newer record: filesystem is already consistent; do nothing.
//! ```
//!
//! # Commit record on-disk layout
//!
//! The commit record is written starting at `journal_start_block`.  For v1 the
//! journal always starts from the beginning (no circularity); only one commit
//! record at a time is needed because each commit updates the superblock before
//! the next transaction begins.
//!
//! ```text
//! Offset  Size  Field
//! ──────  ────  ─────────────────────────────────────────────────────
//!      0     8  record_magic  (b"BAFSJRNL")
//!      8     8  transaction_id  (u64 LE)
//!     16     4  dirty_block_count  (u32 LE)
//!     20     8  block_address_0  (u64 LE, repeated dirty_block_count times)
//!     ..        …
//!     20 + dirty_block_count*8 + 0*BLOCK_SIZE:  block_data_0  (4096 bytes each)
//!     …
//!     last 4 bytes: commit_checksum  (CRC32C of everything before)
//! ```
//!
//! # Impact on the rest of the system
//!
//! - `volume.rs` calls `begin_transaction` before any modifying operation.
//! - `volume.rs` calls `commit_transaction` on explicit flush or unmount.
//! - `volume.rs` calls `recover_journal_on_mount` immediately after reading
//!   the superblock, before any other I/O.

#[cfg(feature = "kernel")]
use alloc::{vec, vec::Vec, collections::BTreeMap};
#[cfg(not(feature = "kernel"))]
use std::{vec, vec::Vec, collections::BTreeMap};

use crate::block_device::BlockDevice;
use crate::checksum::compute_crc32c;
use crate::error::BafsError;
use crate::superblock::{
    BafsSuperblock, BAFS_DEFAULT_BLOCK_SIZE_BYTES, BAFS_SECTORS_PER_BLOCK,
    write_superblock_to_device,
};

// ─── Constants ────────────────────────────────────────────────────────────────

/// Eight-byte magic that identifies the start of a BAFS journal commit record.
const JOURNAL_COMMIT_RECORD_MAGIC: &[u8; 8] = b"BAFSJRNL";

/// Minimum size of a commit record in bytes (header only, zero dirty blocks).
///
/// Layout: magic(8) + transaction_id(8) + dirty_block_count(4) + checksum(4)
/// = 24 bytes.
const JOURNAL_COMMIT_RECORD_HEADER_SIZE_BYTES: usize = 24;

// ─── In-memory transaction ────────────────────────────────────────────────────

/// An in-progress write transaction.
///
/// Modifying operations accumulate dirty blocks in the volume's
/// `BTreeMap<block_address, block_data>` cache.  The `BafsTransaction` holds
/// only the transaction metadata; the dirty blocks themselves live in
/// `BafsVolume`.
pub struct BafsTransaction {
    /// Monotonically increasing identifier for this transaction.
    pub transaction_id: u64,
}

/// Begin a new write transaction.
///
/// The caller must pass the superblock's `last_committed_transaction_id + 1`
/// as the new ID.  After calling this function all writes go into the volume's
/// dirty-block cache until `commit_transaction` is called.
pub fn begin_transaction(next_transaction_id: u64) -> BafsTransaction {
    BafsTransaction {
        transaction_id: next_transaction_id,
    }
}

// ─── Commit ───────────────────────────────────────────────────────────────────

/// Commit a transaction: write dirty blocks to the journal, then to their
/// final locations, then update the superblock.
///
/// `dirty_cache` maps block address → 4 KiB block data.  After this function
/// returns successfully the cache must be cleared by the caller.
///
/// # Crash safety
///
/// Step (b) writes the commit record before the blocks reach their final
/// locations.  If the system crashes:
/// - After step (a) but before (b): no commit record on disk → recovery
///   ignores this transaction → filesystem is in the pre-transaction state.
/// - After step (b) but before (d): recovery replays the commit record →
///   filesystem reaches the post-transaction state.
/// - After step (d): superblock matches the committed state → no recovery
///   needed.
pub fn commit_transaction(
    device: &dyn BlockDevice,
    transaction: &BafsTransaction,
    dirty_cache: &BTreeMap<u64, Vec<u8>>,
    superblock: &mut BafsSuperblock,
) -> Result<(), BafsError> {
    if dirty_cache.is_empty() {
        // Nothing to commit — just bump the transaction ID.
        superblock.last_committed_transaction_id = transaction.transaction_id;
        write_superblock_to_device(device, superblock)?;
        return Ok(());
    }

    // ── (a) Build the commit record in memory ──────────────────────────────

    let block_addresses: Vec<u64> = dirty_cache.keys().copied().collect();
    let dirty_block_count = block_addresses.len() as u32;
    let block_size = BAFS_DEFAULT_BLOCK_SIZE_BYTES as usize;

    // Record size: header(24) + addresses(8 * count) + block_data(4096 * count)
    // + trailing checksum(4).
    let record_body_size = JOURNAL_COMMIT_RECORD_HEADER_SIZE_BYTES
        + 8 * dirty_block_count as usize
        + block_size * dirty_block_count as usize;
    let total_record_size = record_body_size + 4; // +4 for trailing checksum

    let mut record_buffer: Vec<u8> = vec![0u8; total_record_size];

    // Header.
    record_buffer[0..8].copy_from_slice(JOURNAL_COMMIT_RECORD_MAGIC);
    record_buffer[8..16].copy_from_slice(&transaction.transaction_id.to_le_bytes());
    record_buffer[16..20].copy_from_slice(&dirty_block_count.to_le_bytes());

    // Block address list.
    let mut write_cursor = 20usize;
    for &address in &block_addresses {
        record_buffer[write_cursor..write_cursor + 8]
            .copy_from_slice(&address.to_le_bytes());
        write_cursor += 8;
    }

    // Block data.
    for &address in &block_addresses {
        let block_data = dirty_cache
            .get(&address)
            .expect("address came from dirty_cache keys");
        record_buffer[write_cursor..write_cursor + block_size]
            .copy_from_slice(block_data);
        write_cursor += block_size;
    }

    // Trailing CRC32C checksum (covers everything before it).
    let checksum = compute_crc32c(&record_buffer[..record_body_size]);
    record_buffer[record_body_size..record_body_size + 4]
        .copy_from_slice(&checksum.to_le_bytes());

    // ── (b) Write the commit record to the journal area ────────────────────

    write_bytes_to_journal(device, superblock, &record_buffer)?;

    // ── (c) Write each dirty block to its final disk location ──────────────

    for (&block_address, block_data) in dirty_cache {
        let start_lba = block_address * BAFS_SECTORS_PER_BLOCK as u64;
        if !device.write_sectors(start_lba, BAFS_SECTORS_PER_BLOCK, block_data) {
            return Err(BafsError::InputOutputError);
        }
    }

    // ── (d) Write the updated superblock ───────────────────────────────────

    superblock.last_committed_transaction_id = transaction.transaction_id;
    // Update free_block_count in the superblock from the dirty cache size
    // (approximate; the B-tree is authoritative).
    write_superblock_to_device(device, superblock)?;

    Ok(())
}

/// Write `data` to the journal area starting at `journal_start_block`.
///
/// The data may span multiple blocks.  Blocks beyond the journal area are an
/// error (for v1 the record is always small enough to fit).
fn write_bytes_to_journal(
    device: &dyn BlockDevice,
    superblock: &BafsSuperblock,
    data: &[u8],
) -> Result<(), BafsError> {
    let block_size = BAFS_DEFAULT_BLOCK_SIZE_BYTES as usize;
    let journal_capacity_bytes =
        superblock.journal_size_in_blocks as usize * block_size;

    if data.len() > journal_capacity_bytes {
        // The transaction is too large to fit in the journal.  For v1 this
        // should never happen with the conservative 256-block journal.
        return Err(BafsError::OutOfSpace);
    }

    // Write the data in block-sized chunks.
    let mut bytes_written = 0usize;
    let mut current_block = superblock.journal_start_block;

    while bytes_written < data.len() {
        let chunk_size = (data.len() - bytes_written).min(block_size);
        let mut block_buffer = vec![0u8; block_size];
        block_buffer[..chunk_size]
            .copy_from_slice(&data[bytes_written..bytes_written + chunk_size]);

        let start_lba = current_block * BAFS_SECTORS_PER_BLOCK as u64;
        if !device.write_sectors(start_lba, BAFS_SECTORS_PER_BLOCK, &block_buffer) {
            return Err(BafsError::InputOutputError);
        }

        bytes_written += chunk_size;
        current_block += 1;
    }
    Ok(())
}

// ─── Crash recovery ───────────────────────────────────────────────────────────

/// Check the journal for a commit record newer than the superblock and, if
/// found, replay it to bring the filesystem to a consistent state.
///
/// This function is called by `volume.rs` immediately after reading the
/// superblock on mount.  If recovery is performed, the superblock is updated
/// in place and re-written to disk.
///
/// Returns `Ok(true)` if a recovery replay was performed, `Ok(false)` if the
/// filesystem was already consistent.
pub fn recover_journal_on_mount(
    device: &dyn BlockDevice,
    superblock: &mut BafsSuperblock,
) -> Result<bool, BafsError> {
    let block_size = BAFS_DEFAULT_BLOCK_SIZE_BYTES as usize;
    let journal_capacity_bytes =
        superblock.journal_size_in_blocks as usize * block_size;

    // Read the entire journal area into memory.
    let mut journal_buffer: Vec<u8> = vec![0u8; journal_capacity_bytes];
    {
        let start_lba =
            superblock.journal_start_block * BAFS_SECTORS_PER_BLOCK as u64;
        let sector_count = (superblock.journal_size_in_blocks * BAFS_SECTORS_PER_BLOCK as u64) as u32;
        if !device.read_sectors(start_lba, sector_count, &mut journal_buffer) {
            return Err(BafsError::InputOutputError);
        }
    }

    // Check for the magic number.
    if journal_buffer.len() < JOURNAL_COMMIT_RECORD_HEADER_SIZE_BYTES {
        return Ok(false);
    }
    if &journal_buffer[0..8] != JOURNAL_COMMIT_RECORD_MAGIC {
        return Ok(false);
    }

    // Parse the transaction ID.
    let journal_transaction_id =
        u64::from_le_bytes(journal_buffer[8..16].try_into().unwrap());

    // If the journal's transaction is not newer than the superblock, the
    // filesystem is already consistent.
    if journal_transaction_id <= superblock.last_committed_transaction_id {
        return Ok(false);
    }

    // Parse the dirty block count.
    let dirty_block_count =
        u32::from_le_bytes(journal_buffer[16..20].try_into().unwrap()) as usize;

    // Compute expected record size and validate the trailing checksum.
    let record_body_size = JOURNAL_COMMIT_RECORD_HEADER_SIZE_BYTES
        + 8 * dirty_block_count
        + block_size * dirty_block_count;
    let total_record_size = record_body_size + 4;

    if journal_buffer.len() < total_record_size {
        // Journal record is truncated — the write never completed.
        return Ok(false);
    }

    let stored_checksum = u32::from_le_bytes(
        journal_buffer[record_body_size..record_body_size + 4]
            .try_into()
            .unwrap(),
    );
    let computed_checksum = compute_crc32c(&journal_buffer[..record_body_size]);
    if stored_checksum != computed_checksum {
        // Checksum mismatch — the commit record is incomplete or corrupt.
        // Leave the filesystem in its pre-transaction state.
        return Ok(false);
    }

    // Parse block addresses.
    let mut addresses: Vec<u64> = Vec::with_capacity(dirty_block_count);
    let mut read_cursor = 20usize;
    for _ in 0..dirty_block_count {
        let address =
            u64::from_le_bytes(journal_buffer[read_cursor..read_cursor + 8].try_into().unwrap());
        addresses.push(address);
        read_cursor += 8;
    }

    // Replay: write each block to its final location.
    for &address in &addresses {
        let block_data = &journal_buffer[read_cursor..read_cursor + block_size];
        let start_lba = address * BAFS_SECTORS_PER_BLOCK as u64;
        if !device.write_sectors(start_lba, BAFS_SECTORS_PER_BLOCK, block_data) {
            return Err(BafsError::InputOutputError);
        }
        read_cursor += block_size;
    }

    // Update the superblock to reflect the replayed transaction.
    superblock.last_committed_transaction_id = journal_transaction_id;
    write_superblock_to_device(device, superblock)?;

    Ok(true)
}
