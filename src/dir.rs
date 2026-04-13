//! BAFS directory operations: lookup, create, unlink, and iteration.
//!
//! Directory entries are stored inside the main inode B-tree, not in a
//! separate structure.  Each entry uses the key:
//!
//!   `(parent_inode_number, ITEM_TYPE_DIRECTORY_ENTRY, xxh64(filename, seed=0))`
//!
//! and the value is a serialised `BafsDirEntry` struct that contains the child
//! inode number, child type, and the filename itself.
//!
//! # Why xxHash-64 as the key offset?
//!
//! Hashing the filename gives O(log n) lookup without requiring the B-tree to
//! perform string comparisons during traversal.  xxHash-64 is used because it
//! is fast, deterministic across all platforms, and produces well-distributed
//! 64-bit values that keep the B-tree balanced.
//!
//! # Hash collision handling
//!
//! Two filenames may produce the same xxh64 hash.  The B-tree stores both
//! entries at the same key offset (the second insert does NOT overwrite the
//! first because we embed the filename in the value and use a linear scan
//! when a hash collision is detected during lookup).
//!
//! In v1, the probability of a collision in any real directory is negligible
//! and the linear scan is O(collision_count), which is effectively O(1).
//!
//! # On-disk layout of BafsDirEntry
//!
//! ```text
//! Offset  Size  Field
//! ──────  ────  ─────────────────────────────────────────────────────
//!      0     8  child_inode_number
//!      8     1  child_type  (4=directory, 8=regular file, 10=symlink, …)
//!      9     1  name_length_bytes  (1–255)
//!     10     2  reserved  (zero-padded)
//!     12  1–255 filename_bytes  (UTF-8, NOT null-terminated)
//! ```
//!
//! The total size is `12 + name_length_bytes`, stored as the leaf item value.
//!
//! # Impact on the rest of the system
//!
//! - `volume.rs` calls `directory_lookup_entry` when opening a file by path.
//! - `volume.rs` calls `directory_create_entry` when creating a file or dir.
//! - `volume.rs` calls `directory_unlink_entry` when deleting a file or dir.
//! - `volume.rs` calls `directory_iterate` to implement `readdir`.
//! - `kernel.rs` exposes these via the `Inode::lookup`, `Inode::create`,
//!   `Inode::unlink`, and `Inode::readdir` trait methods.

#[cfg(feature = "kernel")]
use alloc::{string::String, vec, vec::Vec, collections::BTreeMap};
#[cfg(not(feature = "kernel"))]
use std::{string::String, vec, vec::Vec, collections::BTreeMap};

use crate::block_device::BlockDevice;
use crate::btree::{
    delete_from_tree, insert_into_tree, iterate_tree_range, lookup_in_tree,
    BafsKey, BafsLeafItem, ITEM_TYPE_DIRECTORY_ENTRY,
};
use crate::error::BafsError;

// ─── Child type constants (POSIX d_type values) ──────────────────────────────

/// `d_type` value for an unknown file type.
pub const CHILD_TYPE_UNKNOWN: u8 = 0;

/// `d_type` value for a named pipe (FIFO).
pub const CHILD_TYPE_FIFO: u8 = 1;

/// `d_type` value for a character device.
pub const CHILD_TYPE_CHAR_DEVICE: u8 = 2;

/// `d_type` value for a directory.
pub const CHILD_TYPE_DIRECTORY: u8 = 4;

/// `d_type` value for a block device.
pub const CHILD_TYPE_BLOCK_DEVICE: u8 = 6;

/// `d_type` value for a regular file.
pub const CHILD_TYPE_REGULAR_FILE: u8 = 8;

/// `d_type` value for a symbolic link.
pub const CHILD_TYPE_SYMLINK: u8 = 10;

/// `d_type` value for a socket.
pub const CHILD_TYPE_SOCKET: u8 = 12;

// ─── xxHash-64 implementation ─────────────────────────────────────────────────
//
// This is a complete, standalone implementation of the xxHash-64 algorithm
// (https://xxhash.com) with seed 0.  It uses only fixed-width integer
// arithmetic and bit operations, so it compiles in no_std with no external
// dependencies.
//
// xxHash-64 primes (from the specification):
const XXH64_PRIME_1: u64 = 0x9E3779B185EBCA87;
const XXH64_PRIME_2: u64 = 0xC2B2AE3D27D4EB4F;
const XXH64_PRIME_3: u64 = 0x165667B19E3779F9;
const XXH64_PRIME_4: u64 = 0x85EBCA77C2B2AE63;
const XXH64_PRIME_5: u64 = 0x27D4EB2F165667C5;

/// Rotate `value` left by `rotation_amount` bits.
#[inline(always)]
fn rotate_left_64(value: u64, rotation_amount: u32) -> u64 {
    value.rotate_left(rotation_amount)
}

/// Process a single 64-bit lane in the xxHash-64 accumulation round.
#[inline(always)]
fn xxh64_round(accumulator: u64, lane_value: u64) -> u64 {
    accumulator
        .wrapping_add(lane_value.wrapping_mul(XXH64_PRIME_2))
        .rotate_left(31)
        .wrapping_mul(XXH64_PRIME_1)
}

/// Merge a finished lane accumulator into the hash state.
#[inline(always)]
fn xxh64_merge_accumulator(hash_state: u64, accumulator: u64) -> u64 {
    let merged = hash_state ^ xxh64_round(0, accumulator);
    merged
        .wrapping_mul(XXH64_PRIME_1)
        .wrapping_add(XXH64_PRIME_4)
}

/// Compute xxHash-64 of `input_bytes` with seed 0.
///
/// This is used exclusively for directory entry key generation.  The result
/// is deterministic across all platforms and compiler versions because
/// xxHash-64 is defined entirely in terms of fixed-width arithmetic.
pub fn xxhash64_with_seed_zero(input_bytes: &[u8]) -> u64 {
    let input_length = input_bytes.len() as u64;
    let mut read_cursor = 0usize;
    let seed: u64 = 0;

    let mut hash_state: u64;

    if input_bytes.len() >= 32 {
        // ── Long path: process 32-byte stripes using 4 accumulators ──
        let mut accumulator_0 = seed
            .wrapping_add(XXH64_PRIME_1)
            .wrapping_add(XXH64_PRIME_2);
        let mut accumulator_1 = seed.wrapping_add(XXH64_PRIME_2);
        let mut accumulator_2 = seed;
        let mut accumulator_3 = seed.wrapping_sub(XXH64_PRIME_1);

        while read_cursor + 32 <= input_bytes.len() {
            accumulator_0 = xxh64_round(
                accumulator_0,
                u64::from_le_bytes(
                    input_bytes[read_cursor..read_cursor + 8].try_into().unwrap(),
                ),
            );
            read_cursor += 8;
            accumulator_1 = xxh64_round(
                accumulator_1,
                u64::from_le_bytes(
                    input_bytes[read_cursor..read_cursor + 8].try_into().unwrap(),
                ),
            );
            read_cursor += 8;
            accumulator_2 = xxh64_round(
                accumulator_2,
                u64::from_le_bytes(
                    input_bytes[read_cursor..read_cursor + 8].try_into().unwrap(),
                ),
            );
            read_cursor += 8;
            accumulator_3 = xxh64_round(
                accumulator_3,
                u64::from_le_bytes(
                    input_bytes[read_cursor..read_cursor + 8].try_into().unwrap(),
                ),
            );
            read_cursor += 8;
        }

        // Merge the four accumulators into the hash state.
        hash_state = rotate_left_64(accumulator_0, 1)
            .wrapping_add(rotate_left_64(accumulator_1, 7))
            .wrapping_add(rotate_left_64(accumulator_2, 12))
            .wrapping_add(rotate_left_64(accumulator_3, 18));

        hash_state = xxh64_merge_accumulator(hash_state, accumulator_0);
        hash_state = xxh64_merge_accumulator(hash_state, accumulator_1);
        hash_state = xxh64_merge_accumulator(hash_state, accumulator_2);
        hash_state = xxh64_merge_accumulator(hash_state, accumulator_3);
    } else {
        // ── Short path: input fits in a single stripe ──
        hash_state = seed.wrapping_add(XXH64_PRIME_5);
    }

    hash_state = hash_state.wrapping_add(input_length);

    // ── Consume remaining bytes ──

    // Process 8 bytes at a time.
    while read_cursor + 8 <= input_bytes.len() {
        let lane = u64::from_le_bytes(
            input_bytes[read_cursor..read_cursor + 8].try_into().unwrap(),
        );
        hash_state ^= xxh64_round(0, lane);
        hash_state = rotate_left_64(hash_state, 27)
            .wrapping_mul(XXH64_PRIME_1)
            .wrapping_add(XXH64_PRIME_4);
        read_cursor += 8;
    }

    // Process 4 bytes at a time.
    if read_cursor + 4 <= input_bytes.len() {
        let lane = u32::from_le_bytes(
            input_bytes[read_cursor..read_cursor + 4].try_into().unwrap(),
        ) as u64;
        hash_state ^= lane.wrapping_mul(XXH64_PRIME_1);
        hash_state = rotate_left_64(hash_state, 23)
            .wrapping_mul(XXH64_PRIME_2)
            .wrapping_add(XXH64_PRIME_3);
        read_cursor += 4;
    }

    // Process one byte at a time.
    while read_cursor < input_bytes.len() {
        let lane = input_bytes[read_cursor] as u64;
        hash_state ^= lane.wrapping_mul(XXH64_PRIME_5);
        hash_state = rotate_left_64(hash_state, 11).wrapping_mul(XXH64_PRIME_1);
        read_cursor += 1;
    }

    // ── Final mix (avalanche) ──
    hash_state ^= hash_state >> 33;
    hash_state = hash_state.wrapping_mul(XXH64_PRIME_2);
    hash_state ^= hash_state >> 29;
    hash_state = hash_state.wrapping_mul(XXH64_PRIME_3);
    hash_state ^= hash_state >> 32;

    hash_state
}

// ─── On-disk directory entry ──────────────────────────────────────────────────

/// In-memory representation of a directory entry.
///
/// On disk the `filename` bytes immediately follow the 12-byte fixed header.
/// We deserialise into this owned struct for ease of manipulation.
#[derive(Clone, Debug)]
pub struct BafsDirEntry {
    /// Inode number of the child filesystem object.
    pub child_inode_number: u64,

    /// POSIX `d_type` value indicating the kind of child object.
    /// See `CHILD_TYPE_*` constants.
    pub child_type: u8,

    /// Number of bytes in `filename`.  Always 1–255.
    pub name_length_bytes: u8,

    /// UTF-8 filename (not null-terminated).
    pub filename: String,
}

/// Serialise a `BafsDirEntry` into bytes for storage as a B-tree leaf value.
fn serialise_dir_entry_to_bytes(entry: &BafsDirEntry) -> Vec<u8> {
    let name_bytes = entry.filename.as_bytes();
    let total_size = 12 + name_bytes.len();
    let mut buffer = vec![0u8; total_size];
    buffer[0..8].copy_from_slice(&entry.child_inode_number.to_le_bytes());
    buffer[8] = entry.child_type;
    buffer[9] = entry.name_length_bytes;
    // bytes 10..12 are reserved, already zero
    buffer[12..12 + name_bytes.len()].copy_from_slice(name_bytes);
    buffer
}

/// Deserialise bytes from a B-tree leaf value into a `BafsDirEntry`.
fn deserialise_dir_entry_from_bytes(bytes: &[u8]) -> Result<BafsDirEntry, BafsError> {
    if bytes.len() < 12 {
        return Err(BafsError::CorruptedStructure);
    }
    let child_inode_number =
        u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let child_type = bytes[8];
    let name_length_bytes = bytes[9];
    let name_end = 12 + name_length_bytes as usize;
    if bytes.len() < name_end {
        return Err(BafsError::CorruptedStructure);
    }
    let filename = String::from_utf8_lossy(&bytes[12..name_end]).into_owned();
    Ok(BafsDirEntry {
        child_inode_number,
        child_type,
        name_length_bytes,
        filename,
    })
}

// ─── Key construction ─────────────────────────────────────────────────────────

/// Construct the B-tree key for a directory entry.
///
/// The offset field of the key is the xxHash-64 of the filename.  This gives
/// O(log n) lookup for any filename without string comparisons during tree
/// traversal.
fn make_directory_entry_key(parent_inode_number: u64, filename: &str) -> BafsKey {
    let filename_hash = xxhash64_with_seed_zero(filename.as_bytes());
    BafsKey::new(parent_inode_number, ITEM_TYPE_DIRECTORY_ENTRY, filename_hash)
}

// ─── Public operations ────────────────────────────────────────────────────────

/// Look up a filename in a directory's entry list.
///
/// Returns `Ok(Some(entry))` if the filename exists, `Ok(None)` if it does
/// not.  Handles hash collisions by comparing filenames byte-by-byte after
/// finding a matching hash key.
pub fn directory_lookup_entry(
    device: &dyn BlockDevice,
    dirty_cache: &BTreeMap<u64, Vec<u8>>,
    inode_tree_root_block: u64,
    parent_inode_number: u64,
    target_filename: &str,
) -> Result<Option<BafsDirEntry>, BafsError> {
    let target_hash = xxhash64_with_seed_zero(target_filename.as_bytes());

    // Scan all directory entries whose hash matches (handles collisions).
    // We use a range scan instead of a point lookup because multiple entries
    // with the same hash can coexist (different filenames, same hash offset).
    let search_key = BafsKey::new(parent_inode_number, ITEM_TYPE_DIRECTORY_ENTRY, target_hash);
    let min_key = search_key;
    let max_key = search_key;

    let matching_items = iterate_tree_range(
        device,
        dirty_cache,
        inode_tree_root_block,
        min_key,
        max_key,
    )?;

    for item in &matching_items {
        let entry = deserialise_dir_entry_from_bytes(&item.value)?;
        if entry.filename == target_filename {
            return Ok(Some(entry));
        }
    }
    Ok(None)
}

/// Add a new entry to a directory.
///
/// Returns `Err(BafsError::AlreadyExists)` if a file with `child_filename`
/// already exists in the directory.  Returns the new inode-tree root block.
pub fn directory_create_entry(
    device: &dyn BlockDevice,
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    inode_tree_root_block: u64,
    parent_inode_number: u64,
    child_filename: &str,
    child_inode_number: u64,
    child_type: u8,
    generation: u64,
    next_free_block: &mut u64,
) -> Result<u64, BafsError> {
    if child_filename.is_empty() || child_filename.len() > 255 {
        return Err(BafsError::InvalidArgument);
    }

    // Check for an existing entry with this name.
    if directory_lookup_entry(
        device,
        dirty_cache,
        inode_tree_root_block,
        parent_inode_number,
        child_filename,
    )?
    .is_some()
    {
        return Err(BafsError::AlreadyExists);
    }

    let entry = BafsDirEntry {
        child_inode_number,
        child_type,
        name_length_bytes: child_filename.len() as u8,
        filename: String::from(child_filename),
    };
    let key = make_directory_entry_key(parent_inode_number, child_filename);
    let value = serialise_dir_entry_to_bytes(&entry);

    insert_into_tree(
        device,
        dirty_cache,
        inode_tree_root_block,
        key,
        value,
        generation,
        next_free_block,
    )
}

/// Remove a directory entry by filename.
///
/// Returns `Err(BafsError::NotFound)` if no entry with `target_filename`
/// exists.  Returns the new inode-tree root block.
///
/// This function does NOT decrement the child inode's `hard_link_count` or
/// free its data; that is the responsibility of `volume.rs`.
pub fn directory_unlink_entry(
    device: &dyn BlockDevice,
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    inode_tree_root_block: u64,
    parent_inode_number: u64,
    target_filename: &str,
    generation: u64,
    next_free_block: &mut u64,
) -> Result<u64, BafsError> {
    let key = make_directory_entry_key(parent_inode_number, target_filename);

    // Verify the entry actually exists before attempting deletion.
    let existing_entry = directory_lookup_entry(
        device,
        dirty_cache,
        inode_tree_root_block,
        parent_inode_number,
        target_filename,
    )?;
    if existing_entry.is_none() {
        return Err(BafsError::NotFound);
    }

    delete_from_tree(
        device,
        dirty_cache,
        inode_tree_root_block,
        key,
        generation,
        next_free_block,
    )
}

/// Iterate directory entries by index.
///
/// Returns the entry at position `entry_index` in the directory (ordered by
/// their B-tree key, i.e. by xxh64 of the filename).  Returns `Ok(None)` when
/// `entry_index` is past the last entry.
///
/// Used by `kernel.rs` to implement `Inode::readdir`.
pub fn directory_iterate(
    device: &dyn BlockDevice,
    dirty_cache: &BTreeMap<u64, Vec<u8>>,
    inode_tree_root_block: u64,
    parent_inode_number: u64,
    entry_index: usize,
) -> Result<Option<BafsDirEntry>, BafsError> {
    let min_key =
        BafsKey::new(parent_inode_number, ITEM_TYPE_DIRECTORY_ENTRY, 0);
    let max_key =
        BafsKey::new(parent_inode_number, ITEM_TYPE_DIRECTORY_ENTRY, u64::MAX);

    let all_entries = iterate_tree_range(
        device,
        dirty_cache,
        inode_tree_root_block,
        min_key,
        max_key,
    )?;

    if entry_index >= all_entries.len() {
        return Ok(None);
    }

    let entry = deserialise_dir_entry_from_bytes(&all_entries[entry_index].value)?;
    Ok(Some(entry))
}

/// Return all directory entries for a given directory inode.
///
/// Collects every B-tree item with
/// `(parent_inode_number, ITEM_TYPE_DIRECTORY_ENTRY, *)` and deserialises them.
/// Used by `volume.rs` for operations that need the full child list.
pub fn directory_list_all_entries(
    device: &dyn BlockDevice,
    dirty_cache: &BTreeMap<u64, Vec<u8>>,
    inode_tree_root_block: u64,
    parent_inode_number: u64,
) -> Result<Vec<BafsDirEntry>, BafsError> {
    let min_key =
        BafsKey::new(parent_inode_number, ITEM_TYPE_DIRECTORY_ENTRY, 0);
    let max_key =
        BafsKey::new(parent_inode_number, ITEM_TYPE_DIRECTORY_ENTRY, u64::MAX);

    let all_items = iterate_tree_range(
        device,
        dirty_cache,
        inode_tree_root_block,
        min_key,
        max_key,
    )?;

    all_items
        .iter()
        .map(|item| deserialise_dir_entry_from_bytes(&item.value))
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The xxHash-64 reference value for the empty string with seed 0 is
    /// 0xEF46DB3751D8E999, per the official test vectors.
    #[test]
    fn xxhash64_empty_string_matches_reference_vector() {
        assert_eq!(xxhash64_with_seed_zero(b""), 0xEF46DB3751D8E999);
    }
}
