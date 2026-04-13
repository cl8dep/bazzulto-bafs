//! BAFS checksum tree: per-data-block CRC32C storage and verification.
//!
//! While metadata blocks (B-tree nodes, superblock) carry their CRC32C
//! checksum inline as the first four bytes, data blocks have their checksums
//! stored in a dedicated B-tree so that the data itself is not disturbed.
//!
//! The checksum B-tree root is stored in `superblock.checksum_tree_root_block`.
//! Each entry uses the key:
//!
//!   `(BAFS_CHECKSUM_OBJECT_ID, ITEM_TYPE_CHECKSUM_ENTRY, block_address)`
//!
//! and the value is a `BafsChecksumItem` (12 bytes) containing the block
//! address and its CRC32C checksum.
//!
//! # Why store checksums in a separate tree?
//!
//! Storing checksums separately from data allows:
//! - Detecting silent data corruption without reading the entire block.
//! - Efficient bulk scrubbing: iterate the checksum tree to find all blocks
//!   that need verification.
//! - Sharing extents (v2+ snapshots/clones) without duplicating checksums.
//!
//! # Impact on the rest of the system
//!
//! - `volume.rs` calls `store_data_block_checksum` after every data write.
//! - `volume.rs` calls `verify_data_block_checksum` before returning data to
//!   a reader.
//! - The checksum tree root may change after every write (CoW); `volume.rs`
//!   stores the updated root in the superblock.

#[cfg(feature = "kernel")]
use alloc::{vec, vec::Vec, collections::BTreeMap};
#[cfg(not(feature = "kernel"))]
use std::{vec, vec::Vec, collections::BTreeMap};

use crate::block_device::BlockDevice;
use crate::btree::{insert_into_tree, lookup_in_tree, BafsKey, ITEM_TYPE_CHECKSUM_ENTRY};
use crate::checksum::{compute_crc32c, verify_crc32c};
use crate::error::BafsError;
use crate::superblock::BAFS_CHECKSUM_OBJECT_ID;

// ─── On-disk structure ────────────────────────────────────────────────────────

/// One entry in the checksum B-tree: the CRC32C of a single 4 KiB data block.
///
/// Stored as the leaf value for the key
/// `(BAFS_CHECKSUM_OBJECT_ID, ITEM_TYPE_CHECKSUM_ENTRY, block_address)`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BafsChecksumItem {
    /// Block address of the data block whose checksum is stored here.
    pub block_address: u64,

    /// CRC32C of the 4 KiB data block at `block_address`.
    pub crc32c_checksum: u32,

    /// Reserved, zero-padded (brings the struct to 16 bytes for alignment).
    pub reserved: u32,
}

const _CHECKSUM_ITEM_SIZE_ASSERTION: () =
    assert!(core::mem::size_of::<BafsChecksumItem>() == 16);

/// Serialise a `BafsChecksumItem` into 16 bytes.
fn serialise_checksum_item_to_bytes(item: &BafsChecksumItem) -> [u8; 16] {
    let mut output = [0u8; 16];
    output[0..8].copy_from_slice(&item.block_address.to_le_bytes());
    output[8..12].copy_from_slice(&item.crc32c_checksum.to_le_bytes());
    output[12..16].copy_from_slice(&item.reserved.to_le_bytes());
    output
}

/// Deserialise 16 bytes into a `BafsChecksumItem`.
fn deserialise_checksum_item_from_bytes(bytes: &[u8]) -> BafsChecksumItem {
    debug_assert!(bytes.len() >= 16);
    let block_address = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let crc32c_checksum = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let reserved = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    BafsChecksumItem { block_address, crc32c_checksum, reserved }
}

// ─── Public operations ────────────────────────────────────────────────────────

/// Compute the CRC32C of `block_data` and store it in the checksum B-tree.
///
/// This must be called after every data block write.  Returns the new
/// checksum-tree root block address (may change due to CoW).
pub fn store_data_block_checksum(
    device: &dyn BlockDevice,
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    checksum_tree_root_block: u64,
    data_block_address: u64,
    block_data: &[u8],
    generation: u64,
    next_free_block: &mut u64,
) -> Result<u64, BafsError> {
    let crc32c_checksum = compute_crc32c(block_data);
    let checksum_item = BafsChecksumItem {
        block_address: data_block_address,
        crc32c_checksum,
        reserved: 0,
    };

    let key = BafsKey::new(
        BAFS_CHECKSUM_OBJECT_ID,
        ITEM_TYPE_CHECKSUM_ENTRY,
        data_block_address,
    );
    let value = serialise_checksum_item_to_bytes(&checksum_item).to_vec();

    insert_into_tree(
        device,
        dirty_cache,
        checksum_tree_root_block,
        key,
        value,
        generation,
        next_free_block,
    )
}

/// Verify that `block_data` matches the CRC32C stored in the checksum tree.
///
/// Returns `Ok(())` if the checksums match, `Err(BafsError::InvalidChecksum)`
/// if they differ.  Returns `Ok(())` (not an error) if no checksum entry
/// exists yet for the block (e.g. freshly allocated but not yet written).
pub fn verify_data_block_checksum(
    device: &dyn BlockDevice,
    dirty_cache: &BTreeMap<u64, Vec<u8>>,
    checksum_tree_root_block: u64,
    data_block_address: u64,
    block_data: &[u8],
) -> Result<(), BafsError> {
    let key = BafsKey::new(
        BAFS_CHECKSUM_OBJECT_ID,
        ITEM_TYPE_CHECKSUM_ENTRY,
        data_block_address,
    );

    match lookup_in_tree(device, dirty_cache, checksum_tree_root_block, key)? {
        None => {
            // No checksum entry: block has not been written yet or is freshly
            // allocated.  Trust it.
            Ok(())
        }
        Some(value_bytes) => {
            if value_bytes.len() < 16 {
                return Err(BafsError::CorruptedStructure);
            }
            let stored_item = deserialise_checksum_item_from_bytes(&value_bytes);
            if verify_crc32c(block_data, stored_item.crc32c_checksum) {
                Ok(())
            } else {
                Err(BafsError::InvalidChecksum {
                    block_address: data_block_address,
                })
            }
        }
    }
}
