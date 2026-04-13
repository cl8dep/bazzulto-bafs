//! BAFS B-tree: on-disk node layout, lookup, insert, and delete.
//!
//! All persistent data structures in BAFS — the inode tree, the free-extent
//! tree, the checksum tree, and per-directory entry lists — are stored in a
//! single unified B-tree format.  A different root block address distinguishes
//! each tree; the key discriminates what kind of item is stored at each
//! position.
//!
//! # Node layout (one node = one 4 KiB block)
//!
//! ```text
//! Offset   Size  Field
//! ──────   ────  ────────────────────────────────────────────────────
//!      0      4  node_checksum  (CRC32C of bytes 4..block_size)
//!      4      1  level  (0 = leaf, 1+ = internal)
//!      5      1  flags  (reserved, zero on disk)
//!      6      2  item_count
//!      8      8  self_block_address
//!     16      8  generation  (transaction_id when last written)
//!     24      ?  items (sorted by key)
//! ```
//!
//! **Leaf nodes**: items are stored as a sorted array of `BafsLeafItemHeader`
//! entries (25 bytes each), followed by value data packed backward from the
//! end of the block.  `data_offset` in each header is an absolute byte offset
//! from the start of the block.
//!
//! **Internal nodes**: items are stored as a sorted array of
//! `BafsInternalItem` entries (33 bytes each).  Each entry holds a separator
//! key and a pointer to the child block whose subtree contains all keys ≥ that
//! separator key (and < the next separator).
//!
//! # Copy-on-Write semantics
//!
//! Every modification to the tree creates new blocks rather than overwriting
//! existing ones.  The modified leaf is written to a newly allocated block;
//! each ancestor on the path from root to leaf is likewise cloned and updated
//! with the new child pointer.  The caller receives the new root block address
//! and must update the superblock accordingly before the transaction commits.
//!
//! All new blocks go into the volume's in-memory dirty-block cache
//! (`BTreeMap<block_address, block_data>`) and are only written to disk when
//! the transaction commits.  Reads check the dirty cache first so that
//! uncommitted writes are immediately visible within the same transaction.
//!
//! # Impact on the rest of the system
//!
//! - `inode.rs` uses the inode B-tree (root = `superblock.inode_tree_root_block`).
//! - `extent.rs` uses the free-extent B-tree.
//! - `checksum_tree.rs` uses the checksum B-tree.
//! - `dir.rs` inserts and looks up directory entries using the inode B-tree
//!   with `item_type = ITEM_TYPE_DIRECTORY_ENTRY`.
//! - `volume.rs` owns the dirty-block cache and passes it to all tree
//!   operations.

#[cfg(feature = "kernel")]
use alloc::{vec, vec::Vec, collections::BTreeMap};
#[cfg(not(feature = "kernel"))]
use std::{vec, vec::Vec, collections::BTreeMap};

use crate::block_device::BlockDevice;
use crate::checksum::{compute_crc32c, verify_crc32c};
use crate::error::BafsError;
use crate::superblock::{
    BAFS_DEFAULT_BLOCK_SIZE_BYTES, BAFS_SECTORS_PER_BLOCK,
};

// ─── Item type constants ──────────────────────────────────────────────────────

/// Item type for inode metadata (BafsInode struct stored as the value).
pub const ITEM_TYPE_INODE: u8 = 0x01;

/// Item type for directory entries (BafsDirEntry struct stored as the value).
pub const ITEM_TYPE_DIRECTORY_ENTRY: u8 = 0x02;

/// Item type for file extent data (BafsExtentData struct stored as the value).
pub const ITEM_TYPE_EXTENT_DATA: u8 = 0x03;

/// Item type for checksum entries (BafsChecksumItem struct stored as the value).
pub const ITEM_TYPE_CHECKSUM_ENTRY: u8 = 0x05;

// ─── Key ─────────────────────────────────────────────────────────────────────

/// The three-part sort key used by every B-tree in BAFS.
///
/// Keys are compared lexicographically: first by `object_id`, then by
/// `item_type`, then by `offset`.  This ordering groups all items belonging to
/// the same object (inode, directory, etc.) into a contiguous range, making
/// range scans efficient.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct BafsKey {
    /// The primary grouping identifier.  For inode items and directory entries
    /// this is the inode number.  For extent items it is the owner inode number.
    /// For checksum and free-extent items it is a reserved sentinel value.
    pub object_id: u64,

    /// Discriminates the kind of item stored at this key within the object.
    /// See `ITEM_TYPE_*` constants.
    pub item_type: u8,

    /// Secondary sort key.  Interpretation depends on `item_type`:
    /// - Inode items: always 0.
    /// - Directory entries: xxHash-64 of the filename (seed 0).
    /// - Extent data: byte offset of the extent within the file.
    /// - Checksum entries: block address of the data block.
    /// - Free-extent entries: block address of the free extent.
    pub offset: u64,
}

impl BafsKey {
    /// Create a new key.
    pub const fn new(object_id: u64, item_type: u8, offset: u64) -> Self {
        BafsKey { object_id, item_type, offset }
    }
}

// ─── Node header ──────────────────────────────────────────────────────────────

/// Size of the node header in bytes.
const NODE_HEADER_SIZE_BYTES: usize = 24;

/// Size of one leaf item header: key(17) + data_offset(4) + data_size(4).
const LEAF_ITEM_HEADER_SIZE_BYTES: usize = 25;

/// Compute the total serialised byte size of a leaf node with the given items.
///
/// The on-disk layout packs item headers forward from byte 24 and value data
/// backward from byte 4096.  This returns the sum of both regions plus the
/// node header, which must not exceed the block size.
fn leaf_serialised_size(leaf_items: &[BafsLeafItem]) -> usize {
    let headers_end = NODE_HEADER_SIZE_BYTES
        + leaf_items.len() * LEAF_ITEM_HEADER_SIZE_BYTES;
    let total_value_bytes: usize = leaf_items.iter().map(|item| item.value.len()).sum();
    headers_end + total_value_bytes
}

/// Returns true if the leaf node's items would overflow a 4 KiB block when
/// serialised.
fn leaf_node_would_overflow(leaf_items: &[BafsLeafItem]) -> bool {
    leaf_serialised_size(leaf_items) > BAFS_DEFAULT_BLOCK_SIZE_BYTES as usize
}

/// Maximum items per internal node.  Each internal item is 33 bytes; 90 items
/// fit comfortably in the remaining space after the 24-byte header.
const INTERNAL_MAX_ITEMS: usize = 90;

// ─── In-memory node representation ───────────────────────────────────────────

/// An in-memory B-tree node, loaded from (or about to be written to) a block.
///
/// We parse the raw on-disk bytes into this struct for easier manipulation,
/// then serialise back before writing.
#[derive(Clone, Debug)]
pub struct BafsTreeNode {
    /// CRC32C of bytes 4..block_size, stored at offset 0.
    pub node_checksum: u32,

    /// 0 = leaf (contains key-value items), ≥1 = internal (contains child pointers).
    pub level: u8,

    /// Reserved flags byte (zero on disk).
    pub flags: u8,

    /// Number of items (leaf items or internal items) in this node.
    pub item_count: u16,

    /// Block address of this node (self-pointer for integrity checking).
    pub self_block_address: u64,

    /// Transaction ID when this node was last written.
    pub generation: u64,

    /// Leaf items (populated only when `level == 0`).
    pub leaf_items: Vec<BafsLeafItem>,

    /// Internal items (populated only when `level > 0`).
    pub internal_items: Vec<BafsInternalItem>,
}

/// A key-value pair stored in a leaf node.
#[derive(Clone, Debug)]
pub struct BafsLeafItem {
    /// Sort key.
    pub key: BafsKey,
    /// Value bytes.  Variable length.
    pub value: Vec<u8>,
}

/// A separator key + child pointer stored in an internal node.
#[derive(Clone, Debug)]
pub struct BafsInternalItem {
    /// The smallest key reachable through `child_block`.
    pub key: BafsKey,
    /// Block address of the child B-tree node.
    pub child_block: u64,
    /// Generation of the child block (used for CoW validation).
    pub child_generation: u64,
}

// ─── Serialisation ────────────────────────────────────────────────────────────

/// Serialise a key to 17 bytes (little-endian).
///
/// Layout: object_id(8) | item_type(1) | offset(8)
fn serialise_key(key: &BafsKey, destination: &mut [u8]) {
    debug_assert!(destination.len() >= 17);
    destination[0..8].copy_from_slice(&key.object_id.to_le_bytes());
    destination[8] = key.item_type;
    destination[9..17].copy_from_slice(&key.offset.to_le_bytes());
}

/// Deserialise a key from 17 bytes (little-endian).
fn deserialise_key(source: &[u8]) -> BafsKey {
    debug_assert!(source.len() >= 17);
    let object_id = u64::from_le_bytes(source[0..8].try_into().unwrap());
    let item_type = source[8];
    let offset = u64::from_le_bytes(source[9..17].try_into().unwrap());
    BafsKey { object_id, item_type, offset }
}

/// Serialise a `BafsTreeNode` into a 4 KiB byte buffer.
///
/// Layout:
/// - Bytes 0..4:   placeholder for checksum (filled in at the end)
/// - Byte 4:       level
/// - Byte 5:       flags
/// - Bytes 6..8:   item_count (u16 LE)
/// - Bytes 8..16:  self_block_address (u64 LE)
/// - Bytes 16..24: generation (u64 LE)
/// - Bytes 24..:   item data (format depends on level)
///
/// For **internal** nodes each item is 33 bytes:
///   key(17) | child_block(8) | child_generation(8)
///
/// For **leaf** nodes items are written in two parts:
///   - Item headers (25 bytes each) grow from offset 24 upward.
///   - Value data is packed from the end of the block downward.
///   - Each header's `data_offset` field (u32 LE at header[17..21]) holds the
///     absolute block offset where the value starts.
pub fn serialise_node_to_block(node: &BafsTreeNode) -> Vec<u8> {
    let block_size = BAFS_DEFAULT_BLOCK_SIZE_BYTES as usize;
    let mut buffer = vec![0u8; block_size];

    // ── Header (24 bytes) ──
    // Bytes 0..4: checksum placeholder (written last).
    buffer[4] = node.level;
    buffer[5] = node.flags;
    buffer[6..8].copy_from_slice(&node.item_count.to_le_bytes());
    buffer[8..16].copy_from_slice(&node.self_block_address.to_le_bytes());
    buffer[16..24].copy_from_slice(&node.generation.to_le_bytes());

    if node.level > 0 {
        // ── Internal node ──
        // Each item: key(17) + child_block(8) + child_generation(8) = 33 bytes.
        let mut write_cursor = NODE_HEADER_SIZE_BYTES;
        for item in &node.internal_items {
            serialise_key(&item.key, &mut buffer[write_cursor..write_cursor + 17]);
            buffer[write_cursor + 17..write_cursor + 25]
                .copy_from_slice(&item.child_block.to_le_bytes());
            buffer[write_cursor + 25..write_cursor + 33]
                .copy_from_slice(&item.child_generation.to_le_bytes());
            write_cursor += 33;
        }
    } else {
        // ── Leaf node ──
        // Item headers grow from offset 24 upward (25 bytes per header).
        // Value data is packed from the end of the block downward.
        let mut header_cursor = NODE_HEADER_SIZE_BYTES;
        let mut value_cursor = block_size; // points just past the last written value

        for item in &node.leaf_items {
            let value_size = item.value.len();
            // Carve space for this value from the end of the block.
            value_cursor -= value_size;
            let data_offset = value_cursor as u32;

            // Write value data.
            buffer[value_cursor..value_cursor + value_size]
                .copy_from_slice(&item.value);

            // Write item header: key(17) + data_offset(4) + data_size(4) = 25 bytes.
            serialise_key(&item.key, &mut buffer[header_cursor..header_cursor + 17]);
            buffer[header_cursor + 17..header_cursor + 21]
                .copy_from_slice(&data_offset.to_le_bytes());
            buffer[header_cursor + 21..header_cursor + 25]
                .copy_from_slice(&(value_size as u32).to_le_bytes());

            header_cursor += 25;
        }
    }

    // ── Checksum ──
    // Covers bytes 4..block_size (everything except the first 4 bytes).
    let checksum = compute_crc32c(&buffer[4..]);
    buffer[0..4].copy_from_slice(&checksum.to_le_bytes());

    buffer
}

/// Deserialise a 4 KiB block into a `BafsTreeNode`.
///
/// Returns `Err(BafsError::InvalidChecksum)` if the CRC32C does not match, or
/// `Err(BafsError::CorruptedStructure)` if the item layout is internally
/// inconsistent.
pub fn deserialise_node_from_block(
    block_data: &[u8],
    block_address: u64,
) -> Result<BafsTreeNode, BafsError> {
    let block_size = BAFS_DEFAULT_BLOCK_SIZE_BYTES as usize;
    debug_assert_eq!(block_data.len(), block_size);

    // ── Verify checksum ──
    // The stored checksum at bytes 0..4 covers bytes 4..block_size.
    let stored_checksum = u32::from_le_bytes(block_data[0..4].try_into().unwrap());
    if !verify_crc32c(&block_data[4..], stored_checksum) {
        return Err(BafsError::InvalidChecksum { block_address });
    }

    // ── Parse header ──
    let level = block_data[4];
    let flags = block_data[5];
    let item_count = u16::from_le_bytes(block_data[6..8].try_into().unwrap()) as usize;
    let self_block_address = u64::from_le_bytes(block_data[8..16].try_into().unwrap());
    let generation = u64::from_le_bytes(block_data[16..24].try_into().unwrap());

    let mut node = BafsTreeNode {
        node_checksum: stored_checksum,
        level,
        flags,
        item_count: item_count as u16,
        self_block_address,
        generation,
        leaf_items: Vec::new(),
        internal_items: Vec::new(),
    };

    if level > 0 {
        // ── Internal node ──
        // Each item is 33 bytes.  Bounds-check first.
        let required_bytes = NODE_HEADER_SIZE_BYTES + item_count * 33;
        if required_bytes > block_size {
            return Err(BafsError::CorruptedStructure);
        }
        let mut read_cursor = NODE_HEADER_SIZE_BYTES;
        for _ in 0..item_count {
            let key = deserialise_key(&block_data[read_cursor..]);
            let child_block = u64::from_le_bytes(
                block_data[read_cursor + 17..read_cursor + 25].try_into().unwrap(),
            );
            let child_generation = u64::from_le_bytes(
                block_data[read_cursor + 25..read_cursor + 33].try_into().unwrap(),
            );
            node.internal_items.push(BafsInternalItem {
                key,
                child_block,
                child_generation,
            });
            read_cursor += 33;
        }
    } else {
        // ── Leaf node ──
        // Headers are 25 bytes each, starting at offset 24.
        let required_header_bytes = NODE_HEADER_SIZE_BYTES + item_count * 25;
        if required_header_bytes > block_size {
            return Err(BafsError::CorruptedStructure);
        }
        let mut header_cursor = NODE_HEADER_SIZE_BYTES;
        for _ in 0..item_count {
            let key = deserialise_key(&block_data[header_cursor..]);
            let data_offset = u32::from_le_bytes(
                block_data[header_cursor + 17..header_cursor + 21].try_into().unwrap(),
            ) as usize;
            let data_size = u32::from_le_bytes(
                block_data[header_cursor + 21..header_cursor + 25].try_into().unwrap(),
            ) as usize;

            // Bounds-check the value region.
            if data_offset + data_size > block_size {
                return Err(BafsError::CorruptedStructure);
            }
            let value = block_data[data_offset..data_offset + data_size].to_vec();

            node.leaf_items.push(BafsLeafItem { key, value });
            header_cursor += 25;
        }
    }

    Ok(node)
}

// ─── Block I/O helpers ────────────────────────────────────────────────────────

/// Read a 4 KiB block from the dirty cache (if present) or from `device`.
pub fn read_block(
    device: &dyn BlockDevice,
    dirty_cache: &BTreeMap<u64, Vec<u8>>,
    block_address: u64,
) -> Result<Vec<u8>, BafsError> {
    // Check the dirty-block cache first so that uncommitted writes are visible.
    if let Some(cached_data) = dirty_cache.get(&block_address) {
        return Ok(cached_data.clone());
    }
    let mut buffer = vec![0u8; BAFS_DEFAULT_BLOCK_SIZE_BYTES as usize];
    let start_lba = block_address * BAFS_SECTORS_PER_BLOCK as u64;
    if !device.read_sectors(start_lba, BAFS_SECTORS_PER_BLOCK, &mut buffer) {
        return Err(BafsError::InputOutputError);
    }
    Ok(buffer)
}

/// Write a 4 KiB block to the dirty cache (it will be flushed on commit).
pub fn write_block_to_cache(
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    block_address: u64,
    block_data: Vec<u8>,
) {
    debug_assert_eq!(block_data.len(), BAFS_DEFAULT_BLOCK_SIZE_BYTES as usize);
    dirty_cache.insert(block_address, block_data);
}

/// Read and parse a B-tree node from the dirty cache or device.
pub fn read_tree_node(
    device: &dyn BlockDevice,
    dirty_cache: &BTreeMap<u64, Vec<u8>>,
    block_address: u64,
) -> Result<BafsTreeNode, BafsError> {
    let block_data = read_block(device, dirty_cache, block_address)?;
    deserialise_node_from_block(&block_data, block_address)
}

/// Serialise a node and write it to the dirty cache.
pub fn write_tree_node_to_cache(
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    node: &BafsTreeNode,
) {
    let block_data = serialise_node_to_block(node);
    write_block_to_cache(dirty_cache, node.self_block_address, block_data);
}

// ─── Lookup ───────────────────────────────────────────────────────────────────

/// Search a B-tree for the value associated with `target_key`.
///
/// Traverses from `root_block_address` down to the appropriate leaf, following
/// child pointers in internal nodes.  Returns `Ok(Some(value_bytes))` if the
/// exact key is found, `Ok(None)` if the key does not exist in the tree.
pub fn lookup_in_tree(
    device: &dyn BlockDevice,
    dirty_cache: &BTreeMap<u64, Vec<u8>>,
    root_block_address: u64,
    target_key: BafsKey,
) -> Result<Option<Vec<u8>>, BafsError> {
    let mut current_block_address = root_block_address;

    loop {
        let node = read_tree_node(device, dirty_cache, current_block_address)?;

        if node.level == 0 {
            // ── Leaf node: binary-search for the exact key ──
            match node
                .leaf_items
                .binary_search_by_key(&target_key, |item| item.key)
            {
                Ok(index) => return Ok(Some(node.leaf_items[index].value.clone())),
                Err(_) => return Ok(None),
            }
        } else {
            // ── Internal node: find the child whose subtree may contain the key ──
            // The child at index `i` contains all keys in [internal_items[i].key,
            // internal_items[i+1].key).  We want the last item whose key ≤ target_key.
            let child_index = find_child_index_for_key(&node.internal_items, target_key);
            current_block_address = node.internal_items[child_index].child_block;
        }
    }
}

/// Find the index of the internal item whose child subtree should contain
/// `target_key`.
///
/// Returns the index of the rightmost internal item with a separator key ≤
/// `target_key`.  If `target_key` is smaller than all separator keys, returns
/// index 0 (the leftmost child).
fn find_child_index_for_key(
    internal_items: &[BafsInternalItem],
    target_key: BafsKey,
) -> usize {
    // Binary-search for the insertion point of `target_key`.
    let insertion_point = internal_items
        .partition_point(|item| item.key <= target_key);
    // The child we want is the one immediately before the insertion point.
    // Clamp to 0 so we never underflow.
    insertion_point.saturating_sub(1)
}

// ─── Range iteration ──────────────────────────────────────────────────────────

/// Collect all leaf items whose key falls in [`min_key`, `max_key`].
///
/// Traverses the tree from the root, then performs an in-order scan of all
/// leaf nodes in the range.  Results are returned in ascending key order.
///
/// Used by `dir.rs` to enumerate directory entries and by `extent.rs` to find
/// all extents belonging to an inode.
pub fn iterate_tree_range(
    device: &dyn BlockDevice,
    dirty_cache: &BTreeMap<u64, Vec<u8>>,
    root_block_address: u64,
    min_key: BafsKey,
    max_key: BafsKey,
) -> Result<Vec<BafsLeafItem>, BafsError> {
    let mut results = Vec::new();
    collect_range(
        device,
        dirty_cache,
        root_block_address,
        min_key,
        max_key,
        &mut results,
    )?;
    Ok(results)
}

/// Recursive helper for `iterate_tree_range`.
fn collect_range(
    device: &dyn BlockDevice,
    dirty_cache: &BTreeMap<u64, Vec<u8>>,
    block_address: u64,
    min_key: BafsKey,
    max_key: BafsKey,
    output: &mut Vec<BafsLeafItem>,
) -> Result<(), BafsError> {
    let node = read_tree_node(device, dirty_cache, block_address)?;

    if node.level == 0 {
        // Leaf: collect all items in [min_key, max_key].
        for item in &node.leaf_items {
            if item.key >= min_key && item.key <= max_key {
                output.push(item.clone());
            }
        }
    } else {
        // Internal: recurse into children that could overlap the range.
        let items = &node.internal_items;
        for index in 0..items.len() {
            // This child's range is [items[index].key, items[index+1].key) or
            // [items[index].key, +∞) for the last child.
            let child_min_key = items[index].key;
            let child_max_key = items
                .get(index + 1)
                .map(|next| next.key)
                .unwrap_or(BafsKey::new(u64::MAX, u8::MAX, u64::MAX));

            // Skip this child if its range does not overlap [min_key, max_key].
            if child_max_key < min_key || child_min_key > max_key {
                continue;
            }
            collect_range(
                device,
                dirty_cache,
                items[index].child_block,
                min_key,
                max_key,
                output,
            )?;
        }
    }
    Ok(())
}

// ─── Insert ───────────────────────────────────────────────────────────────────

/// Insert or replace a key-value pair in the B-tree.
///
/// Uses Copy-on-Write semantics: new blocks are allocated from
/// `next_free_block` and written to `dirty_cache`.  The function returns the
/// block address of the (possibly new) root node.
///
/// `next_free_block` is incremented for each new block allocated.
///
/// `freed_blocks` is appended with the block addresses of every node that is
/// superseded by a CoW clone during this operation.  The caller is responsible
/// for returning those blocks to the free-extent pool after the transaction
/// commits.
pub fn insert_into_tree(
    device: &dyn BlockDevice,
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    root_block_address: u64,
    key: BafsKey,
    value: Vec<u8>,
    generation: u64,
    next_free_block: &mut u64,
    freed_blocks: &mut Vec<u64>,
) -> Result<u64, BafsError> {
    // Recursively insert, getting back the (possibly new) root block address.
    let (new_root_address, split_result) = insert_recursive(
        device,
        dirty_cache,
        root_block_address,
        key,
        value,
        generation,
        next_free_block,
        freed_blocks,
    )?;

    match split_result {
        None => Ok(new_root_address),
        Some(promoted_item) => {
            // The root was split.  Create a new root internal node with two children.
            // The original root block is superseded; record it as freed.
            freed_blocks.push(root_block_address);
            let new_root_block = *next_free_block;
            *next_free_block += 1;

            // Determine the separator key for the original (left) child.
            // We need the first key in the new (right) child's subtree; that is
            // exactly `promoted_item.key`.
            let old_root_node = read_tree_node(device, dirty_cache, new_root_address)?;
            let left_first_key = if old_root_node.level == 0 {
                old_root_node.leaf_items[0].key
            } else {
                old_root_node.internal_items[0].key
            };

            let new_root = BafsTreeNode {
                node_checksum: 0, // filled by serialise_node_to_block
                level: old_root_node.level + 1,
                flags: 0,
                item_count: 2,
                self_block_address: new_root_block,
                generation,
                leaf_items: Vec::new(),
                internal_items: vec![
                    BafsInternalItem {
                        key: left_first_key,
                        child_block: new_root_address,
                        child_generation: generation,
                    },
                    BafsInternalItem {
                        key: promoted_item.key,
                        child_block: promoted_item.child_block,
                        child_generation: generation,
                    },
                ],
            };
            write_tree_node_to_cache(dirty_cache, &new_root);
            Ok(new_root_block)
        }
    }
}

/// Result of a recursive insert: the (possibly new) block address of the node
/// that was modified, plus an optional split item that needs to be promoted to
/// the parent if the node was full and had to be split.
struct SplitResult {
    /// Separator key for the new right-hand node.
    key: BafsKey,
    /// Block address of the new right-hand node.
    child_block: u64,
}

/// Recursively insert `key`/`value` into the subtree rooted at `block_address`.
///
/// Returns `(new_block_address, Option<SplitResult>)`.  If the subtree root
/// was cloned (CoW), `new_block_address` differs from `block_address`.  If the
/// node was full and needed to be split, `SplitResult` carries the separator
/// key and right-hand child address that the caller must insert into the
/// parent.
///
/// `freed_blocks` collects every old block address that is superseded by a CoW
/// clone so the caller can return them to the free-extent pool.
fn insert_recursive(
    device: &dyn BlockDevice,
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    block_address: u64,
    key: BafsKey,
    value: Vec<u8>,
    generation: u64,
    next_free_block: &mut u64,
    freed_blocks: &mut Vec<u64>,
) -> Result<(u64, Option<SplitResult>), BafsError> {
    let mut node = read_tree_node(device, dirty_cache, block_address)?;

    if node.level == 0 {
        // ── Leaf node ──
        // Find the insertion or replacement position.
        match node.leaf_items.binary_search_by_key(&key, |item| item.key) {
            Ok(existing_index) => {
                // Key already exists — replace the value.
                node.leaf_items[existing_index].value = value;
            }
            Err(insertion_index) => {
                // Key is new — insert in sorted order.
                node.leaf_items.insert(insertion_index, BafsLeafItem { key, value });
                node.item_count += 1;
            }
        }

        if !leaf_node_would_overflow(&node.leaf_items) {
            // Node is within capacity — write it as a CoW clone.
            // The old block at block_address is superseded.
            freed_blocks.push(block_address);
            let new_block = *next_free_block;
            *next_free_block += 1;
            node.self_block_address = new_block;
            node.generation = generation;
            write_tree_node_to_cache(dirty_cache, &node);
            Ok((new_block, None))
        } else {
            // Node is full — split it into two halves.
            let (left_node, right_node, split_key) =
                split_leaf_node(node, generation, next_free_block, freed_blocks, block_address);
            let left_block = left_node.self_block_address;
            let right_block = right_node.self_block_address;
            write_tree_node_to_cache(dirty_cache, &left_node);
            write_tree_node_to_cache(dirty_cache, &right_node);
            Ok((left_block, Some(SplitResult { key: split_key, child_block: right_block })))
        }
    } else {
        // ── Internal node ──
        // Find the child that should receive the insertion.
        let child_index = find_child_index_for_key(&node.internal_items, key);
        let child_block_address = node.internal_items[child_index].child_block;

        let (new_child_block, split_result) = insert_recursive(
            device,
            dirty_cache,
            child_block_address,
            key,
            value,
            generation,
            next_free_block,
            freed_blocks,
        )?;

        // Update the child pointer in this node.
        node.internal_items[child_index].child_block = new_child_block;
        node.internal_items[child_index].child_generation = generation;

        // If the child was split, insert the promoted separator into this node.
        if let Some(promoted) = split_result {
            let insert_position = child_index + 1;
            node.internal_items.insert(
                insert_position,
                BafsInternalItem {
                    key: promoted.key,
                    child_block: promoted.child_block,
                    child_generation: generation,
                },
            );
            node.item_count += 1;
        }

        if node.internal_items.len() <= INTERNAL_MAX_ITEMS {
            // Within capacity — CoW clone.
            // The old block at block_address is superseded.
            freed_blocks.push(block_address);
            let new_block = *next_free_block;
            *next_free_block += 1;
            node.self_block_address = new_block;
            node.generation = generation;
            write_tree_node_to_cache(dirty_cache, &node);
            Ok((new_block, None))
        } else {
            // Full — split internal node.
            let (left_node, right_node, split_key) =
                split_internal_node(node, generation, next_free_block, freed_blocks, block_address);
            let left_block = left_node.self_block_address;
            let right_block = right_node.self_block_address;
            write_tree_node_to_cache(dirty_cache, &left_node);
            write_tree_node_to_cache(dirty_cache, &right_node);
            Ok((left_block, Some(SplitResult { key: split_key, child_block: right_block })))
        }
    }
}

/// Split a full leaf node into two roughly equal halves.
///
/// Returns `(left_node, right_node, right_first_key)`.  The right node's first
/// key is the separator that the parent must insert to route lookups correctly.
///
/// `freed_blocks` receives `old_block_address` (the original block being split)
/// because that block is superseded by the two newly allocated halves.
fn split_leaf_node(
    mut full_node: BafsTreeNode,
    generation: u64,
    next_free_block: &mut u64,
    freed_blocks: &mut Vec<u64>,
    old_block_address: u64,
) -> (BafsTreeNode, BafsTreeNode, BafsKey) {
    // The original node block is superseded by the two new halves.
    freed_blocks.push(old_block_address);

    // Find a split point where both halves fit in a single block.
    // Start at the midpoint and adjust if needed.
    let block_size = BAFS_DEFAULT_BLOCK_SIZE_BYTES as usize;
    let mut split_index = full_node.leaf_items.len() / 2;

    // Ensure the left half fits; if not, move the split point left.
    while split_index > 1 && leaf_serialised_size(&full_node.leaf_items[..split_index]) > block_size {
        split_index -= 1;
    }
    // Ensure the right half fits; if not, move the split point right.
    while split_index < full_node.leaf_items.len() - 1
        && leaf_serialised_size(&full_node.leaf_items[split_index..]) > block_size
    {
        split_index += 1;
    }

    let right_items: Vec<BafsLeafItem> = full_node.leaf_items.drain(split_index..).collect();
    let split_key = right_items[0].key;

    let right_block = *next_free_block;
    *next_free_block += 1;
    let left_block = *next_free_block;
    *next_free_block += 1;

    let right_node = BafsTreeNode {
        node_checksum: 0,
        level: 0,
        flags: 0,
        item_count: right_items.len() as u16,
        self_block_address: right_block,
        generation,
        leaf_items: right_items,
        internal_items: Vec::new(),
    };

    full_node.item_count = full_node.leaf_items.len() as u16;
    full_node.self_block_address = left_block;
    full_node.generation = generation;

    (full_node, right_node, split_key)
}

/// Split a full internal node into two roughly equal halves.
///
/// The middle separator key is promoted to the parent; it is not stored in
/// either of the resulting nodes.
///
/// `freed_blocks` receives `old_block_address` (the original block being split)
/// because that block is superseded by the two newly allocated halves.
fn split_internal_node(
    mut full_node: BafsTreeNode,
    generation: u64,
    next_free_block: &mut u64,
    freed_blocks: &mut Vec<u64>,
    old_block_address: u64,
) -> (BafsTreeNode, BafsTreeNode, BafsKey) {
    // The original node block is superseded by the two new halves.
    freed_blocks.push(old_block_address);

    let split_index = full_node.internal_items.len() / 2;
    // The item at `split_index` becomes the promoted separator.
    let promoted_item = full_node.internal_items.remove(split_index);
    let right_items: Vec<BafsInternalItem> =
        full_node.internal_items.drain(split_index..).collect();
    let split_key = promoted_item.key;

    let right_block = *next_free_block;
    *next_free_block += 1;
    let left_block = *next_free_block;
    *next_free_block += 1;

    // The promoted separator goes UP to the parent — it must NOT appear in
    // either child.  The right child receives all items that came after the
    // promoted separator, starting with a new first entry whose key is the
    // smallest key reachable through the first right child's subtree.
    //
    // Concretely: `promoted_item` was the middle item (index split_index).
    // Its child_block pointer is the leftmost child of the right half.
    // The right node's separator for that child is `promoted_item.key`, and
    // the right node contains [promoted_item, right_items[0], right_items[1], ...].
    //
    // Wait — in an internal node, each item's `key` is the separator for its
    // child subtree.  The standard B-tree split for internal nodes:
    //   left  = items[0 .. split_index]       (kept in full_node)
    //   mid   = items[split_index]             (promoted to parent)
    //   right = items[split_index+1 ..]       (in right_node)
    //
    // But the right child needs a first entry that routes into mid's child_block.
    // In BAFS internal nodes, item[i].child_block is the subtree for keys
    // >= item[i].key (and < item[i+1].key).  So the promoted item's child
    // becomes the first child of the right node, with key = promoted_item.key.
    //
    // This is exactly what the original code did — re-adding promoted_item
    // as the first entry of the right node — and it IS correct for the BAFS
    // internal node representation where item[i].key is the *minimum* key of
    // the subtree rooted at item[i].child_block.
    //
    // The promoted key sent to the parent is `promoted_item.key` (the boundary
    // between left and right subtrees).
    //
    // So: right node = [promoted_item, right_items[0], right_items[1], ...].
    // This means the promoted key appears in both the parent AND as the first
    // key of the right child.  That is intentional in this representation:
    // the parent's separator says "all keys >= this go right", and the right
    // child's first item confirms the minimum key in that subtree.
    let mut right_items_with_first = vec![promoted_item];
    right_items_with_first.extend(right_items);

    let right_node = BafsTreeNode {
        node_checksum: 0,
        level: full_node.level,
        flags: 0,
        item_count: right_items_with_first.len() as u16,
        self_block_address: right_block,
        generation,
        leaf_items: Vec::new(),
        internal_items: right_items_with_first,
    };

    full_node.item_count = full_node.internal_items.len() as u16;
    full_node.self_block_address = left_block;
    full_node.generation = generation;

    (full_node, right_node, split_key)
}

// ─── Delete ───────────────────────────────────────────────────────────────────

/// Remove the item with `target_key` from the B-tree.
///
/// Returns the new root block address (may change due to CoW or root collapse).
/// Returns `Err(BafsError::NotFound)` if the key does not exist.
///
/// `freed_blocks` is appended with the block addresses of every node that is
/// superseded by a CoW clone during this operation.  The caller is responsible
/// for returning those blocks to the free-extent pool after the transaction
/// commits.
pub fn delete_from_tree(
    device: &dyn BlockDevice,
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    root_block_address: u64,
    target_key: BafsKey,
    generation: u64,
    next_free_block: &mut u64,
    freed_blocks: &mut Vec<u64>,
) -> Result<u64, BafsError> {
    let (new_root, _) = delete_recursive(
        device,
        dirty_cache,
        root_block_address,
        target_key,
        generation,
        next_free_block,
        freed_blocks,
    )?;
    Ok(new_root)
}

/// Recursive helper for `delete_from_tree`.
///
/// Returns `(new_block_address, key_was_deleted)`.
///
/// `freed_blocks` collects every old block address that is superseded by a CoW
/// clone so the caller can return them to the free-extent pool.
fn delete_recursive(
    device: &dyn BlockDevice,
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    block_address: u64,
    target_key: BafsKey,
    generation: u64,
    next_free_block: &mut u64,
    freed_blocks: &mut Vec<u64>,
) -> Result<(u64, bool), BafsError> {
    let mut node = read_tree_node(device, dirty_cache, block_address)?;

    if node.level == 0 {
        // ── Leaf node ──
        match node
            .leaf_items
            .binary_search_by_key(&target_key, |item| item.key)
        {
            Ok(index) => {
                node.leaf_items.remove(index);
                node.item_count -= 1;
                // The old block at block_address is superseded by the CoW clone.
                freed_blocks.push(block_address);
                let new_block = *next_free_block;
                *next_free_block += 1;
                node.self_block_address = new_block;
                node.generation = generation;
                write_tree_node_to_cache(dirty_cache, &node);
                Ok((new_block, true))
            }
            Err(_) => Err(BafsError::NotFound),
        }
    } else {
        // ── Internal node ──
        let child_index = find_child_index_for_key(&node.internal_items, target_key);
        let child_block_address = node.internal_items[child_index].child_block;

        let (new_child_block, deleted) = delete_recursive(
            device,
            dirty_cache,
            child_block_address,
            target_key,
            generation,
            next_free_block,
            freed_blocks,
        )?;

        if !deleted {
            return Err(BafsError::NotFound);
        }

        node.internal_items[child_index].child_block = new_child_block;
        node.internal_items[child_index].child_generation = generation;

        // The old block at block_address is superseded by the CoW clone.
        freed_blocks.push(block_address);
        let new_block = *next_free_block;
        *next_free_block += 1;
        node.self_block_address = new_block;
        node.generation = generation;
        write_tree_node_to_cache(dirty_cache, &node);
        Ok((new_block, true))
    }
}

// ─── Empty tree creation ──────────────────────────────────────────────────────

/// Create an empty leaf node (an empty B-tree) and write it to `dirty_cache`.
///
/// Returns the block address of the new leaf.  Used during `bafs_format` to
/// initialise the three tree roots.
pub fn create_empty_tree(
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    block_address: u64,
    generation: u64,
) {
    let empty_leaf = BafsTreeNode {
        node_checksum: 0,
        level: 0,
        flags: 0,
        item_count: 0,
        self_block_address: block_address,
        generation,
        leaf_items: Vec::new(),
        internal_items: Vec::new(),
    };
    write_tree_node_to_cache(dirty_cache, &empty_leaf);
}
