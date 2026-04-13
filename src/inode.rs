//! BAFS inode: on-disk layout, read, write, and allocation.
//!
//! An inode (index node) holds the metadata for one file, directory, or other
//! filesystem object.  BAFS stores inodes inside the inode B-tree using the
//! key `(inode_number, ITEM_TYPE_INODE, 0)`.  The value is a serialised
//! `BafsInode` struct (128 bytes).
//!
//! # On-disk layout of BafsInode (128 bytes, little-endian)
//!
//! ```text
//! Offset  Size  Field
//! ──────  ────  ──────────────────────────────────────────────────────
//!      0     8  inode_number
//!      8     8  file_size_in_bytes  (0 for directories)
//!     16     8  allocated_block_count
//!     24     8  generation  (transaction_id when last modified)
//!     32     8  creation_time_nanoseconds
//!     40     8  modification_time_nanoseconds
//!     48     8  status_change_time_nanoseconds
//!     56     8  access_time_nanoseconds
//!     64     4  posix_mode  (type + permission bits)
//!     68     4  owner_uid
//!     72     4  owner_gid
//!     76     4  hard_link_count
//!     80     4  inode_flags  (IMMUTABLE, APPEND_ONLY, NO_ATIME, …)
//!     84     4  compression_algorithm  (0 = none; v3+ feature)
//!     88     8  encryption_key_id  (0 = none; v3+ feature)
//!     96    32  reserved  (zero-padded)
//! ```
//!
//! # Impact on the rest of the system
//!
//! - `volume.rs` calls `read_inode_from_tree` and `write_inode_to_tree` for
//!   every file/directory operation.
//! - `dir.rs` reads the parent inode to update its `hard_link_count` and
//!   `modification_time_nanoseconds` when creating or removing directory entries.
//! - `kernel.rs` converts `BafsInode` fields into the kernel's `InodeStat`
//!   struct for the VFS layer.

#[cfg(feature = "kernel")]
use alloc::{vec, vec::Vec, collections::BTreeMap};
#[cfg(not(feature = "kernel"))]
use std::{vec, vec::Vec, collections::BTreeMap};

use crate::block_device::BlockDevice;
use crate::btree::{lookup_in_tree, insert_into_tree, BafsKey, ITEM_TYPE_INODE};
use crate::error::BafsError;

// ─── POSIX mode constants ─────────────────────────────────────────────────────

/// Mode bits that identify a regular file (`S_IFREG` in POSIX).
pub const INODE_MODE_REGULAR_FILE: u32 = 0o100000;

/// Mode bits that identify a directory (`S_IFDIR` in POSIX).
pub const INODE_MODE_DIRECTORY: u32 = 0o040000;

/// Default permission bits for a newly created regular file (`rw-r--r--`).
pub const INODE_DEFAULT_FILE_PERMISSIONS: u32 = 0o644;

/// Default permission bits for a newly created directory (`rwxr-xr-x`).
pub const INODE_DEFAULT_DIRECTORY_PERMISSIONS: u32 = 0o755;

// ─── On-disk structure ────────────────────────────────────────────────────────

/// Metadata record for one filesystem object, stored in the inode B-tree.
///
/// All fields are in little-endian byte order on disk.  The struct is
/// `#[repr(C)]` with a compile-time size assertion to guarantee the 128-byte
/// layout matches the specification.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BafsInode {
    /// Unique inode number.  Monotonically increasing; never reused after the
    /// inode is freed (v1 policy).
    pub inode_number: u64,

    /// File size in bytes.  Zero for directories (their size is determined by
    /// the number of directory-entry items in the B-tree, not by this field).
    pub file_size_in_bytes: u64,

    /// Number of 4 KiB blocks allocated to hold this inode's data extents.
    pub allocated_block_count: u64,

    /// Transaction ID when this inode was last modified.  Monotonically
    /// increasing; used to detect stale cached inodes.
    pub generation: u64,

    /// Nanoseconds since the Unix epoch when this inode was created.
    pub creation_time_nanoseconds: u64,

    /// Nanoseconds since the Unix epoch when the file's data was last modified.
    pub modification_time_nanoseconds: u64,

    /// Nanoseconds since the Unix epoch when this inode's metadata (mode,
    /// owner, link count, etc.) was last changed.
    pub status_change_time_nanoseconds: u64,

    /// Nanoseconds since the Unix epoch when the file's data was last accessed.
    pub access_time_nanoseconds: u64,

    /// POSIX file type and permission bits.  The upper 12 bits encode the
    /// object type (`INODE_MODE_REGULAR_FILE`, `INODE_MODE_DIRECTORY`, etc.)
    /// and the lower 12 bits encode the permission bits.
    pub posix_mode: u32,

    /// POSIX user ID of the file's owner.
    pub owner_uid: u32,

    /// POSIX group ID of the file's owner.
    pub owner_gid: u32,

    /// Number of hard links pointing to this inode.  When this reaches zero
    /// the inode and its data extents are freed.  Directories always start at 2
    /// (one for the parent's entry, one for the directory itself via `.`).
    pub hard_link_count: u32,

    /// Bitmask of per-inode flags.  In v1, bits 0 (IMMUTABLE) and 1
    /// (APPEND_ONLY) and 2 (NO_ATIME) are defined.  All other bits are zero.
    pub inode_flags: u32,

    /// Compression algorithm used for this inode's data.  Always 0 (none) in
    /// v1; v3+ may set this to 1 (LZ4) or 2 (Zstandard).
    pub compression_algorithm: u32,

    /// Encryption key identifier.  Always 0 (not encrypted) in v1; v4+ sets
    /// this to an index into the key table.
    pub encryption_key_id: u64,

    /// Reserved bytes, zero-padded.  Sized to bring the total struct size to
    /// exactly 128 bytes.
    pub reserved: [u8; 32],
}

// Compile-time guarantee that the struct is exactly 128 bytes.
const _INODE_SIZE_ASSERTION: () = assert!(core::mem::size_of::<BafsInode>() == 128);

impl BafsInode {
    /// Create a new inode for a regular file with default permissions.
    ///
    /// `timestamp_nanoseconds` should be the current wall-clock time in
    /// nanoseconds since the Unix epoch.  In the kernel, this comes from the
    /// system clock; in tests it can be any fixed value.
    pub fn new_regular_file(
        inode_number: u64,
        generation: u64,
        timestamp_nanoseconds: u64,
    ) -> Self {
        BafsInode {
            inode_number,
            file_size_in_bytes: 0,
            allocated_block_count: 0,
            generation,
            creation_time_nanoseconds: timestamp_nanoseconds,
            modification_time_nanoseconds: timestamp_nanoseconds,
            status_change_time_nanoseconds: timestamp_nanoseconds,
            access_time_nanoseconds: timestamp_nanoseconds,
            posix_mode: INODE_MODE_REGULAR_FILE | INODE_DEFAULT_FILE_PERMISSIONS,
            owner_uid: 0,
            owner_gid: 0,
            hard_link_count: 1,
            inode_flags: 0,
            compression_algorithm: 0,
            encryption_key_id: 0,
            reserved: [0u8; 32],
        }
    }

    /// Create a new inode for a directory with default permissions.
    ///
    /// The `hard_link_count` starts at 2: one for the parent directory's entry
    /// pointing to this directory, and one for the implicit `.` entry.
    pub fn new_directory(
        inode_number: u64,
        generation: u64,
        timestamp_nanoseconds: u64,
    ) -> Self {
        BafsInode {
            inode_number,
            file_size_in_bytes: 0,
            allocated_block_count: 0,
            generation,
            creation_time_nanoseconds: timestamp_nanoseconds,
            modification_time_nanoseconds: timestamp_nanoseconds,
            status_change_time_nanoseconds: timestamp_nanoseconds,
            access_time_nanoseconds: timestamp_nanoseconds,
            posix_mode: INODE_MODE_DIRECTORY | INODE_DEFAULT_DIRECTORY_PERMISSIONS,
            owner_uid: 0,
            owner_gid: 0,
            hard_link_count: 2,
            inode_flags: 0,
            compression_algorithm: 0,
            encryption_key_id: 0,
            reserved: [0u8; 32],
        }
    }

    /// Returns `true` if this inode represents a regular file.
    pub fn is_regular_file(&self) -> bool {
        (self.posix_mode & 0o170000) == INODE_MODE_REGULAR_FILE
    }

    /// Returns `true` if this inode represents a directory.
    pub fn is_directory(&self) -> bool {
        (self.posix_mode & 0o170000) == INODE_MODE_DIRECTORY
    }
}

// ─── Serialisation ────────────────────────────────────────────────────────────

/// Serialise a `BafsInode` into 128 bytes (little-endian).
pub fn serialise_inode_to_bytes(inode: &BafsInode) -> [u8; 128] {
    // Safety: BafsInode is repr(C) with no padding (verified by size assertion).
    let mut output = [0u8; 128];
    unsafe {
        core::ptr::copy_nonoverlapping(
            inode as *const BafsInode as *const u8,
            output.as_mut_ptr(),
            128,
        );
    }
    output
}

/// Deserialise 128 bytes into a `BafsInode`.
pub fn deserialise_inode_from_bytes(bytes: &[u8]) -> BafsInode {
    debug_assert!(bytes.len() >= 128);
    unsafe { core::ptr::read(bytes.as_ptr() as *const BafsInode) }
}

// ─── B-tree I/O ───────────────────────────────────────────────────────────────

/// Look up an inode by number in the inode B-tree.
///
/// Returns `Ok(inode)` if the inode is found, `Err(BafsError::NotFound)` if
/// the inode number does not exist in the tree.
pub fn read_inode_from_tree(
    device: &dyn BlockDevice,
    dirty_cache: &BTreeMap<u64, Vec<u8>>,
    inode_tree_root_block: u64,
    inode_number: u64,
) -> Result<BafsInode, BafsError> {
    let search_key = BafsKey::new(inode_number, ITEM_TYPE_INODE, 0);
    match lookup_in_tree(device, dirty_cache, inode_tree_root_block, search_key)? {
        Some(value_bytes) => {
            if value_bytes.len() < 128 {
                return Err(BafsError::CorruptedStructure);
            }
            Ok(deserialise_inode_from_bytes(&value_bytes))
        }
        None => Err(BafsError::NotFound),
    }
}

/// Insert or update an inode in the inode B-tree.
///
/// Returns the (possibly new) root block address of the inode B-tree.
pub fn write_inode_to_tree(
    device: &dyn BlockDevice,
    dirty_cache: &mut BTreeMap<u64, Vec<u8>>,
    inode_tree_root_block: u64,
    inode: &BafsInode,
    generation: u64,
    next_free_block: &mut u64,
) -> Result<u64, BafsError> {
    let key = BafsKey::new(inode.inode_number, ITEM_TYPE_INODE, 0);
    let value = serialise_inode_to_bytes(inode).to_vec();
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

/// Generate the next inode number by incrementing the counter in the superblock.
///
/// `current_inode_count` is `superblock.allocated_inode_count`.  We return the
/// new inode number (= old count + 1) and the caller is responsible for
/// storing the incremented count back into the superblock.
pub fn allocate_next_inode_number(current_inode_count: u64) -> u64 {
    current_inode_count + 1
}
