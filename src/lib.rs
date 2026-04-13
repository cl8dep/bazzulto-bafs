//! # BAFS — Bazzulto File System
//!
//! BAFS is a copy-on-write, journaled filesystem designed for the Bazzulto OS.
//! It is implemented as a standalone Rust crate so that it can be used both
//! inside the kernel (`no_std + alloc`) and in userspace tools and test
//! harnesses (`std`).
//!
//! ## Architecture overview
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  Public API (this file)                                          │
//! │  bafs_format  bafs_mount  BafsVolume  BafsFormatOptions  …      │
//! ├──────────────────────────────────────────────────────────────────┤
//! │  volume.rs   High-level file/directory operations                │
//! ├──────────────────────────────────────────────────────────────────┤
//! │  dir.rs      inode.rs    extent.rs   checksum_tree.rs           │
//! ├──────────────────────────────────────────────────────────────────┤
//! │  btree.rs    Unified B-tree (CoW, lookup, insert, delete)        │
//! ├──────────────────────────────────────────────────────────────────┤
//! │  journal.rs  superblock.rs  checksum.rs                         │
//! ├──────────────────────────────────────────────────────────────────┤
//! │  block_device.rs  BlockDevice trait (disk I/O abstraction)       │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Feature flags
//!
//! | Feature     | Effect                                                    |
//! |-------------|-----------------------------------------------------------|
//! | `kernel`    | Activates `no_std + alloc`; compiles `kernel.rs`          |
//! | `userspace` | Activates `std`; for tooling and host-side usage           |
//! | (neither)   | Core structures only; used as a library without any I/O   |
//!
//! ## Quick start (host-side test)
//!
//! ```rust,no_run
//! use bafs::{bafs_format, bafs_mount, BafsFormatOptions, volume::bafs_unmount};
//! use bafs::volume::{volume_create_file, volume_write_file_data, volume_read_file_data};
//!
//! // (Define MemoryDisk implementing BlockDevice, then:)
//! // bafs_format(&disk, BafsFormatOptions::default()).unwrap();
//! // let mut vol = bafs_mount(disk).unwrap();
//! // let file_ino = volume_create_file(&mut vol, 1, "hello.txt").unwrap();
//! // volume_write_file_data(&mut vol, file_ino, 0, b"hello, bafs").unwrap();
//! // bafs_unmount(vol).unwrap();
//! ```

// ── no_std gate ───────────────────────────────────────────────────────────────
// When compiled with feature="kernel" the crate is no_std.  The `alloc` crate
// provides Vec, BTreeMap, String, and Arc.

#![cfg_attr(feature = "kernel", no_std)]

#[cfg(feature = "kernel")]
extern crate alloc;

// ── Module declarations ───────────────────────────────────────────────────────

/// Abstract block device interface used for all disk I/O.
pub mod block_device;

/// CRC32C checksum algorithm (hand-rolled, no external dependencies).
pub mod checksum;

/// On-disk superblock layout, serialisation, and I/O.
pub mod superblock;

/// Unified copy-on-write B-tree: node layout, lookup, insert, delete.
pub mod btree;

/// Write-ahead journal: transaction lifecycle and crash recovery.
pub mod journal;

/// On-disk inode structure, read/write through the inode B-tree.
pub mod inode;

/// Extent allocation: file-to-disk block mapping, free-extent B-tree.
pub mod extent;

/// Directory operations: lookup, create, unlink, iterate.  Includes xxHash-64.
pub mod dir;

/// Per-data-block checksum storage and verification (checksum B-tree).
pub mod checksum_tree;

/// High-level volume operations: format, mount, unmount, file/dir CRUD.
pub mod volume;

/// Kernel VFS `Inode` trait implementation (compiled only with feature="kernel").
#[cfg(feature = "kernel")]
pub mod kernel;

// ── Public re-exports ─────────────────────────────────────────────────────────
//
// These are the symbols that consumers (the kernel, userspace tools, and tests)
// are expected to import directly.

pub use block_device::BlockDevice;
pub use error::BafsError;
pub use volume::{bafs_format, bafs_mount, BafsFormatOptions, BafsVolume};

/// Error types for all BAFS operations.
pub mod error;
