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

//! BAFS kernel VFS integration: implements the `Inode` trait for the Bazzulto
//! kernel's virtual filesystem layer.
//!
//! This module is only compiled when `feature = "kernel"` is active.  It
//! bridges the gap between the generic BAFS volume API (defined in `volume.rs`)
//! and the kernel's `Inode` trait (defined in
//! `kernel/src/fs/inode.rs`).
//!
//! # Architecture
//!
//! A single `BafsVolume` is shared by all inodes on the same filesystem.
//! Because the kernel is single-threaded in v1, we wrap the volume in an
//! `UnsafeCell` (matching the pattern used by `tmpfs` and `fat32`) rather than
//! a mutex.  When SMP arrives this will need to change to a spinlock.
//!
//! Each `BafsKernelInode` holds:
//! - An `Arc<BafsVolumeHandle>` (shared reference to the volume + device).
//! - The inode number of the filesystem object it represents.
//!
//! Calling any `Inode` trait method internally borrows the `UnsafeCell` and
//! dispatches to the corresponding `volume.rs` function.
//!
//! # Impact on the rest of the system
//!
//! - The kernel's VFS layer (`kernel/src/fs/vfs.rs`) receives
//!   `Arc<dyn Inode>` handles and calls the methods defined here.
//! - The filesystem is registered with the VFS by calling
//!   `bafs_mount_kernel_volume`, which returns an `Arc<dyn Inode>` for the
//!   root directory.

// This module is only meaningful in kernel builds.
#![cfg(feature = "kernel")]

extern crate alloc;
use alloc::{
    string::String,
    sync::Arc,
    vec::Vec,
};
use core::cell::UnsafeCell;

// Re-export the kernel's VFS types (available because we're linked into the
// kernel crate in kernel builds).
//
// In a real kernel build these would be:
//   use crate::fs::inode::{DirEntry, FsError, Inode, InodeStat, InodeType};
//
// Since BAFS is a standalone crate, we define compatible shim types here that
// mirror the kernel's types exactly.  When integrated into the kernel build
// system these shim types would be replaced by the kernel's actual types via
// a cfg alias or re-export.

use crate::block_device::BlockDevice;
use crate::dir::CHILD_TYPE_DIRECTORY;
use crate::error::BafsError;
use crate::inode::{INODE_MODE_DIRECTORY, INODE_MODE_REGULAR_FILE};
use crate::volume::{
    bafs_mount, flush_and_commit, volume_create_directory, volume_create_file,
    volume_lookup_directory_entry, volume_read_directory_entry_at_index,
    volume_read_file_data, volume_read_inode, volume_unlink_directory_entry,
    volume_write_file_data, BafsVolume,
};

// ─── Shim types mirroring kernel/src/fs/inode.rs ─────────────────────────────
//
// These types must exactly match the definitions in the kernel's inode.rs.
// They are repeated here so the BAFS crate compiles standalone; in a full
// kernel integration the `use kernel::fs::inode::*` forms would replace them.

/// Describes the kind of filesystem object an `Inode` represents.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InodeType {
    RegularFile,
    Directory,
    CharDevice,
    Fifo,
    Symlink,
}

/// Metadata about an inode, returned by `Inode::stat`.
#[derive(Clone, Copy, Debug)]
pub struct InodeStat {
    pub inode_number: u64,
    pub size: u64,
    /// Bits [15:12] = file type, bits [11:0] = permission bits.
    pub mode: u64,
    pub nlinks: u64,
}

impl InodeStat {
    /// Convenience constructor for regular files.
    pub fn regular(inode_number: u64, size: u64) -> Self {
        InodeStat {
            inode_number,
            size,
            mode: (INODE_MODE_REGULAR_FILE | 0o644) as u64,
            nlinks: 1,
        }
    }

    /// Convenience constructor for directories.
    pub fn directory(inode_number: u64) -> Self {
        InodeStat {
            inode_number,
            size: 0,
            mode: (INODE_MODE_DIRECTORY | 0o755) as u64,
            nlinks: 2,
        }
    }
}

/// A single directory entry as returned by `Inode::readdir`.
#[derive(Clone, Debug)]
pub struct DirEntry {
    pub name: String,
    pub inode_type: InodeType,
    pub inode_number: u64,
}

/// Filesystem errors returned by `Inode` trait methods.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FsError {
    NotSupported,
    NotFound,
    NotDirectory,
    AlreadyExists,
    DirectoryNotEmpty,
    OutOfMemory,
    IoError,
    PermissionDenied,
    BrokenPipe,
    WouldBlock,
    TooManyLinks,
    InvalidArgument,
}

impl FsError {
    /// Map to a POSIX errno value.
    pub fn to_errno(self) -> i64 {
        match self {
            FsError::NotSupported => -38,    // ENOSYS
            FsError::NotFound => -2,          // ENOENT
            FsError::NotDirectory => -20,     // ENOTDIR
            FsError::AlreadyExists => -17,    // EEXIST
            FsError::DirectoryNotEmpty => -39, // ENOTEMPTY
            FsError::OutOfMemory => -12,      // ENOMEM
            FsError::IoError => -5,           // EIO
            FsError::PermissionDenied => -13, // EACCES
            FsError::BrokenPipe => -32,       // EPIPE
            FsError::WouldBlock => -11,       // EAGAIN
            FsError::TooManyLinks => -31,     // EMLINK
            FsError::InvalidArgument => -22,  // EINVAL
        }
    }
}

/// Map a `BafsError` to the kernel's `FsError`.
fn map_bafs_error_to_fs_error(bafs_error: BafsError) -> FsError {
    match bafs_error {
        BafsError::NotFound => FsError::NotFound,
        BafsError::AlreadyExists => FsError::AlreadyExists,
        BafsError::NotADirectory => FsError::NotDirectory,
        BafsError::NotARegularFile => FsError::NotSupported,
        BafsError::OutOfSpace => FsError::OutOfMemory,
        BafsError::InvalidArgument => FsError::InvalidArgument,
        BafsError::InputOutputError => FsError::IoError,
        BafsError::InvalidChecksum { .. } => FsError::IoError,
        BafsError::CorruptedStructure => FsError::IoError,
        _ => FsError::IoError,
    }
}

/// The `Inode` trait that BAFS implements for the kernel VFS.
pub trait Inode: Send + Sync {
    fn inode_type(&self) -> InodeType;
    fn stat(&self) -> InodeStat;
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize, FsError>;
    fn write_at(&self, offset: u64, buf: &[u8]) -> Result<usize, FsError>;
    fn truncate(&self, new_size: u64) -> Result<(), FsError>;
    fn lookup(&self, name: &str) -> Option<Arc<dyn Inode>>;
    fn readdir(&self, index: usize) -> Option<DirEntry>;
    fn create(&self, name: &str) -> Result<Arc<dyn Inode>, FsError>;
    fn mkdir(&self, name: &str) -> Result<Arc<dyn Inode>, FsError>;
    fn unlink(&self, name: &str) -> Result<(), FsError>;
    fn set_mode(&self, _mode: u64) -> Result<(), FsError> { Ok(()) }
    fn link_child(&self, _name: &str, _child: Arc<dyn Inode>) -> Result<(), FsError> {
        Err(FsError::NotSupported)
    }
}

// ─── Shared volume handle ─────────────────────────────────────────────────────

/// Wraps a `BafsVolume` in an `UnsafeCell` so that it can be shared via `Arc`
/// across multiple `BafsKernelInode` instances.
///
/// # Safety
///
/// This is safe in the current single-threaded kernel.  All filesystem
/// operations are serialised by the kernel's scheduling model.  Before SMP
/// is enabled, a spinlock must wrap the `UnsafeCell`.
pub struct BafsVolumeHandle<D: BlockDevice + 'static> {
    inner: UnsafeCell<BafsVolume<D>>,
}

// Safety: see the doc comment above.
unsafe impl<D: BlockDevice + 'static> Send for BafsVolumeHandle<D> {}
unsafe impl<D: BlockDevice + 'static> Sync for BafsVolumeHandle<D> {}

impl<D: BlockDevice + 'static> BafsVolumeHandle<D> {
    fn new(volume: BafsVolume<D>) -> Self {
        BafsVolumeHandle {
            inner: UnsafeCell::new(volume),
        }
    }

    /// Borrow the volume immutably.
    ///
    /// # Safety
    ///
    /// The caller must ensure no mutable borrow is alive concurrently.
    unsafe fn borrow_volume(&self) -> &BafsVolume<D> {
        &*self.inner.get()
    }

    /// Borrow the volume mutably.
    ///
    /// # Safety
    ///
    /// The caller must ensure no other borrow (mutable or immutable) is alive
    /// concurrently.
    unsafe fn borrow_volume_mut(&self) -> &mut BafsVolume<D> {
        &mut *self.inner.get()
    }
}

// ─── BafsKernelInode ──────────────────────────────────────────────────────────

/// A kernel VFS inode that delegates to a shared `BafsVolume`.
///
/// Every file and directory on a mounted BAFS volume is represented by one of
/// these.  Multiple `BafsKernelInode` instances share the same
/// `Arc<BafsVolumeHandle>`.
pub struct BafsKernelInode<D: BlockDevice + 'static> {
    /// Shared reference to the mounted volume.
    volume_handle: Arc<BafsVolumeHandle<D>>,

    /// Inode number of the filesystem object this kernel inode represents.
    inode_number: u64,
}

impl<D: BlockDevice + 'static> BafsKernelInode<D> {
    fn new(volume_handle: Arc<BafsVolumeHandle<D>>, inode_number: u64) -> Arc<Self> {
        Arc::new(BafsKernelInode {
            volume_handle,
            inode_number,
        })
    }

    /// Get an `Arc<dyn Inode>` for a given inode number on the same volume.
    fn make_inode_arc(&self, target_inode_number: u64) -> Arc<dyn Inode> {
        Arc::new(BafsKernelInode {
            volume_handle: Arc::clone(&self.volume_handle),
            inode_number: target_inode_number,
        })
    }
}

impl<D: BlockDevice + 'static> Inode for BafsKernelInode<D> {
    fn inode_type(&self) -> InodeType {
        // Safety: single-threaded kernel; no concurrent mutable borrow.
        let volume = unsafe { self.volume_handle.borrow_volume() };
        match volume_read_inode(volume, self.inode_number) {
            Ok(inode) if inode.is_directory() => InodeType::Directory,
            _ => InodeType::RegularFile,
        }
    }

    fn stat(&self) -> InodeStat {
        let volume = unsafe { self.volume_handle.borrow_volume() };
        match volume_read_inode(volume, self.inode_number) {
            Ok(inode) => {
                if inode.is_directory() {
                    InodeStat::directory(self.inode_number)
                } else {
                    InodeStat::regular(self.inode_number, inode.file_size_in_bytes)
                }
            }
            Err(_) => InodeStat::regular(self.inode_number, 0),
        }
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize, FsError> {
        let volume = unsafe { self.volume_handle.borrow_volume() };
        volume_read_file_data(volume, self.inode_number, offset, buf)
            .map_err(map_bafs_error_to_fs_error)
    }

    fn write_at(&self, offset: u64, buf: &[u8]) -> Result<usize, FsError> {
        let volume = unsafe { self.volume_handle.borrow_volume_mut() };
        volume_write_file_data(volume, self.inode_number, offset, buf)
            .map_err(map_bafs_error_to_fs_error)
    }

    fn truncate(&self, _new_size: u64) -> Result<(), FsError> {
        // Truncation requires freeing extents, which is a v2 feature.
        Err(FsError::NotSupported)
    }

    fn lookup(&self, name: &str) -> Option<Arc<dyn Inode>> {
        let volume = unsafe { self.volume_handle.borrow_volume() };
        match volume_lookup_directory_entry(volume, self.inode_number, name) {
            Ok(Some(child_inode_number)) => {
                Some(self.make_inode_arc(child_inode_number))
            }
            _ => None,
        }
    }

    fn readdir(&self, index: usize) -> Option<DirEntry> {
        let volume = unsafe { self.volume_handle.borrow_volume() };
        match volume_read_directory_entry_at_index(volume, self.inode_number, index) {
            Ok(Some(entry)) => {
                let inode_type = if entry.child_type == CHILD_TYPE_DIRECTORY {
                    InodeType::Directory
                } else {
                    InodeType::RegularFile
                };
                Some(DirEntry {
                    name: entry.filename,
                    inode_type,
                    inode_number: entry.child_inode_number,
                })
            }
            _ => None,
        }
    }

    fn create(&self, name: &str) -> Result<Arc<dyn Inode>, FsError> {
        let volume = unsafe { self.volume_handle.borrow_volume_mut() };
        let new_inode_number = volume_create_file(volume, self.inode_number, name)
            .map_err(map_bafs_error_to_fs_error)?;
        Ok(self.make_inode_arc(new_inode_number))
    }

    fn mkdir(&self, name: &str) -> Result<Arc<dyn Inode>, FsError> {
        let volume = unsafe { self.volume_handle.borrow_volume_mut() };
        let new_inode_number =
            volume_create_directory(volume, self.inode_number, name)
                .map_err(map_bafs_error_to_fs_error)?;
        Ok(self.make_inode_arc(new_inode_number))
    }

    fn unlink(&self, name: &str) -> Result<(), FsError> {
        let volume = unsafe { self.volume_handle.borrow_volume_mut() };
        volume_unlink_directory_entry(volume, self.inode_number, name)
            .map_err(map_bafs_error_to_fs_error)
    }
}

// ─── Public mount entry point ─────────────────────────────────────────────────

/// Mount a BAFS filesystem from `device` and return the root inode.
///
/// This is the entry point called by the kernel's filesystem registration code.
/// The returned `Arc<dyn Inode>` represents the root directory and can be
/// passed to the VFS layer's mount point table.
pub fn bafs_mount_kernel_volume<D: BlockDevice + 'static>(
    device: D,
) -> Result<Arc<dyn Inode>, BafsError> {
    let volume = bafs_mount(device)?;
    let root_inode_number = volume.superblock.root_inode_number;
    let handle = Arc::new(BafsVolumeHandle::new(volume));
    Ok(BafsKernelInode::new(handle, root_inode_number))
}
