//! Error types for the Bazzulto File System (BAFS).
//!
//! This module defines `BafsError`, the single error type returned by every
//! fallible BAFS operation.  All public functions in the crate return
//! `Result<T, BafsError>` so callers have a uniform way to handle failures.
//!
//! # Impact on the rest of the system
//!
//! When BAFS is compiled with `feature = "kernel"`, the `kernel` module maps
//! `BafsError` values to the kernel's `FsError` enum so that the VFS layer
//! gets standard POSIX-compatible error codes.  In userspace (tools, tests)
//! callers receive the `BafsError` directly and can format it with `Display`.

/// Every possible failure that can occur inside a BAFS operation.
///
/// Variants are kept fine-grained so that callers can distinguish between, for
/// example, a bad checksum (indicating on-disk corruption) and a missing entry
/// (indicating a normal not-found condition).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BafsError {
    /// A read or write to the underlying block device failed.
    InputOutputError,

    /// The CRC32C checksum stored in a block does not match the checksum
    /// computed over its contents.  The `block_address` field identifies which
    /// block is corrupt.
    InvalidChecksum { block_address: u64 },

    /// The eight-byte magic number at the start of the superblock did not
    /// match the expected value `b"BAFS\x1B\x00\x00\x00"`.
    InvalidMagicNumber,

    /// The superblock reports a filesystem version that this implementation
    /// does not support.  `found_version` is what was read from disk.
    UnsupportedVersion { found_version: u32 },

    /// The filesystem has no free blocks left to satisfy an allocation request.
    OutOfSpace,

    /// A requested file, directory, or tree entry does not exist.
    NotFound,

    /// A directory operation was attempted on an inode that is not a directory.
    NotADirectory,

    /// A file operation was attempted on an inode that is not a regular file.
    NotARegularFile,

    /// A create operation failed because the target name already exists in the
    /// parent directory.
    AlreadyExists,

    /// A caller passed an argument that is outside the valid range (e.g. a
    /// zero-length name, an offset past the end of the file, etc.).
    InvalidArgument,

    /// On-disk metadata is internally inconsistent in a way that cannot be
    /// attributed to a simple I/O error (e.g. a B-tree node claims to be
    /// internal but its level field is zero, or an extent points outside the
    /// data area).
    CorruptedStructure,

    /// The requested operation is not supported in the current filesystem
    /// version or configuration.
    NotSupported,
}

impl core::fmt::Display for BafsError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BafsError::InputOutputError => {
                write!(formatter, "I/O error on the underlying block device")
            }
            BafsError::InvalidChecksum { block_address } => {
                write!(
                    formatter,
                    "CRC32C checksum mismatch on block {}",
                    block_address
                )
            }
            BafsError::InvalidMagicNumber => {
                write!(formatter, "not a BAFS filesystem (invalid magic number)")
            }
            BafsError::UnsupportedVersion { found_version } => {
                write!(
                    formatter,
                    "unsupported BAFS version {} (this implementation supports v1 only)",
                    found_version
                )
            }
            BafsError::OutOfSpace => {
                write!(formatter, "no free space left on the filesystem")
            }
            BafsError::NotFound => {
                write!(formatter, "no such file or directory")
            }
            BafsError::NotADirectory => {
                write!(formatter, "not a directory")
            }
            BafsError::NotARegularFile => {
                write!(formatter, "not a regular file")
            }
            BafsError::AlreadyExists => {
                write!(formatter, "file or directory already exists")
            }
            BafsError::InvalidArgument => {
                write!(formatter, "invalid argument")
            }
            BafsError::CorruptedStructure => {
                write!(formatter, "filesystem metadata is corrupted")
            }
            BafsError::NotSupported => {
                write!(formatter, "operation not supported")
            }
        }
    }
}
