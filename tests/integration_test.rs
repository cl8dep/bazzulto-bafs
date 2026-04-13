//! Integration test for BAFS: format → mount → write → read → unmount → remount → read.
//!
//! This test exercises the full BAFS lifecycle on a 64 MiB in-memory disk.  It
//! verifies that:
//!
//! 1. A freshly formatted volume can be mounted.
//! 2. Files can be created and written.
//! 3. Written data can be read back within the same mount.
//! 4. After unmounting (which commits the journal) and remounting, the data
//!    persists and can be read again.
//!
//! The test runs on the host (x86-64 or AArch64 macOS / Linux) using
//! `cargo test`.  No QEMU is required.

use std::cell::UnsafeCell;
use std::sync::{Arc, Mutex};

use bafs::block_device::BlockDevice;
use bafs::volume::{
    bafs_format, bafs_mount, bafs_unmount, volume_create_file,
    volume_lookup_directory_entry, volume_read_file_data, volume_write_file_data,
    BafsFormatOptions,
};

// ─── MemoryDisk ───────────────────────────────────────────────────────────────

/// An in-memory block device backed by a `Vec<u8>`.
///
/// The `data` is wrapped in a `Mutex` so that `MemoryDisk` satisfies the
/// `Send + Sync` bounds required by `BlockDevice`.
///
/// After unmounting one `BafsVolume` the same `MemoryDisk` can be passed to
/// `bafs_mount` again because the `Arc<Mutex<Vec<u8>>>` keeps the bytes alive.
pub struct MemoryDisk {
    /// Raw sector storage.  Length = total_sectors * SECTOR_SIZE_BYTES.
    data: Arc<Mutex<Vec<u8>>>,
    /// Total number of 512-byte sectors.
    total_sectors: u64,
}

const SECTOR_SIZE_BYTES: usize = 512;

/// Size of the in-memory test disk.
///
/// 512 MiB is a reasonable default for an integration test: large enough that
/// the filesystem overhead (journal, tree root blocks) is negligible relative
/// to the data area, and small enough to allocate without swapping on a modern
/// development machine (it is backed by a Vec<u8>, so the OS won't actually
/// fault all pages unless they are written to).
const MEMORY_DISK_SIZE_BYTES: usize = 512 * 1024 * 1024; // 512 MiB

impl MemoryDisk {
    /// Create a new 512 MiB in-memory disk initialised to all-zeros.
    ///
    /// 512 MiB is chosen because:
    /// - It gives the filesystem a realistic data area (≈510 MiB after overhead).
    /// - A Vec<u8> of this size is allocated virtually; the OS pages are only
    ///   faulted in when actually written, so the test process does not use
    ///   512 MiB of physical RAM unless the test writes that much data.
    pub fn new_512mib() -> Self {
        let total_sectors = MEMORY_DISK_SIZE_BYTES as u64 / SECTOR_SIZE_BYTES as u64;
        MemoryDisk {
            data: Arc::new(Mutex::new(vec![0u8; MEMORY_DISK_SIZE_BYTES])),
            total_sectors,
        }
    }

    /// Clone the Arc so that a second `BafsVolume` can be mounted on the same
    /// bytes after the first is unmounted.
    pub fn clone_arc(&self) -> Self {
        MemoryDisk {
            data: Arc::clone(&self.data),
            total_sectors: self.total_sectors,
        }
    }
}

impl BlockDevice for MemoryDisk {
    fn read_sectors(
        &self,
        start_lba: u64,
        sector_count: u32,
        destination_buffer: &mut [u8],
    ) -> bool {
        let start_byte = start_lba as usize * SECTOR_SIZE_BYTES;
        let byte_count = sector_count as usize * SECTOR_SIZE_BYTES;
        let data = self.data.lock().unwrap();
        if start_byte + byte_count > data.len() {
            return false;
        }
        destination_buffer[..byte_count].copy_from_slice(&data[start_byte..start_byte + byte_count]);
        true
    }

    fn write_sectors(
        &self,
        start_lba: u64,
        sector_count: u32,
        source_buffer: &[u8],
    ) -> bool {
        let start_byte = start_lba as usize * SECTOR_SIZE_BYTES;
        let byte_count = sector_count as usize * SECTOR_SIZE_BYTES;
        let mut data = self.data.lock().unwrap();
        if start_byte + byte_count > data.len() {
            return false;
        }
        data[start_byte..start_byte + byte_count].copy_from_slice(&source_buffer[..byte_count]);
        true
    }

    fn total_sector_count(&self) -> u64 {
        self.total_sectors
    }

    fn device_name(&self) -> &str {
        "MemoryDisk (64 MiB, in-process)"
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

/// Full lifecycle test: format → mount → create file → write → read →
/// unmount → remount → open → read again.
///
/// This is the canonical correctness test for BAFS v1.
#[test]
fn bafs_full_lifecycle_write_and_read_survives_remount() {
    // ── Step 1: Format the disk ───────────────────────────────────────────────

    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default())
        .expect("format should succeed on a fresh 64 MiB disk");

    // ── Step 2: Mount the formatted volume ────────────────────────────────────

    let disk_for_first_mount = disk.clone_arc();
    let mut volume = bafs_mount(disk_for_first_mount)
        .expect("mount should succeed immediately after format");

    // The root inode number is always 1 in BAFS v1.
    let root_inode_number = volume.superblock.root_inode_number;
    assert_eq!(root_inode_number, 1, "root inode number should be 1");

    // ── Step 3: Create a file in the root directory ───────────────────────────

    let file_inode_number = volume_create_file(&mut volume, root_inode_number, "hello.txt")
        .expect("creating hello.txt in the root directory should succeed");

    assert!(
        file_inode_number > root_inode_number,
        "new file inode number should be greater than the root inode number"
    );

    // ── Step 4: Write data to the file ────────────────────────────────────────

    let content_to_write: &[u8] = b"hello, bafs";
    let bytes_written =
        volume_write_file_data(&mut volume, file_inode_number, 0, content_to_write)
            .expect("writing to hello.txt should succeed");

    assert_eq!(
        bytes_written,
        content_to_write.len(),
        "all bytes should have been written"
    );

    // ── Step 5: Read back within the same mount and verify ────────────────────

    let mut read_buffer = vec![0u8; content_to_write.len()];
    let bytes_read =
        volume_read_file_data(&volume, file_inode_number, 0, &mut read_buffer)
            .expect("reading hello.txt within the same mount should succeed");

    assert_eq!(bytes_read, content_to_write.len(), "should read all written bytes");
    assert_eq!(
        &read_buffer[..bytes_read],
        content_to_write,
        "read content should match written content"
    );

    // ── Step 6: Unmount (commits the journal, flushes to disk) ───────────────

    bafs_unmount(volume).expect("unmount should succeed");

    // ── Step 7: Remount the same disk (same Vec<u8> bytes) ───────────────────

    let disk_for_second_mount = disk.clone_arc();
    let volume2 = bafs_mount(disk_for_second_mount)
        .expect("remount after unmount should succeed");

    // ── Step 8: Look up the file by name ─────────────────────────────────────

    let root_inode_number_after_remount = volume2.superblock.root_inode_number;
    let looked_up_inode_number = volume_lookup_directory_entry(
        &volume2,
        root_inode_number_after_remount,
        "hello.txt",
    )
    .expect("directory lookup should not return an error")
    .expect("hello.txt should be found in the root directory after remount");

    assert_eq!(
        looked_up_inode_number, file_inode_number,
        "the looked-up inode number should match the one assigned at creation"
    );

    // ── Step 9: Read the file again and verify persistence ───────────────────

    let mut persistent_read_buffer = vec![0u8; content_to_write.len()];
    let persistent_bytes_read = volume_read_file_data(
        &volume2,
        looked_up_inode_number,
        0,
        &mut persistent_read_buffer,
    )
    .expect("reading hello.txt after remount should succeed");

    assert_eq!(
        persistent_bytes_read,
        content_to_write.len(),
        "should read all bytes after remount"
    );
    assert_eq!(
        &persistent_read_buffer[..persistent_bytes_read],
        content_to_write,
        "content must match the original write after remount — this is the persistence guarantee"
    );

    // Unmount the second volume cleanly.
    bafs_unmount(volume2).expect("second unmount should succeed");
}

/// Verify that the CRC32C standard test vector is correct.
///
/// This is a sanity check on the checksum module that is fast to run and
/// catches any corruption of the polynomial table.
#[test]
fn crc32c_standard_test_vector_is_correct() {
    use bafs::checksum::compute_crc32c;
    // Standard CRC32C test vector: "123456789" → 0xE306_9283.
    assert_eq!(compute_crc32c(b"123456789"), 0xE306_9283);
}

/// Verify that xxHash-64 produces the known reference value for the empty string.
#[test]
fn xxhash64_empty_string_reference_value_is_correct() {
    use bafs::dir::xxhash64_with_seed_zero;
    // Reference: xxhash64("", seed=0) = 0xEF46DB3751D8E999
    assert_eq!(xxhash64_with_seed_zero(b""), 0xEF46DB3751D8E999);
}

/// Verify that creating two files in the same directory and looking them both
/// up works correctly (exercises directory B-tree with multiple entries).
#[test]
fn bafs_directory_with_multiple_files_lookup_works() {
    let disk = MemoryDisk::new_512mib();
    bafs_format(&disk, BafsFormatOptions::default()).unwrap();

    let mut volume = bafs_mount(disk.clone_arc()).unwrap();
    let root = volume.superblock.root_inode_number;

    let inode_a = volume_create_file(&mut volume, root, "alpha.txt").unwrap();
    let inode_b = volume_create_file(&mut volume, root, "beta.txt").unwrap();

    volume_write_file_data(&mut volume, inode_a, 0, b"file alpha").unwrap();
    volume_write_file_data(&mut volume, inode_b, 0, b"file beta").unwrap();

    bafs_unmount(volume).unwrap();

    // Remount and look up both files.
    let volume2 = bafs_mount(disk.clone_arc()).unwrap();
    let root2 = volume2.superblock.root_inode_number;

    let found_a = volume_lookup_directory_entry(&volume2, root2, "alpha.txt")
        .unwrap()
        .expect("alpha.txt should exist");
    let found_b = volume_lookup_directory_entry(&volume2, root2, "beta.txt")
        .unwrap()
        .expect("beta.txt should exist");

    assert_ne!(found_a, found_b, "different files should have different inode numbers");

    let mut buf_a = vec![0u8; 10];
    volume_read_file_data(&volume2, found_a, 0, &mut buf_a).unwrap();
    assert_eq!(&buf_a, b"file alpha");

    let mut buf_b = vec![0u8; 9];
    volume_read_file_data(&volume2, found_b, 0, &mut buf_b).unwrap();
    assert_eq!(&buf_b, b"file beta");

    bafs_unmount(volume2).unwrap();
}
