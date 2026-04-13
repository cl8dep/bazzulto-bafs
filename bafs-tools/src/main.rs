//! bafs-tools — host-side developer CLI for BAFS volumes.
//!
//! This binary is a **development tool** for use on macOS and Linux.  It is
//! not part of the Bazzulto OS userspace; those tools will be implemented
//! separately once the kernel reaches a suitable stage.
//!
//! # Purpose
//!
//! bafs-tools lets a developer format, inspect, and benchmark a BAFS volume
//! on their workstation without QEMU or real hardware.  The target is either
//! a regular file (used as a raw disk image) or a block device (e.g. a USB
//! drive or a macOS disk image attached with `hdiutil attach -nomount`).
//!
//! # Subcommands
//!
//! ```text
//! bafs-tools format  <image>              Format the image as a new BAFS volume
//! bafs-tools info    <image>              Print superblock fields
//! bafs-tools ls      <image> [path]       List the contents of a directory
//! bafs-tools write   <image> <dest> <src> Write a host file into the BAFS volume
//! bafs-tools read    <image> <path>       Read a file from the BAFS volume to stdout
//! bafs-tools bench   <image>              Write + read benchmark (reports MiB/s)
//! ```
//!
//! # Safe usage with disk images
//!
//! The recommended workflow that avoids any risk to real disks:
//!
//! ```bash
//! # Create a 1 GiB raw image (only allocates space when written, on macOS)
//! dd if=/dev/zero of=/tmp/bafs.img bs=1m count=1024
//!
//! # Format it
//! bafs-tools format /tmp/bafs.img
//!
//! # Inspect
//! bafs-tools info /tmp/bafs.img
//! bafs-tools ls   /tmp/bafs.img /
//! ```
//!
//! # Using a real block device (advanced)
//!
//! On macOS you can attach a raw disk image as a block device:
//!
//! ```bash
//! hdiutil attach -imagekey diskimage-class=CRawDiskImage -nomount /tmp/bafs.img
//! # → /dev/disk4
//! bafs-tools format /dev/disk4
//! ```
//!
//! NEVER point bafs-tools at your system disk or a disk containing data you
//! care about.  `format` overwrites the first few blocks unconditionally.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::Instant;

use bafs::block_device::BlockDevice;
use bafs::volume::{
    bafs_format, bafs_mount, bafs_unmount, flush_and_commit, volume_create_file,
    volume_lookup_directory_entry, volume_read_file_data, volume_write_file_data,
    BafsFormatOptions,
};

// ─── FileBlockDevice ──────────────────────────────────────────────────────────

/// A `BlockDevice` implementation backed by a regular file or block device node.
///
/// On macOS and Linux, block devices (`/dev/diskN`, `/dev/sdX`) expose the
/// same `pread`/`pwrite` interface as regular files when opened with the
/// correct flags.  This struct works for both cases.
///
/// The `sector_count` is determined once at construction by seeking to the end
/// of the file.  For block device nodes on Linux, `ioctl(BLKGETSIZE64)` would
/// be more correct, but `seek(End)` works for raw images and is portable to
/// macOS without additional unsafe code.
struct FileBlockDevice {
    /// The underlying file handle.  Opened with read+write access.
    file: File,

    /// Total number of 512-byte sectors, computed from the file size at open.
    total_sector_count: u64,
}

/// Size of one sector in bytes — must match the kernel's `BlockDevice` contract.
const SECTOR_SIZE_BYTES: usize = 512;

impl FileBlockDevice {
    /// Open `path` as a read-write block device.
    ///
    /// Returns an error if the file cannot be opened or if its size is not a
    /// multiple of 512 bytes (which would indicate a corrupt image).
    fn open(path: &str) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let file_size_in_bytes = file.metadata()?.len();
        if file_size_in_bytes % SECTOR_SIZE_BYTES as u64 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "image size {} is not a multiple of {} bytes",
                    file_size_in_bytes, SECTOR_SIZE_BYTES
                ),
            ));
        }
        let total_sector_count = file_size_in_bytes / SECTOR_SIZE_BYTES as u64;
        Ok(FileBlockDevice { file, total_sector_count })
    }
}

impl BlockDevice for FileBlockDevice {
    fn read_sectors(
        &self,
        start_lba: u64,
        sector_count: u32,
        destination_buffer: &mut [u8],
    ) -> bool {
        let byte_offset = start_lba * SECTOR_SIZE_BYTES as u64;
        let byte_count = sector_count as usize * SECTOR_SIZE_BYTES;
        match self.file.read_at(&mut destination_buffer[..byte_count], byte_offset) {
            Ok(n) if n == byte_count => true,
            _ => false,
        }
    }

    fn write_sectors(
        &self,
        start_lba: u64,
        sector_count: u32,
        source_buffer: &[u8],
    ) -> bool {
        let byte_offset = start_lba * SECTOR_SIZE_BYTES as u64;
        let byte_count = sector_count as usize * SECTOR_SIZE_BYTES;
        match self.file.write_at(&source_buffer[..byte_count], byte_offset) {
            Ok(n) if n == byte_count => true,
            _ => false,
        }
    }

    fn total_sector_count(&self) -> u64 {
        self.total_sector_count
    }

    fn device_name(&self) -> &str {
        "FileBlockDevice"
    }
}

// ─── Subcommand implementations ───────────────────────────────────────────────

/// Format the image at `image_path` as a new, empty BAFS volume.
///
/// This is destructive: the first few hundred blocks of the image are
/// overwritten unconditionally.  Always use a dedicated test image.
fn subcommand_format(image_path: &str) -> io::Result<()> {
    let device = FileBlockDevice::open(image_path)?;
    let size_gib = device.total_sector_count * SECTOR_SIZE_BYTES as u64 / (1024 * 1024 * 1024);
    println!("Formatting {} ({} GiB) as BAFS …", image_path, size_gib);
    bafs_format(&device, BafsFormatOptions::default())
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;
    println!("Format complete.");
    Ok(())
}

/// Print the superblock fields of the BAFS volume at `image_path`.
fn subcommand_info(image_path: &str) -> io::Result<()> {
    let device = FileBlockDevice::open(image_path)?;
    let volume = bafs_mount(device)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;

    let superblock = &volume.superblock;
    let block_size = superblock.block_size_in_bytes;
    let total_bytes = superblock.total_block_count * block_size as u64;
    let free_bytes = superblock.free_block_count * block_size as u64;
    let used_bytes = total_bytes.saturating_sub(free_bytes);
    let journal_bytes = superblock.journal_size_in_blocks * block_size as u64;

    println!("BAFS Volume Info");
    println!("────────────────────────────────────────");
    println!("  Version             : {}", superblock.version);
    println!("  Block size          : {} bytes", block_size);
    println!(
        "  Total capacity      : {} ({} blocks)",
        format_bytes(total_bytes),
        superblock.total_block_count
    );
    println!(
        "  Used                : {} ({:.1}%)",
        format_bytes(used_bytes),
        used_bytes as f64 / total_bytes as f64 * 100.0
    );
    println!(
        "  Free                : {}",
        format_bytes(free_bytes)
    );
    println!(
        "  Journal             : {} ({} blocks)",
        format_bytes(journal_bytes),
        superblock.journal_size_in_blocks
    );
    println!(
        "  Allocated inodes    : {}",
        superblock.allocated_inode_count
    );
    println!(
        "  Last transaction ID : {}",
        superblock.last_committed_transaction_id
    );
    println!("  Root inode number   : {}", superblock.root_inode_number);

    bafs_unmount(volume)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;
    Ok(())
}

/// List the contents of a directory inside the BAFS volume.
///
/// `directory_path` is ignored in v1 (only the root directory exists); it is
/// accepted for forward compatibility with the argument interface.
fn subcommand_ls(image_path: &str, _directory_path: &str) -> io::Result<()> {
    let device = FileBlockDevice::open(image_path)?;
    let volume = bafs_mount(device)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;

    let root_inode_number = volume.superblock.root_inode_number;

    // Iterate directory entries using volume_lookup_directory_entry is not
    // enough for listing; we need to iterate.  Use the public
    // directory_iterate helper via a simple scan loop.
    use bafs::dir::directory_iterate;

    let mut index: usize = 0;
    println!("{:<8}  {}", "inode", "name");
    println!("{}", "─".repeat(40));
    loop {
        match directory_iterate(
            &volume.device,
            &volume.dirty_block_cache,
            volume.superblock.inode_tree_root_block,
            root_inode_number,
            index,
        ) {
            Ok(Some(entry)) => {
                let type_char = match entry.child_type {
                    bafs::dir::CHILD_TYPE_DIRECTORY    => 'd',
                    bafs::dir::CHILD_TYPE_REGULAR_FILE => '-',
                    _                                  => '?',
                };
                println!(
                    "{:<8}  {}{}",
                    entry.child_inode_number,
                    type_char,
                    entry.filename
                );
                index += 1;
            }
            Ok(None) => break,
            Err(error) => {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("{:?}", error),
                ));
            }
        }
    }

    if index == 0 {
        println!("(empty directory)");
    }

    bafs_unmount(volume)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;
    Ok(())
}

/// Write a host file (or stdin if `host_source_path` is "-") into the BAFS volume.
///
/// The destination is always created in the root directory.  The name is taken
/// from the last path component of `volume_destination_path`.
fn subcommand_write(
    image_path: &str,
    volume_destination_path: &str,
    host_source_path: &str,
) -> io::Result<()> {
    // Read source data into memory.
    let source_data: Vec<u8> = if host_source_path == "-" {
        let mut buffer = Vec::new();
        io::stdin().read_to_end(&mut buffer)?;
        buffer
    } else {
        std::fs::read(host_source_path)?
    };

    // Extract the filename component from the destination path.
    let filename = Path::new(volume_destination_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(volume_destination_path);

    let device = FileBlockDevice::open(image_path)?;
    let mut volume = bafs_mount(device)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;

    let root_inode_number = volume.superblock.root_inode_number;

    // Create the file (fails if it already exists in v1).
    let file_inode_number = volume_create_file(&mut volume, root_inode_number, filename)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;

    let bytes_written =
        volume_write_file_data(&mut volume, file_inode_number, 0, &source_data)
            .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;

    println!(
        "Wrote {} ({}) to /{} (inode {})",
        host_source_path,
        format_bytes(bytes_written as u64),
        filename,
        file_inode_number
    );

    bafs_unmount(volume)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;
    Ok(())
}

/// Read a file from the BAFS volume and write it to stdout.
fn subcommand_read(image_path: &str, volume_path: &str) -> io::Result<()> {
    // Extract filename.
    let filename = Path::new(volume_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(volume_path);

    let device = FileBlockDevice::open(image_path)?;
    let volume = bafs_mount(device)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;

    let root_inode_number = volume.superblock.root_inode_number;

    let file_inode_number = volume_lookup_directory_entry(&volume, root_inode_number, filename)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("/{} not found", filename)))?;

    // Read file in 4 MiB chunks to handle large files without using excessive RAM.
    let chunk_size: usize = 4 * 1024 * 1024;
    let mut offset: u64 = 0;
    let mut output = io::stdout().lock();

    loop {
        let mut buffer = vec![0u8; chunk_size];
        let bytes_read = volume_read_file_data(&volume, file_inode_number, offset, &mut buffer)
            .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;
        if bytes_read == 0 {
            break;
        }
        output.write_all(&buffer[..bytes_read])?;
        offset += bytes_read as u64;
        if bytes_read < chunk_size {
            break;
        }
    }

    bafs_unmount(volume)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;
    Ok(())
}

/// Sequential write + read benchmark.
///
/// Writes a configurable number of small files, then reads them back,
/// and reports throughput in MiB/s for each phase.
///
/// This exercises the full stack: allocation, B-tree insert, journal commit,
/// journal recovery on remount, B-tree lookup, and extent read.
///
/// Note: the bench is limited to small files per commit because the v1 journal
/// logs full block data (both metadata and file data).  A future version will
/// separate data writes from metadata journaling, lifting this restriction.
fn subcommand_bench(image_path: &str) -> io::Result<()> {
    // Number of 1 MiB files to write.  Chosen so the benchmark finishes in
    // a few seconds even on a slow spinning disk image.
    const FILE_COUNT: usize = 64;
    // File size is kept below the journal capacity so that each per-file commit
    // fits.  The journal is ~1% of the disk; on a 512 MiB disk that is ~5 MiB.
    // A single file with its B-tree metadata uses roughly 300 KiB of journal
    // space, so 256 KiB per file is safe with headroom.
    const FILE_SIZE_BYTES: usize = 256 * 1024; // 256 KiB per file
    // Commit after every file to keep the dirty-block cache bounded.
    const FILES_PER_COMMIT: usize = 1;

    let total_bytes = FILE_COUNT as u64 * FILE_SIZE_BYTES as u64;
    println!(
        "BAFS bench: writing {} × 256 KiB files ({} total) …",
        FILE_COUNT,
        format_bytes(total_bytes)
    );

    // ── Write phase ───────────────────────────────────────────────────────────

    let device = FileBlockDevice::open(image_path)?;
    let mut volume = bafs_mount(device)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;

    let root_inode_number = volume.superblock.root_inode_number;

    // Build a 1 MiB payload with a recognisable pattern so we can verify it.
    let payload: Vec<u8> = (0..FILE_SIZE_BYTES).map(|index| (index & 0xFF) as u8).collect();

    let write_start = Instant::now();
    for file_index in 0..FILE_COUNT {
        let filename = format!("bench_{:04}.bin", file_index);
        let file_inode_number =
            volume_create_file(&mut volume, root_inode_number, &filename)
                .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;
        volume_write_file_data(&mut volume, file_inode_number, 0, &payload)
            .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;

        // Commit periodically to keep the dirty-block cache within the journal size.
        if (file_index + 1) % FILES_PER_COMMIT == 0 {
            flush_and_commit(&mut volume)
                .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;
        }
    }
    bafs_unmount(volume)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;

    let write_elapsed = write_start.elapsed();
    let write_mib_per_sec = total_bytes as f64 / (1024.0 * 1024.0) / write_elapsed.as_secs_f64();
    println!(
        "  Write: {:.2} MiB/s  ({:.2} s for {} MiB)",
        write_mib_per_sec,
        write_elapsed.as_secs_f64(),
        total_bytes / (1024 * 1024)
    );

    // ── Read phase (remount to exercise journal recovery) ─────────────────────

    println!("Remounting and reading back …");
    let device2 = FileBlockDevice::open(image_path)?;
    let volume2 = bafs_mount(device2)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;

    let root_inode_number2 = volume2.superblock.root_inode_number;
    let mut read_buffer = vec![0u8; FILE_SIZE_BYTES];
    let mut verified_files: usize = 0;

    let read_start = Instant::now();
    for file_index in 0..FILE_COUNT {
        let filename = format!("bench_{:04}.bin", file_index);
        let file_inode_number =
            volume_lookup_directory_entry(&volume2, root_inode_number2, &filename)
                .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::NotFound, format!("{} not found after remount", filename))
                })?;

        let bytes_read =
            volume_read_file_data(&volume2, file_inode_number, 0, &mut read_buffer)
                .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;

        // Verify payload integrity.
        let expected: Vec<u8> = (0..bytes_read).map(|index| (index & 0xFF) as u8).collect();
        if read_buffer[..bytes_read] != expected[..] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("data corruption in {} after remount", filename),
            ));
        }
        verified_files += 1;
    }

    let read_elapsed = read_start.elapsed();
    let read_mib_per_sec = total_bytes as f64 / (1024.0 * 1024.0) / read_elapsed.as_secs_f64();
    println!(
        "  Read:  {:.2} MiB/s  ({:.2} s for {} MiB)",
        read_mib_per_sec,
        read_elapsed.as_secs_f64(),
        total_bytes / (1024 * 1024)
    );
    println!("  Verified {} / {} files — no corruption.", verified_files, FILE_COUNT);

    bafs_unmount(volume2)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{:?}", error)))?;

    Ok(())
}

// ─── Utility ──────────────────────────────────────────────────────────────────

/// Format a byte count as a human-readable string (GiB / MiB / KiB / B).
fn format_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;
    const KIB: u64 = 1024;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// ─── Entry point ──────────────────────────────────────────────────────────────

fn print_usage() {
    eprintln!("bafs-tools — BAFS developer CLI\n");
    eprintln!("USAGE:");
    eprintln!("  bafs-tools format  <image>                  Format image as BAFS");
    eprintln!("  bafs-tools info    <image>                  Print superblock info");
    eprintln!("  bafs-tools ls      <image> [path]           List root directory");
    eprintln!("  bafs-tools write   <image> <dest> <src>     Write host file into volume");
    eprintln!("  bafs-tools read    <image> <path>           Read file to stdout");
    eprintln!("  bafs-tools bench   <image>                  Write+read throughput benchmark");
    eprintln!();
    eprintln!("SAFE WORKFLOW (disk image — no hardware risk):");
    eprintln!("  dd if=/dev/zero of=/tmp/bafs.img bs=1m count=1024");
    eprintln!("  bafs-tools format /tmp/bafs.img");
    eprintln!("  bafs-tools info   /tmp/bafs.img");
    eprintln!("  bafs-tools write  /tmp/bafs.img /hello.txt /etc/hostname");
    eprintln!("  bafs-tools ls     /tmp/bafs.img");
    eprintln!("  bafs-tools read   /tmp/bafs.img /hello.txt");
    eprintln!("  bafs-tools bench  /tmp/bafs.img");
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();

    if arguments.len() < 3 {
        print_usage();
        std::process::exit(1);
    }

    let subcommand = arguments[1].as_str();
    let image_path = arguments[2].as_str();

    let result = match subcommand {
        "format" => subcommand_format(image_path),
        "info"   => subcommand_info(image_path),
        "ls"     => {
            let directory_path = arguments.get(3).map(String::as_str).unwrap_or("/");
            subcommand_ls(image_path, directory_path)
        }
        "write" => {
            if arguments.len() < 5 {
                eprintln!("error: write requires <image> <dest> <src>");
                std::process::exit(1);
            }
            subcommand_write(image_path, &arguments[3], &arguments[4])
        }
        "read" => {
            if arguments.len() < 4 {
                eprintln!("error: read requires <image> <path>");
                std::process::exit(1);
            }
            subcommand_read(image_path, &arguments[3])
        }
        "bench" => subcommand_bench(image_path),
        unknown => {
            eprintln!("error: unknown subcommand '{}'", unknown);
            print_usage();
            std::process::exit(1);
        }
    };

    if let Err(error) = result {
        eprintln!("error: {}", error);
        std::process::exit(1);
    }
}
