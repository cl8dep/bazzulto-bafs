# BAFS — Bazzulto File System

[![CI](https://github.com/cl8dep/bazzulto-bafs/actions/workflows/ci.yml/badge.svg)](https://github.com/cl8dep/bazzulto-bafs/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Rust](https://img.shields.io/badge/rust-nightly-orange.svg)
![no_std](https://img.shields.io/badge/no__std-compatible-green.svg)

A modern **Copy-on-Write filesystem** designed for the [Bazzulto OS](https://github.com/cl8dep/bazzulto-os) kernel. Zero external dependencies. Runs on `no_std + alloc` for the kernel and on `std` for host-side tooling and tests.

## Motivation

Bazzulto OS needed a filesystem that:

- **Never corrupts data on power loss** — every write is atomic via Copy-on-Write semantics and journal-backed commit
- **Detects silent bit-rot** — CRC32C checksums on every data block and every B-tree metadata node
- **Scales beyond toy workloads** — B-tree structures provide O(log n) lookups, tested with 1000+ files
- **Has zero external dependencies** — every algorithm (CRC32C, xxHash-64, B-tree) is hand-rolled, keeping the kernel build hermetic with no network access or vendored code
- **Is testable in isolation** — the `BlockDevice` trait abstraction lets the entire filesystem run against in-memory disks, fault-injecting disks, and real disk images

BAFS draws inspiration from btrfs (CoW B-trees, extent-based allocation, checksums) while keeping the design focused on correctness and simplicity over feature breadth.

## Features

### Implemented (v1.0)

- **Copy-on-Write** — no block is ever overwritten in-place; old data remains valid until the superblock atomically commits the new tree root
- **CRC32C checksums** on all data blocks and metadata nodes — corruption is detected on every read
- **Atomic transactions** via journal-backed commit — power loss never produces a partially-written state
- **B-tree storage** for inodes, directory entries, data extents, checksums, and free extents — all in a single unified format
- **Extent-based allocation** with free-extent coalescing — contiguous block runs are tracked efficiently
- **Full space reclamation on delete** — unlink frees data extents, removes inode and extent records, updates free block count
- **Superblock backup** — primary superblock corruption automatically falls back to the backup copy
- **Journal crash recovery** — incomplete transactions are detected and replayed on mount
- **Host-side developer CLI** (`bafs-tools`) — format, inspect, read/write files, and benchmark BAFS volumes on macOS and Linux without QEMU

### Planned

- **Snapshots and clones** (v2)
- **Fine-grained locking for SMP** (v2)
- **Transparent compression** (v3)
- **Per-volume and per-file encryption** (v3/v4)

## Repository Structure

```
src/
  lib.rs            Crate root, feature gates
  superblock.rs     Superblock layout, read/write, backup fallback
  btree.rs          B-tree node format, insert, lookup, delete, split
  inode.rs          Inode struct (128 bytes), inode tree operations
  extent.rs         Extent allocator, free-extent tree, block freeing
  dir.rs            Directory entries, lookup, create, unlink
  journal.rs        Write-ahead log, commit, crash recovery
  checksum.rs       CRC32C implementation (hand-rolled, no dependencies)
  checksum_tree.rs  Per-block checksum storage and verification
  block_device.rs   BlockDevice trait (the hardware abstraction boundary)
  error.rs          Error types
  volume.rs         High-level API: format, mount, unmount, file/dir ops
  kernel.rs         VFS Inode trait impl (feature = "kernel")
tests/
  integration_test.rs  34 comprehensive tests (see Testing below)
bafs-tools/
  src/main.rs       Host-side CLI for disk images
```

## Building

### Run the tests (macOS / Linux)

```bash
# Run from OUTSIDE the kernel tree to avoid the kernel's cargo config:
cd /tmp
cargo test --manifest-path /path/to/bafs/Cargo.toml
```

### Build for the kernel (no_std, AArch64)

```bash
cargo build --features kernel --target aarch64-unknown-none \
    -Zbuild-std=core,alloc,compiler_builtins \
    -Zbuild-std-features=compiler-builtins-mem
```

### Use bafs-tools on a disk image

```bash
# Create a 1 GiB raw image
dd if=/dev/zero of=/tmp/bafs.img bs=1m count=1024

# Format, inspect, write, list, read, benchmark
cargo run -p bafs-tools --manifest-path /path/to/bafs/Cargo.toml --release -- \
    format /tmp/bafs.img

cargo run -p bafs-tools ... -- info   /tmp/bafs.img
cargo run -p bafs-tools ... -- write  /tmp/bafs.img /hello.txt /etc/hostname
cargo run -p bafs-tools ... -- ls     /tmp/bafs.img
cargo run -p bafs-tools ... -- read   /tmp/bafs.img /hello.txt
cargo run -p bafs-tools ... -- bench  /tmp/bafs.img
```

## Feature Flags

| Flag | Mode | Use case |
|------|------|----------|
| `kernel` | `no_std + alloc` | Kernel module implementing the Bazzulto VFS `Inode` trait |
| `userspace` | `std` | Host-side tooling (`bafs-tools`, `bafs-mkfs`, `bafs-fsck`) |
| _(none)_ | `std` | Library-only: core structures and algorithms, used by tests |

## Testing

The test suite includes 34 integration tests and 5 unit tests covering:

| Category | Tests | What it validates |
|----------|-------|-------------------|
| Disk size boundaries | 2 | Minimum viable disk, too-small rejection |
| Large disk (10 GiB) | 2 | 1 MiB write/read, 1000 files with remount |
| Basic correctness | 7 | Empty file, overwrite, partial read, 256 KiB file, nested dirs, max filename |
| Multi-session persistence | 3 | Three mount cycles, flush_and_commit checkpoint, free-space accounting |
| Directory / B-tree stress | 2 | 500 files survive remount, bulk unlink of 250 |
| Crash simulation | 4 | FaultInjectionDisk: crash during format, before commit, mid-journal, mid-delete |
| Corruption detection | 2 | Bit-flip in data block detected by CRC32C, bit-flip in checksum |
| Unlink correctness | 3 | Lookup-after-unlink, unlink-nonexistent, create-unlink-recreate |
| Free-space accounting | 1 | free_block_count tracks writes and deletes accurately |
| Journal recovery | 1 | Committed journal data restored after simulated incomplete flush |
| Superblock backup | 1 | Primary zeroed, mount succeeds via backup |
| CoW recycling | 2 | No OutOfSpace over many sessions, cross-session data isolation |
| Space reclamation | 1 | Unlink 200 files, verify blocks freed, refill same space |
| Hash / checksum sanity | 3 | CRC32C test vectors, xxHash-64 reference value |

All tests use in-memory `MemoryDisk` and `FaultInjectionDisk` implementations of the `BlockDevice` trait.

## Comparison with Other Filesystems

| Feature | BAFS v1 | ext4 | btrfs | APFS | ZFS |
|:--------|:-------:|:----:|:-----:|:----:|:---:|
| Copy-on-Write              | :white_check_mark: | :x: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Data checksums             | :white_check_mark: CRC32C | :x: | :white_check_mark: CRC32C | :white_check_mark: Fletcher-64 | :white_check_mark: SHA-256 |
| Metadata checksums         | :white_check_mark: CRC32C | :white_check_mark: CRC32C | :white_check_mark: CRC32C | :white_check_mark: Fletcher-64 | :white_check_mark: Fletcher-4 |
| Atomic transactions        | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| B-tree indexed             | :white_check_mark: | Partial (H-tree) | :white_check_mark: | :white_check_mark: | :x: (DMU+ZAP) |
| Extent-based allocation    | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Space reclaim on delete    | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Snapshots                  | :clock3: v2 | :x: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Compression                | :clock3: v3 | :x: | :white_check_mark: zstd | :white_check_mark: lzfse | :white_check_mark: zstd |
| Encryption                 | :clock3: v3 | :white_check_mark: fscrypt | :x: | :white_check_mark: | :white_check_mark: |
| Deduplication              | :x: | :x: | :white_check_mark: | :white_check_mark: Clones | :white_check_mark: |
| RAID                       | :x: | :x: (mdraid) | :white_check_mark: Native | :x: (Fusion) | :white_check_mark: RAID-Z |
| Max file size              | 16 EiB | 16 TiB | 16 EiB | 8 EiB | 16 EiB |
| Max volume size            | 64 ZiB | 1 EiB | 16 EiB | 8 EiB | 256 ZiB |
| Zero external dependencies | **:white_check_mark:** | :x: | :x: | :x: | :x: |
| `no_std` compatible        | **:white_check_mark:** | :x: | :x: | :x: | :x: |
| Lines of code              | **~3,500** | ~60K | ~170K | Proprietary | ~600K |

> :white_check_mark: = supported&ensp; :x: = not supported&ensp; :clock3: = planned

BAFS is intentionally minimal: it prioritizes correctness and `no_std` portability over feature breadth. It targets embedded and OS kernels where zero dependencies and auditability matter more than petabyte-scale features.

## On-Disk Layout

```
Block 0        Reserved (boot sector)
Block 1        Primary superblock (512 bytes, CRC32C protected)
Block 2        Backup superblock
Blocks 3..N    Journal area (1% of disk, min 256 blocks, max 16384)
Blocks N..end  Data area (B-tree nodes, file data, free extents)
```

Each B-tree node occupies one 4 KiB block:

```
Offset  Size  Field
     0     4  CRC32C checksum (covers bytes 4..4096)
     4     1  Level (0 = leaf, 1+ = internal)
     5     1  Flags (reserved)
     6     2  Item count
     8     8  Self block address
    16     8  Generation (transaction ID)
    24   ...  Items (leaf: key+value pairs; internal: key+child pointers)
```

## Architecture

BAFS uses a **single unified B-tree format** for all persistent data:

- **Inode tree** — file/directory metadata (128-byte inodes), directory entries, data extent mappings
- **Free-extent tree** — tracks unallocated block runs for recycling
- **Checksum tree** — per-block CRC32C values for data integrity verification

All tree modifications use Copy-on-Write: modified nodes are written to newly allocated blocks, and the tree root is updated atomically via the superblock. The journal provides crash recovery for the window between writing new blocks and updating the superblock.

## API Overview

```rust
// Format a new volume
bafs_format(&device, BafsFormatOptions::default())?;

// Mount and operate
let mut vol = bafs_mount(device)?;
let inode = volume_create_file(&mut vol, root, "hello.txt")?;
volume_write_file_data(&mut vol, inode, 0, b"Hello, BAFS!")?;

// Read back
let mut buf = vec![0u8; 12];
volume_read_file_data(&vol, inode, 0, &mut buf)?;

// Directory operations
let entry = volume_lookup_directory_entry(&vol, root, "hello.txt")?;
volume_unlink_directory_entry(&mut vol, root, "hello.txt")?;

// Commit and unmount
bafs_unmount(vol)?;
```

## License

[GNU General Public License v3.0](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
