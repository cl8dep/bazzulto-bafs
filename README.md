# bazzulto-bafs

BAFS (Bazzulto File System) — native filesystem driver for the [Bazzulto OS](https://github.com/cl8dep/bazzulto-os).

## Overview

BAFS is a modern Copy-on-Write filesystem designed for Bazzulto. It provides:

- **Copy-on-Write** — no block is ever overwritten in-place
- **Checksums on all data and metadata** — silent corruption is detected on every read
- **Atomic transactions** — power loss never produces a partially-written state
- **B-tree structures** — O(log n) lookups, scalable to millions of files
- **Snapshots and clones** (v2)
- **Transparent compression** (v3)
- **Per-volume and per-file encryption** (v3/v4)

## Full Specification

See [BAFS.md](https://github.com/cl8dep/bazzulto-os/wiki/BAFS) in the Bazzulto OS wiki.

## Repository structure

```
src/
├── lib.rs          # crate root, feature gates
├── superblock.rs   # superblock read/write, version check
├── btree.rs        # B-tree node format, insert, lookup, split, merge
├── inode.rs        # inode struct, inode tree operations
├── extent.rs       # extent allocator, free-extent tree
├── journal.rs      # write-ahead log, commit, crash recovery
├── checksum.rs     # CRC32C implementation and checksum tree
├── dir.rs          # directory B-tree, lookup, create, unlink
└── kernel.rs       # VFS Inode trait impl (feature = "kernel")
```

## Features

- `kernel` — compile as a kernel module implementing the Bazzulto VFS `Inode` trait (`no_std + alloc`)
- `userspace` — compile with `std` for use in `bafs-mkfs`, `bafs-fsck`, and host-side tests

## License

MIT

