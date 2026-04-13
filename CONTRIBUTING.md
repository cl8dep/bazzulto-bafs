# Contributing to BAFS

Thank you for your interest in contributing to the Bazzulto File System.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch: `git checkout -b my-feature`
4. Make your changes
5. Run the tests (see below)
6. Commit with a clear message
7. Push and open a Pull Request

## Running Tests

Tests must be run from **outside** the kernel tree to avoid inheriting the kernel's `no_std` cargo config:

```bash
cd /tmp
cargo test --manifest-path /path/to/bafs/Cargo.toml
```

All tests must pass before submitting a PR. The CI workflow runs the same command.

### Verifying the kernel build

```bash
cd /tmp
cargo build --manifest-path /path/to/bafs/Cargo.toml \
    --features kernel --target aarch64-unknown-none \
    -Zbuild-std=core,alloc,compiler_builtins \
    -Zbuild-std-features=compiler-builtins-mem
```

## Code Style

- Use full, verbose names for files, functions, macros, and variables. Never abbreviate.
- Prefer named constants over raw literals.
- Keep functions focused and small.
- Document public API items with doc comments.
- Add comments only where the logic is not self-evident.

## Commit Messages

- Start with a verb in imperative mood: "Fix", "Add", "Remove", "Update"
- First line under 72 characters
- Use the body to explain *why*, not *what* (the diff shows what changed)

## What to Contribute

### Good first issues

- Adding tests for uncovered edge cases
- Improving documentation and doc comments
- Performance measurements and benchmarks

### Architecture changes

For changes that affect on-disk format, B-tree layout, journal protocol, or the public API, please open an issue first to discuss the approach. These changes have broad impact and benefit from early feedback.

### Bug reports

When reporting a bug, include:

- Steps to reproduce
- Expected behavior
- Actual behavior
- Test output (preferably a minimal failing test case)

## Testing Guidelines

- Every bug fix should include a regression test
- New features should include tests covering the golden path and at least one edge case
- Tests use `MemoryDisk` (in-memory block device) or `FaultInjectionDisk` (crash simulation)
- Flush periodically with `flush_and_commit` in tests that create many files to stay within journal capacity

## License

By contributing, you agree that your contributions will be licensed under the [GNU General Public License v3.0](LICENSE).
