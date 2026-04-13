//! CRC32C (Castagnoli) checksum implementation for BAFS.
//!
//! Every metadata block and every data block stored by BAFS carries a CRC32C
//! checksum.  Metadata blocks embed the checksum in the first four bytes of the
//! node header; data blocks have their checksum stored in the checksum B-tree
//! (see `checksum_tree.rs`).
//!
//! # Why CRC32C instead of CRC32?
//!
//! CRC32C uses the Castagnoli polynomial, which has better error-detection
//! properties for the block sizes used by modern storage (4 KiB – 64 KiB).
//! Modern processors (ARM with the `crc` extension, x86-64 with SSE4.2) can
//! accelerate CRC32C with a single instruction.  This implementation is a
//! portable software fallback that works on all targets without requiring
//! compiler intrinsics.
//!
//! # Impact on the rest of the system
//!
//! - `superblock.rs` calls `compute_crc32c` to protect the superblock.
//! - `btree.rs` calls it to verify/update every B-tree node it reads or writes.
//! - `checksum_tree.rs` calls it to store per-data-block checksums.
//! - `journal.rs` calls it to protect the commit record.

// ─── Lookup table ─────────────────────────────────────────────────────────────
//
// The table is computed at compile time by a const fn so that no runtime
// initialisation is needed.  Each entry `TABLE[byte]` gives the CRC32C
// remainder after processing a single byte value `byte` with all other input
// bits zero.

/// The Castagnoli polynomial in bit-reversed (LSB-first) form.
///
/// In LSB-first CRC computation the polynomial is written in reverse so that
/// the feedback register shifts right instead of left, allowing the same loop
/// to process both individual bits and whole bytes.
///
/// Normal form (MSB-first):  0x1EDC6F41
/// Reversed (LSB-first):     0x82F63B78
///
/// Derivation: reverse all 32 bits of 0x1EDC6F41.
///   0x1EDC6F41 = 0001_1110_1101_1100_0110_1111_0100_0001
///   reversed   = 1000_0010_1111_0110_0011_1011_0111_1000
///             = 0x82F63B78
const CASTAGNOLI_POLYNOMIAL_REVERSED: u32 = 0x82F63B78;

/// Build the 256-entry CRC32C lookup table at compile time.
///
/// Each entry is the CRC32C of a single byte value (padded to 32 bits with
/// zero data following it).  This lets the main loop process one byte per
/// iteration with only an XOR and a table lookup.
const fn build_crc32c_lookup_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut index = 0usize;
    while index < 256 {
        // Start with the byte value in the low 8 bits.
        let mut remainder = index as u32;
        let mut bit = 0usize;
        while bit < 8 {
            // If the lowest bit is set, XOR in the polynomial (the bit has
            // "fallen off" the register and feeds back through the generator).
            if remainder & 1 != 0 {
                remainder = (remainder >> 1) ^ CASTAGNOLI_POLYNOMIAL_REVERSED;
            } else {
                remainder >>= 1;
            }
            bit += 1;
        }
        table[index] = remainder;
        index += 1;
    }
    table
}

/// Pre-computed CRC32C lookup table (256 entries, built at compile time).
static CRC32C_LOOKUP_TABLE: [u32; 256] = build_crc32c_lookup_table();

// ─── Public API ───────────────────────────────────────────────────────────────

/// Compute the CRC32C checksum of `data`.
///
/// The initial value is `0xFFFF_FFFF` (standard CRC32 convention) and the
/// final value is bit-inverted before returning, which matches the CRC32C
/// algorithm used by BTRFS, ext4, iSCSI, and SCTP.
///
/// # Example
/// ```
/// # use bafs::checksum::compute_crc32c;
/// assert_eq!(compute_crc32c(b""), 0x0000_0000);  // empty input
/// assert_eq!(compute_crc32c(b"123456789"), 0xE306_9283); // standard test vector
/// ```
pub fn compute_crc32c(data: &[u8]) -> u32 {
    let mut accumulator: u32 = 0xFFFF_FFFF;
    for &byte in data {
        // XOR the next byte into the low 8 bits, look up the remainder, and
        // shift the old high bits down.  This processes one byte per iteration.
        let table_index = ((accumulator ^ byte as u32) & 0xFF) as usize;
        accumulator = (accumulator >> 8) ^ CRC32C_LOOKUP_TABLE[table_index];
    }
    // Final XOR with all-ones to invert the accumulator (standard convention).
    accumulator ^ 0xFFFF_FFFF
}

/// Verify that `data` matches `expected_checksum`.
///
/// Returns `true` if `compute_crc32c(data) == expected_checksum`, `false`
/// otherwise.  Callers should convert `false` into a
/// `BafsError::InvalidChecksum` rather than panicking.
pub fn verify_crc32c(data: &[u8], expected_checksum: u32) -> bool {
    compute_crc32c(data) == expected_checksum
}

/// Compute the CRC32C of `data` and write it as a little-endian `u32` into the
/// first four bytes of `destination_buffer`.
///
/// This is a convenience wrapper used by B-tree node serialisation: the node
/// header always has the checksum at offset 0, covering bytes 4..block_size.
pub fn write_checksum_into_buffer_header(data_after_header: &[u8], destination_buffer: &mut [u8]) {
    debug_assert!(
        destination_buffer.len() >= 4,
        "destination buffer must be at least 4 bytes to hold the checksum"
    );
    let checksum = compute_crc32c(data_after_header);
    destination_buffer[0..4].copy_from_slice(&checksum.to_le_bytes());
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard CRC32C test vector: the ASCII string "123456789" must
    /// produce 0xE306_9283.  This is published in the CRC32C specification
    /// and used by many implementations as a self-test.
    #[test]
    fn standard_crc32c_test_vector_produces_correct_result() {
        let input = b"123456789";
        let expected = 0xE306_9283u32;
        assert_eq!(compute_crc32c(input), expected);
    }

    /// An empty slice must produce 0x0000_0000 because the initial accumulator
    /// (0xFFFF_FFFF) XORed with the final mask (0xFFFF_FFFF) gives 0.
    #[test]
    fn empty_input_produces_zero_checksum() {
        assert_eq!(compute_crc32c(b""), 0x0000_0000);
    }

    /// Round-trip: compute a checksum then verify it successfully.
    #[test]
    fn verify_accepts_correct_checksum() {
        let data = b"hello, bafs filesystem";
        let checksum = compute_crc32c(data);
        assert!(verify_crc32c(data, checksum));
    }

    /// A one-bit flip in the data must cause verification to fail.
    #[test]
    fn verify_rejects_corrupted_data() {
        let data = b"hello, bafs filesystem";
        let checksum = compute_crc32c(data);
        let mut corrupted = data.to_vec();
        corrupted[0] ^= 0x01; // flip one bit
        assert!(!verify_crc32c(&corrupted, checksum));
    }
}
