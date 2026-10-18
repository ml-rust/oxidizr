//! A minimal TensorBoard event file writer for scalar summaries.
//!
//! An event file is a TFRecord stream. Each record is:
//! `u64 length`, `u32 masked CRC32C of the length`, `data`, `u32 masked CRC32C of data`,
//! all little-endian. Each `data` is one `tensorflow.Event` protobuf:
//!
//! - `Event`: `wall_time` (1, double), `step` (2, int64), `file_version` (3, string),
//!   `summary` (5, message)
//! - `Summary`: `value` (1, repeated message)
//! - `Summary.Value`: `tag` (1, string), `simple_value` (2, float)
//!
//! The first record of a file holds `file_version: "brain.Event:2"`.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// The version string TensorBoard expects in the first event.
pub const FILE_VERSION: &str = "brain.Event:2";

/// Event files this process opened, so two in one second never share a name.
static OPENED: AtomicU64 = AtomicU64::new(0);

/// An open event file.
pub struct EventWriter {
    file: BufWriter<File>,
    path: PathBuf,
}

impl EventWriter {
    /// Creates `events.out.tfevents.<secs>.oxidizr.<pid>.<n>` in `dir` and
    /// writes its version event. `dir` is created when absent.
    ///
    /// # Errors
    /// Returns an error when the directory or the file cannot be created or written.
    pub fn create(dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let n = OPENED.fetch_add(1, Ordering::Relaxed);
        let name = format!(
            "events.out.tfevents.{secs}.oxidizr.{}.{n}",
            std::process::id()
        );
        let path = dir.join(name);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let mut writer = Self {
            file: BufWriter::new(file),
            path,
        };
        let version = encode_event(wall_time(), 0, &EventBody::FileVersion(FILE_VERSION));
        writer.write_record(&version)?;
        writer.file.flush()?;
        Ok(writer)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Writes one event of `values` at `step` and flushes the file.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn scalars(&mut self, step: u64, values: &[(&str, f64)]) -> io::Result<()> {
        let event = encode_event(wall_time(), step, &EventBody::Scalars(values));
        self.write_record(&event)?;
        self.file.flush()
    }

    fn write_record(&mut self, data: &[u8]) -> io::Result<()> {
        self.file.write_all(&frame_record(data))
    }
}

/// Seconds since the Unix epoch, with fractions.
fn wall_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// What one event carries.
pub enum EventBody<'a> {
    FileVersion(&'a str),
    /// Tags and values. Each value is stored as an f32 `simple_value`.
    Scalars(&'a [(&'a str, f64)]),
}

/// The protobuf bytes of one `Event`.
pub fn encode_event(wall_time: f64, step: u64, body: &EventBody<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    // Field 1, wire type 1 (64-bit).
    out.push(0x09);
    out.extend_from_slice(&wall_time.to_le_bytes());
    // Field 2, wire type 0 (varint).
    out.push(0x10);
    push_varint(&mut out, step);
    match body {
        EventBody::FileVersion(version) => push_bytes_field(&mut out, 3, version.as_bytes()),
        EventBody::Scalars(values) => {
            let mut summary = Vec::new();
            for (tag, value) in values.iter() {
                let mut entry = Vec::new();
                push_bytes_field(&mut entry, 1, tag.as_bytes());
                // Field 2, wire type 5 (32-bit).
                entry.push(0x15);
                entry.extend_from_slice(&(*value as f32).to_le_bytes());
                push_bytes_field(&mut summary, 1, &entry);
            }
            push_bytes_field(&mut out, 5, &summary);
        }
    }
    out
}

/// A length-delimited field: key `(field << 3) | 2`, the length, the bytes.
fn push_bytes_field(out: &mut Vec<u8>, field: u64, bytes: &[u8]) {
    push_varint(out, (field << 3) | 2);
    push_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn push_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// One TFRecord: the framed and checksummed `data`.
pub fn frame_record(data: &[u8]) -> Vec<u8> {
    let len = (data.len() as u64).to_le_bytes();
    let mut out = Vec::with_capacity(data.len() + 16);
    out.extend_from_slice(&len);
    out.extend_from_slice(&masked_crc32c(&len).to_le_bytes());
    out.extend_from_slice(data);
    out.extend_from_slice(&masked_crc32c(data).to_le_bytes());
    out
}

/// The TFRecord checksum: CRC32C rotated right by 15 bits plus `0xa282ead8`.
pub fn masked_crc32c(data: &[u8]) -> u32 {
    crc32c(data).rotate_right(15).wrapping_add(0xa282_ead8)
}

/// Reflected CRC32C (Castagnoli) polynomial.
const CASTAGNOLI: u32 = 0x82f6_3b78;

const CRC_TABLE: [u32; 256] = crc_table();

const fn crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ CASTAGNOLI
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// CRC32C of `data`, as RFC 3720 defines it.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc = CRC_TABLE[((crc ^ byte as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 3720 appendix B.4 vectors, plus the common check string.
    #[test]
    fn crc32c_matches_the_known_vectors() {
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
        assert_eq!(crc32c(&[0u8; 32]), 0x8a91_36aa);
        assert_eq!(crc32c(&[0xffu8; 32]), 0x62a8_ab43);
        let ascending: Vec<u8> = (0u8..32).collect();
        assert_eq!(crc32c(&ascending), 0x46dd_794e);
        let descending: Vec<u8> = (0u8..32).rev().collect();
        assert_eq!(crc32c(&descending), 0x113f_db5c);
    }

    #[test]
    fn varints_use_seven_bits_per_byte() {
        let mut out = Vec::new();
        push_varint(&mut out, 300);
        assert_eq!(out, [0xac, 0x02]);
    }

    /// Bytes worked out by hand from the field table in the module docs.
    #[test]
    fn a_scalar_event_encodes_to_the_hand_computed_bytes() {
        let bytes = encode_event(1.5, 3, &EventBody::Scalars(&[("a", 2.0)]));
        let want: [u8; 23] = [
            0x09, 0, 0, 0, 0, 0, 0, 0xf8, 0x3f, // wall_time 1.5
            0x10, 0x03, // step 3
            0x2a, 0x0a, // summary, 10 bytes
            0x0a, 0x08, // value, 8 bytes
            0x0a, 0x01, b'a', // tag "a"
            0x15, 0, 0, 0, 0x40, // simple_value 2.0
        ];
        assert_eq!(bytes, want);
    }

    #[test]
    fn the_version_event_holds_the_version_string() {
        let bytes = encode_event(0.0, 0, &EventBody::FileVersion(FILE_VERSION));
        let mut want = vec![0x09, 0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x00, 0x1a, 13];
        want.extend_from_slice(FILE_VERSION.as_bytes());
        assert_eq!(bytes, want);
    }

    #[test]
    fn a_record_frames_its_length_and_both_checksums() {
        let framed = frame_record(b"");
        assert_eq!(framed.len(), 16);
        assert_eq!(&framed[..8], &[0u8; 8]);
        assert_eq!(&framed[8..12], &masked_crc32c(&[0u8; 8]).to_le_bytes());
        // CRC32C of no bytes is 0, so its masked form is the mask constant.
        assert_eq!(&framed[12..], &0xa282_ead8u32.to_le_bytes());
        let framed = frame_record(b"xyz");
        assert_eq!(framed.len(), 19);
        assert_eq!(&framed[12..15], b"xyz");
        assert_eq!(&framed[15..], &masked_crc32c(b"xyz").to_le_bytes());
    }
}
