//! Stable hashes stored in checkpoint metas.
//!
//! `--resume` compares these hashes, so they must not change between Rust
//! releases. `std`'s `DefaultHasher` gives no such guarantee. Both hashes here
//! use 64-bit FNV-1a, whose output is fixed by its published definition.
//!
//! - Data file: FNV-1a over the first [`HEAD_BYTES`] of the file, then the file
//!   size as a little-endian `u64`.
//! - Model config: FNV-1a over a canonical JSON text with sorted object keys.
//!
//! The `legacy_*` functions reproduce the `DefaultHasher` hashes of metas
//! written without `hash_algo`. Resume checks of such metas call them.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Read;

/// Bytes of a data file that enter its hash. The file size enters too.
pub const HEAD_BYTES: u64 = 1024 * 1024;

/// Read size while hashing a data file head.
const CHUNK_BYTES: usize = 64 * 1024;

/// The algorithm behind the hashes of one checkpoint meta.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HashAlgo {
    /// `std`'s `DefaultHasher`. Its output can change between Rust releases.
    /// A meta without a `hash_algo` key used it.
    Legacy,
    /// 64-bit FNV-1a as this module defines it.
    Fnv1a64,
}

impl HashAlgo {
    /// The algorithm of a meta that has no `hash_algo` key.
    pub fn legacy() -> Self {
        Self::Legacy
    }
}

/// Streaming 64-bit FNV-1a.
#[derive(Debug, Clone, Copy)]
pub struct Fnv1a64 {
    state: u64,
}

impl Fnv1a64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    /// A hasher over no bytes yet.
    pub fn new() -> Self {
        Self {
            state: Self::OFFSET_BASIS,
        }
    }

    /// Feeds `bytes`. Pieces in order give the hash of their concatenation.
    pub fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= u64::from(byte);
            self.state = self.state.wrapping_mul(Self::PRIME);
        }
    }

    /// The hash of every byte fed so far.
    pub fn finish(&self) -> u64 {
        self.state
    }
}

impl Default for Fnv1a64 {
    fn default() -> Self {
        Self::new()
    }
}

/// A hash as 16 lowercase hex digits, the form every stored hash takes.
pub fn hex(value: u64) -> String {
    format!("{value:016x}")
}

/// Hashes the first [`HEAD_BYTES`] of `reader`, then `size` as a little-endian `u64`.
///
/// `size` is the full length of the data, which can exceed the bytes read.
pub fn hash_head_and_size<Rd: Read>(reader: Rd, size: u64) -> std::io::Result<String> {
    let mut hasher = Fnv1a64::new();
    let mut head = reader.take(HEAD_BYTES);
    let mut buffer = vec![0u8; CHUNK_BYTES];
    loop {
        let n = head.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    hasher.update(&size.to_le_bytes());
    Ok(hex(hasher.finish()))
}

/// Opens the data file at `path` and returns it with its size in bytes.
fn open_with_size(path: &str) -> anyhow::Result<(fs::File, u64)> {
    let file = fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("failed to open data file '{path}' to hash it: {e}"))?;
    let size = file
        .metadata()
        .map_err(|e| anyhow::anyhow!("failed to read the size of data file '{path}': {e}"))?
        .len();
    Ok((file, size))
}

/// Hashes the data file at `path`: its first [`HEAD_BYTES`] and its size.
///
/// A multi-GB token file costs one 1 MiB read.
pub fn compute_file_hash(path: &str) -> anyhow::Result<String> {
    let (file, size) = open_with_size(path)?;
    hash_head_and_size(file, size)
        .map_err(|e| anyhow::anyhow!("failed to read data file '{path}' to hash it: {e}"))
}

/// The `DefaultHasher` data file hash, as metas without `hash_algo` stored it.
///
/// It hashes the first [`HEAD_BYTES`] as a byte vector, then the size. Only
/// resume checks of such metas call this.
pub fn legacy_file_hash(path: &str) -> anyhow::Result<String> {
    let (file, size) = open_with_size(path)?;
    let mut head = Vec::new();
    file.take(HEAD_BYTES)
        .read_to_end(&mut head)
        .map_err(|e| anyhow::anyhow!("failed to read data file '{path}' to hash it: {e}"))?;
    let mut hasher = DefaultHasher::new();
    head.hash(&mut hasher);
    size.hash(&mut hasher);
    Ok(hex(hasher.finish()))
}

/// JSON text of `value` with every object's keys sorted.
///
/// The text depends on the value only, never on map ordering in `serde_json`.
pub fn canonical_json(value: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::Value::String(key.clone()).to_string());
                out.push(':');
                if let Some(item) = map.get(key) {
                    write_canonical(item, out);
                }
            }
            out.push('}');
        }
        serde_json::Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

/// The stable hash of a JSON value: FNV-1a over [`canonical_json`].
pub fn hash_json(value: &serde_json::Value) -> String {
    let mut hasher = Fnv1a64::new();
    hasher.update(canonical_json(value).as_bytes());
    hex(hasher.finish())
}

/// The `DefaultHasher` hash of `text`, as metas without `hash_algo` stored it.
///
/// Only resume checks of such metas call this.
pub fn legacy_hash_str(text: &str) -> String {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hex(hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fnv(bytes: &[u8]) -> String {
        let mut hasher = Fnv1a64::new();
        hasher.update(bytes);
        hex(hasher.finish())
    }

    /// Published FNV-1a 64 test vectors.
    #[test]
    fn fnv1a64_matches_the_published_vectors() {
        assert_eq!(fnv(b""), "cbf29ce484222325");
        assert_eq!(fnv(b"a"), "af63dc4c8601ec8c");
        assert_eq!(fnv(b"foobar"), "85944171f73967e8");
    }

    #[test]
    fn updates_in_pieces_equal_one_update() {
        let mut pieces = Fnv1a64::new();
        pieces.update(b"foo");
        pieces.update(b"bar");
        assert_eq!(hex(pieces.finish()), fnv(b"foobar"));
    }

    /// Known answer: two u32 tokens `1, 2`, then the size 8 as a u64.
    #[test]
    fn a_small_token_file_hashes_to_its_known_answer() {
        let bytes = [1u8, 0, 0, 0, 2, 0, 0, 0];
        let got = hash_head_and_size(Cursor::new(bytes), bytes.len() as u64).expect("hash");
        assert_eq!(got, "1b7f497a5b010e8e");
    }

    /// Bytes past the head do not enter the hash. The size does.
    #[test]
    fn only_the_head_and_the_size_enter_a_large_file_hash() {
        let len = HEAD_BYTES as usize + 100;
        let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        let got = hash_head_and_size(Cursor::new(&bytes), len as u64).expect("hash");
        assert_eq!(got, "ce37732f4ada93b0");

        let mut tail_changed = bytes.clone();
        tail_changed[len - 1] ^= 0xff;
        let same = hash_head_and_size(Cursor::new(&tail_changed), len as u64).expect("hash");
        assert_eq!(same, got, "a byte past the head changed the hash");

        let other_size = hash_head_and_size(Cursor::new(&bytes), len as u64 + 4).expect("hash");
        assert_ne!(other_size, got, "the size did not enter the hash");
    }

    #[test]
    fn canonical_json_sorts_keys_at_every_level() {
        let value = serde_json::json!({"b": 1, "a": {"d": [2, {"z": 0, "y": null}], "c": "x"}});
        assert_eq!(
            canonical_json(&value),
            r#"{"a":{"c":"x","d":[2,{"y":null,"z":0}]},"b":1}"#
        );
    }

    #[test]
    fn missing_hash_algo_reads_as_legacy() {
        #[derive(Deserialize)]
        struct Meta {
            #[serde(default = "HashAlgo::legacy")]
            hash_algo: HashAlgo,
        }
        let old: Meta = serde_json::from_str("{}").expect("parse");
        assert_eq!(old.hash_algo, HashAlgo::Legacy);
        let new: Meta = serde_json::from_str(r#"{"hash_algo":"fnv1a64"}"#).expect("parse");
        assert_eq!(new.hash_algo, HashAlgo::Fnv1a64);
    }
}
