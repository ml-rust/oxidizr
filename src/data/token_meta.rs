//! The `.tokens.json` sidecar beside a token file that `oxidizr tokenize` writes.
//!
//! A `.bin` token file has no header. The sidecar records what produced it:
//! the tokenizer, the separator and BOS ids, the token and document counts,
//! a full-file hash, and every input file with its own hash.
//!
//! The sidecar path appends [`TOKEN_META_SUFFIX`] to the token file path:
//! `out.bin` -> `out.bin.tokens.json`. The sidecar holds no document offsets.

use crate::data_hash::HashAlgo;
use crate::prelude::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The sidecar format version this build writes and the newest it reads.
pub const TOKEN_META_VERSION: u32 = 1;

/// Appended to a token file path to name its sidecar.
pub const TOKEN_META_SUFFIX: &str = ".tokens.json";

/// The tokenizer behind a token file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TokenizerRecord {
    /// A splintr vocabulary name, such as `cl100k_base`.
    Name {
        /// The name as `--tokenizer` gave it.
        name: String,
    },
    /// A `tokenizer.json` file.
    File {
        /// The path as `--tokenizer` gave it.
        path: String,
        /// FNV-1a 64 of the file bytes, as 16 hex digits.
        hash: String,
    },
}

/// One input file of a token file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputRecord {
    /// The path as discovered from `-i`.
    pub path: String,
    /// `txt` or `jsonl`: how the file was read.
    pub format: String,
    /// File size in bytes.
    pub bytes: u64,
    /// FNV-1a 64 of the file bytes, as 16 hex digits.
    pub hash: String,
}

/// The `tokenize` options that decide the output bytes.
///
/// `--batch-bytes` is absent: the output does not depend on it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenOptions {
    /// `--format`: `auto`, `txt` or `jsonl`.
    pub format: String,
    /// `--field`: the JSONL field that holds the text.
    pub field: String,
    /// `--txt-docs`: `file` or `blank-line`.
    pub txt_docs: String,
    /// `--allow-special`: text spelling a special token encodes to its id.
    pub allow_special: bool,
}

/// The sidecar of one token file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenMeta {
    /// [`TOKEN_META_VERSION`] at write time.
    pub format_version: u32,
    /// The oxidizr version that wrote the file.
    pub oxidizr_version: String,
    /// The splintr version that encoded the text, or `unknown`.
    pub splintr_version: String,
    /// The tokenizer that encoded the text.
    pub tokenizer: TokenizerRecord,
    /// The id after every document: the tokenizer's EOS id.
    pub separator_id: u32,
    /// The id before every document's content, when the tokenizer places one.
    pub bos_id: Option<u32>,
    /// The tokenizer's vocabulary size: its highest id plus one.
    pub tokenizer_vocab_size: u64,
    /// The highest id in the file. `None` for a file of zero tokens.
    pub max_token_id: Option<u32>,
    /// Tokens in the file. The file holds `4 * num_tokens` bytes.
    pub num_tokens: u64,
    /// Documents in the file, empty documents excluded.
    pub num_documents: u64,
    /// Empty documents in the inputs. None of them reached the file.
    pub skipped_empty_documents: u64,
    /// The algorithm behind every hash in this sidecar.
    pub hash_algo: HashAlgo,
    /// Hash of every byte of the token file, as 16 hex digits.
    pub file_hash: String,
    /// Input files in the order they were read.
    pub inputs: Vec<InputRecord>,
    /// The options that decide the output bytes.
    pub options: TokenOptions,
}

/// The sidecar path of a token file: `out.bin` -> `out.bin.tokens.json`.
pub fn token_meta_path(bin_path: &Path) -> PathBuf {
    let mut name = bin_path.as_os_str().to_owned();
    name.push(TOKEN_META_SUFFIX);
    PathBuf::from(name)
}

/// Writes `meta` beside `bin_path` atomically.
///
/// # Errors
/// Returns an error naming the sidecar path when the write fails.
pub fn write_token_meta(bin_path: &Path, meta: &TokenMeta) -> Result<()> {
    let path = token_meta_path(bin_path);
    let json = serde_json::to_string_pretty(meta).map_err(|e| Error::DataError {
        reason: format!("failed to encode token sidecar {}: {e}", path.display()),
    })?;
    boostr::format::write_atomic(&path, |w| w.write_all(json.as_bytes())).map_err(|e| {
        Error::DataError {
            reason: format!(
                "failed to write token sidecar {}: {e}. Check that the directory is writable, \
                 then run `oxidizr tokenize` again.",
                path.display()
            ),
        }
    })
}

/// Reads the sidecar beside `bin_path`, if one exists.
///
/// `Ok(None)` means no sidecar. A sidecar that exists but does not parse is an
/// error, and so is one from a newer format version.
pub fn read_token_meta(bin_path: &Path) -> Result<Option<TokenMeta>> {
    let path = token_meta_path(bin_path);
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path).map_err(|e| Error::DataError {
        reason: format!("failed to read token sidecar {}: {e}", path.display()),
    })?;
    parse_token_meta(&text, &path).map(Some)
}

/// Parses sidecar text. The version is checked before the other fields.
fn parse_token_meta(text: &str, path: &Path) -> Result<TokenMeta> {
    let bad = |e: serde_json::Error| Error::DataError {
        reason: format!(
            "token sidecar {} does not parse: {e}. Run `oxidizr tokenize` again to rewrite it.",
            path.display()
        ),
    };
    let value: serde_json::Value = serde_json::from_str(text).map_err(bad)?;
    let version = value
        .get("format_version")
        .and_then(serde_json::Value::as_u64);
    if let Some(v) = version.filter(|&v| v > u64::from(TOKEN_META_VERSION)) {
        return Err(Error::DataError {
            reason: format!(
                "token sidecar {} has format_version {v}. This oxidizr reads version {} \
                 and older. Upgrade oxidizr.",
                path.display(),
                TOKEN_META_VERSION
            ),
        });
    }
    serde_json::from_value(value).map_err(bad)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> TokenMeta {
        TokenMeta {
            format_version: TOKEN_META_VERSION,
            oxidizr_version: "0.1.0".to_owned(),
            splintr_version: "0.22.0".to_owned(),
            tokenizer: TokenizerRecord::File {
                path: "tok/tokenizer.json".to_owned(),
                hash: "0123456789abcdef".to_owned(),
            },
            separator_id: 7,
            bos_id: Some(6),
            tokenizer_vocab_size: 8,
            max_token_id: Some(7),
            num_tokens: 12,
            num_documents: 3,
            skipped_empty_documents: 1,
            hash_algo: HashAlgo::Fnv1a64,
            file_hash: "fedcba9876543210".to_owned(),
            inputs: vec![InputRecord {
                path: "a.jsonl".to_owned(),
                format: "jsonl".to_owned(),
                bytes: 40,
                hash: "00000000000000ff".to_owned(),
            }],
            options: TokenOptions {
                format: "auto".to_owned(),
                field: "text".to_owned(),
                txt_docs: "file".to_owned(),
                allow_special: false,
            },
        }
    }

    #[test]
    fn the_sidecar_path_appends_the_suffix() {
        assert_eq!(
            token_meta_path(Path::new("data/out.bin")),
            PathBuf::from("data/out.bin.tokens.json")
        );
        assert_eq!(
            token_meta_path(Path::new("out")),
            PathBuf::from("out.tokens.json")
        );
    }

    #[test]
    fn a_sidecar_round_trips_through_its_text() {
        let meta = sample();
        let text = serde_json::to_string_pretty(&meta).expect("encode");
        let back = parse_token_meta(&text, Path::new("x.tokens.json")).expect("parse");
        assert_eq!(back, meta);
        assert!(text.contains(r#""kind": "file""#), "{text}");
        assert!(text.contains(r#""hash_algo": "fnv1a64""#), "{text}");
    }

    #[test]
    fn a_newer_version_is_refused_by_number() {
        let mut value = serde_json::to_value(sample()).expect("encode");
        value["format_version"] = serde_json::json!(TOKEN_META_VERSION + 1);
        let message = parse_token_meta(&value.to_string(), Path::new("x.tokens.json"))
            .expect_err("a newer version must fail")
            .to_string();
        assert!(message.contains("format_version 2"), "{message}");
        assert!(message.contains("x.tokens.json"), "{message}");
    }

    #[test]
    fn a_broken_sidecar_names_its_path() {
        let message = parse_token_meta("{", Path::new("y.tokens.json"))
            .expect_err("broken json must fail")
            .to_string();
        assert!(message.contains("y.tokens.json"), "{message}");
    }
}
