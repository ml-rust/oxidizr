//! Text tokenizer selection for the commands that take `--tokenizer`.
//!
//! A value ending in `.json` names a HuggingFace `tokenizer.json`. Any other
//! value is a splintr vocabulary name. A name loads only when its vocabulary
//! is bundled through a `vocab-*` cargo feature. The default build bundles none.

use crate::data::TokenizerRecord;
use crate::data_hash::{hex, Fnv1a64};
use anyhow::{anyhow, Result};
use splintr::{AnyTokenizer, PretrainedVocab, Tokenize, TokenizerError};
use std::path::Path;

/// The vocabulary names that `--tokenizer` help and errors list.
///
/// Any other splintr name loads too when its `vocab-*` feature is on.
pub const TOKENIZER_NAMES: [&str; 4] = ["cl100k_base", "o200k_base", "llama3", "deepseek_v3"];

/// Where a `--tokenizer` value points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerSource<'a> {
    /// A splintr vocabulary name.
    Name(&'a str),
    /// A `tokenizer.json` file that exists.
    JsonFile(&'a Path),
}

impl<'a> TokenizerSource<'a> {
    /// Reads a `--tokenizer` value.
    ///
    /// A `.json` value that names no file is an error. Reading it as a name
    /// would report an unknown vocabulary and hide the typo in the path.
    ///
    /// # Errors
    /// Returns an error naming the value when it ends in `.json` and no file exists there.
    pub fn parse(value: &'a str) -> Result<Self> {
        let path = Path::new(value);
        let is_json = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
        if !is_json {
            return Ok(Self::Name(value));
        }
        if !path.is_file() {
            return Err(anyhow!(
                "--tokenizer '{value}' ends in .json but names no file. Pass a tokenizer.json \
                 path that exists, or one of the vocabulary names: {}.",
                TOKENIZER_NAMES.join(", ")
            ));
        }
        Ok(Self::JsonFile(path))
    }
}

/// A loaded tokenizer with the ids a token stream needs.
pub struct LoadedTokenizer {
    /// The splintr handle. Its policy places [`Self::bos_id`] before the content.
    pub tokenizer: AnyTokenizer,
    /// The end-of-sequence id. It follows every document.
    pub eos_id: u32,
    /// The id the policy places before every document's content, if any.
    pub bos_id: Option<u32>,
    /// Whether the policy already places `eos_id` after the content.
    pub template_ends_with_eos: bool,
    /// The vocabulary size: the highest id the tokenizer can emit, plus one.
    pub vocab_size: u64,
    /// The name, or the path with its hash, for the sidecar.
    pub record: TokenizerRecord,
}

/// Loads the tokenizer a `--tokenizer` value names.
///
/// - A name gets splintr's BOS for that vocabulary: `llama3` and `deepseek_v3`
///   have one, `cl100k_base` and `o200k_base` have none. The EOS id is
///   splintr's EOS for that vocabulary.
/// - A `tokenizer.json` keeps its own post-processor template. Its EOS id is
///   the one splintr reads from its added tokens.
///
/// # Errors
/// Returns an error for an unknown name, a vocabulary this build does not
/// bundle, an unreadable or invalid file, or a file without an EOS token.
pub fn load_tokenizer(value: &str) -> Result<LoadedTokenizer> {
    match TokenizerSource::parse(value)? {
        TokenizerSource::Name(name) => load_name(name),
        TokenizerSource::JsonFile(path) => load_json(path),
    }
}

fn load_name(name: &str) -> Result<LoadedTokenizer> {
    let vocab = PretrainedVocab::from_name(name).ok_or_else(|| {
        anyhow!(
            "--tokenizer '{name}' is not a splintr vocabulary name. Pass one of {}, or a \
             tokenizer.json path. A value is read as a path only when it ends in .json.",
            TOKENIZER_NAMES.join(", ")
        )
    })?;
    let tokenizer = splintr::from_vocab(vocab).map_err(|e| {
        let missing_feature = match &e {
            TokenizerError::VocabNotBundled(_, feature) => Some(*feature),
            _ => None,
        };
        match missing_feature {
            Some(feature) => anyhow!(
                "--tokenizer '{name}': {e}. Rebuild oxidizr with `--features {feature}`, or \
                 pass a tokenizer.json path."
            ),
            None => anyhow!(
                "--tokenizer '{name}' did not load: {e}. Pass a tokenizer.json path instead."
            ),
        }
    })?;
    let tokenizer = tokenizer.with_leading_bos(splintr::bos_token_id(vocab));
    let record = TokenizerRecord::Name {
        name: name.to_owned(),
    };
    Ok(finish(tokenizer, splintr::eos_token_id(vocab), record))
}

fn load_json(path: &Path) -> Result<LoadedTokenizer> {
    let shown = path.display();
    let bytes = std::fs::read(path).map_err(|e| {
        anyhow!("--tokenizer '{shown}' could not be read: {e}. Check the path and its permissions.")
    })?;
    let mut hasher = Fnv1a64::new();
    hasher.update(&bytes);
    let tokenizer = splintr::from_json_bytes(&bytes).map_err(|e| {
        anyhow!(
            "--tokenizer '{shown}' is not a tokenizer.json splintr can load: {e}. Pass the \
             tokenizer.json of a HuggingFace checkpoint."
        )
    })?;
    let eos_id = tokenizer.eos_token_id().ok_or_else(|| {
        anyhow!(
            "--tokenizer '{shown}' names no end-of-sequence token. splintr looks for </s>, \
             <eos>, <|endoftext|>, <|end_of_text|> or [SEP] among its added tokens. Pass a \
             tokenizer.json that declares one."
        )
    })?;
    let record = TokenizerRecord::File {
        path: shown.to_string(),
        hash: hex(hasher.finish()),
    };
    Ok(finish(tokenizer, eos_id, record))
}

fn finish(tokenizer: AnyTokenizer, eos_id: u32, record: TokenizerRecord) -> LoadedTokenizer {
    let policy = tokenizer.policy();
    let bos_id = policy.leading_special();
    // One sentinel content id shows what the template places after the content.
    let template_ends_with_eos = policy.apply_single(vec![u32::MAX]).last() == Some(&eos_id);
    let vocab_size = Tokenize::vocab_size(&tokenizer) as u64;
    LoadedTokenizer {
        tokenizer,
        eos_id,
        bos_id,
        template_ends_with_eos,
        vocab_size,
        record,
    }
}

/// A byte-level BPE `tokenizer.json` over `a`, `b` and `Ġ`, with
/// `<|endoftext|>` as id 3. Unit tests encode text from those pieces only.
#[cfg(test)]
pub(crate) const TINY_TOKENIZER_JSON: &str = r#"{
    "added_tokens": [{"id": 3, "content": "<|endoftext|>", "special": true}],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
    "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "Ġ": 2}, "merges": []}
}"#;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch_file(name: &str, text: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("oxidizr-tokenizer-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, text).expect("write tokenizer");
        path
    }

    fn remove_scratch(path: &Path) {
        let dir = path.parent().expect("a scratch file has a directory");
        std::fs::remove_dir_all(dir).expect("remove scratch dir");
    }

    #[test]
    fn a_bare_value_is_a_name() {
        assert_eq!(
            TokenizerSource::parse("cl100k_base").expect("parse"),
            TokenizerSource::Name("cl100k_base")
        );
    }

    #[test]
    fn a_missing_json_path_is_not_read_as_a_name() {
        let message = TokenizerSource::parse("/no/such/dir/tokenizer.json")
            .expect_err("a .json naming no file must fail")
            .to_string();
        assert!(message.contains("/no/such/dir/tokenizer.json"), "{message}");
    }

    #[test]
    fn an_unknown_name_lists_the_names() {
        let message = load_tokenizer("no_such_vocab").err().map(|e| e.to_string());
        let message = message.expect("an unknown name must fail");
        assert!(message.contains("no_such_vocab"), "{message}");
        assert!(message.contains("cl100k_base"), "{message}");
    }

    #[cfg(not(feature = "vocab-cl100k"))]
    #[test]
    fn an_unbundled_name_names_the_feature() {
        let message = load_tokenizer("cl100k_base").err().map(|e| e.to_string());
        let message = message.expect("an unbundled vocabulary must fail");
        assert!(message.contains("vocab-cl100k"), "{message}");
    }

    #[test]
    fn a_json_file_loads_with_its_eos_and_hash() {
        let path = scratch_file("tiny", TINY_TOKENIZER_JSON);
        let value = path.to_string_lossy().into_owned();
        let loaded = load_tokenizer(&value).expect("the tiny tokenizer must load");
        remove_scratch(&path);
        assert_eq!(loaded.eos_id, 3);
        assert_eq!(loaded.bos_id, None);
        assert!(!loaded.template_ends_with_eos);
        let mut hasher = Fnv1a64::new();
        hasher.update(TINY_TOKENIZER_JSON.as_bytes());
        let want = TokenizerRecord::File {
            path: value,
            hash: hex(hasher.finish()),
        };
        assert_eq!(loaded.record, want);
    }

    #[test]
    fn a_json_file_without_eos_is_refused() {
        let json = r#"{
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
            "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
        }"#;
        let path = scratch_file("no-eos", json);
        let message = load_tokenizer(&path.to_string_lossy())
            .err()
            .map(|e| e.to_string())
            .expect("a tokenizer without EOS must fail");
        remove_scratch(&path);
        assert!(message.contains("end-of-sequence"), "{message}");
    }
}
