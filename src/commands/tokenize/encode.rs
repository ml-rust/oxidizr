//! Document encoding for `oxidizr tokenize`.
//!
//! One document becomes the policy template around its content ids, then the
//! EOS id. The ids of one document depend on its text alone. Batch size,
//! thread count and the large-document path never change them.

use crate::tokenizer::LoadedTokenizer;
use anyhow::{anyhow, Result};
use splintr::{AnyTokenizer, Backend, SpecialMode};

/// A document above this many bytes is split across threads inside splintr.
pub const LARGE_DOC_BYTES: usize = 1 << 20;

/// Encodes batches of documents into one id stream.
pub struct DocEncoder<'a> {
    loaded: &'a LoadedTokenizer,
    mode: SpecialMode<'static>,
    /// A handle whose `encode_rayon` gives the ids of `encode_with(_, mode)`.
    /// `None` for a backend with no parallel path inside one text.
    large: Option<AnyTokenizer>,
}

impl<'a> DocEncoder<'a> {
    /// An encoder over `loaded`. `allow_special` matches special-token text.
    pub fn new(loaded: &'a LoadedTokenizer, allow_special: bool) -> Self {
        let mode = if allow_special {
            SpecialMode::All
        } else {
            SpecialMode::Ordinary
        };
        // `encode_rayon` matches special-token text when the BPE handle has
        // matching on. Under `Ordinary` the large-document handle turns it off.
        let large = match loaded.tokenizer.backend() {
            Backend::Bpe(bpe) => {
                let bpe = if allow_special {
                    bpe.clone()
                } else {
                    bpe.clone().with_added_token_matching(false)
                };
                let policy = loaded.tokenizer.policy().clone();
                Some(AnyTokenizer::new(Backend::Bpe(bpe), policy))
            }
            Backend::Unigram(_) | Backend::WordPiece(_) | Backend::Spm(_) => None,
        };
        Self {
            loaded,
            mode,
            large,
        }
    }

    /// Appends the ids of every document in `docs` to `out`, in order.
    ///
    /// # Errors
    /// Returns splintr's error when a document does not encode.
    pub fn encode_into(&self, docs: &[String], out: &mut Vec<u32>) -> Result<()> {
        let mut small: Vec<&str> = Vec::new();
        for doc in docs {
            if doc.len() > LARGE_DOC_BYTES {
                self.encode_small(&small, out)?;
                small.clear();
                let ids = self.encode_large(doc)?;
                self.push_document(&ids, out);
            } else {
                small.push(doc);
            }
        }
        self.encode_small(&small, out)
    }

    fn encode_small(&self, docs: &[&str], out: &mut Vec<u32>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let encoded = self
            .loaded
            .tokenizer
            .encode_batch_with(docs, &self.mode)
            .map_err(|e| anyhow!("a document did not encode: {e}. Report this with the input."))?;
        for ids in &encoded {
            self.push_document(ids, out);
        }
        Ok(())
    }

    fn encode_large(&self, doc: &str) -> Result<Vec<u32>> {
        match &self.large {
            Some(handle) => Ok(handle.encode_rayon(doc)),
            None => self
                .loaded
                .tokenizer
                .encode_with(doc, &self.mode)
                .map_err(|e| {
                    anyhow!("a document did not encode: {e}. Report this with the input.")
                }),
        }
    }

    fn push_document(&self, ids: &[u32], out: &mut Vec<u32>) {
        out.extend_from_slice(ids);
        if !self.loaded.template_ends_with_eos {
            out.push(self.loaded.eos_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{load_tokenizer, TINY_TOKENIZER_JSON};

    fn tiny(name: &str) -> LoadedTokenizer {
        let dir = std::env::temp_dir().join(format!(
            "oxidizr-tokenize-encode-{}-{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, TINY_TOKENIZER_JSON).expect("write tokenizer");
        let loaded = load_tokenizer(&path.to_string_lossy()).expect("load the tiny tokenizer");
        std::fs::remove_dir_all(&dir).expect("remove scratch dir");
        loaded
    }

    fn reference(loaded: &LoadedTokenizer, docs: &[String]) -> Vec<u32> {
        let mut out = Vec::new();
        for doc in docs {
            let ids = loaded
                .tokenizer
                .encode_with(doc, &SpecialMode::Ordinary)
                .expect("encode");
            out.extend(ids);
            out.push(loaded.eos_id);
        }
        out
    }

    #[test]
    fn each_document_is_its_ids_then_the_separator() {
        let loaded = tiny("small");
        let docs = vec!["ab".to_owned(), "a b".to_owned(), "b".to_owned()];
        let mut out = Vec::new();
        DocEncoder::new(&loaded, false)
            .encode_into(&docs, &mut out)
            .expect("encode");
        assert_eq!(out, vec![0, 1, 3, 0, 2, 1, 3, 1, 3]);
        assert_eq!(out, reference(&loaded, &docs));
    }

    #[test]
    fn a_large_document_gives_the_ids_of_the_small_path() {
        let loaded = tiny("large");
        let large = "ab a ".repeat(LARGE_DOC_BYTES / 5 + 7);
        assert!(large.len() > LARGE_DOC_BYTES);
        let docs = vec!["a".to_owned(), large, "b".to_owned()];
        let mut out = Vec::new();
        DocEncoder::new(&loaded, false)
            .encode_into(&docs, &mut out)
            .expect("encode");
        assert_eq!(out, reference(&loaded, &docs));
    }
}
