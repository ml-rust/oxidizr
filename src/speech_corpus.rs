//! Recording WHICH speech id layout a packed `.bin` corpus was written under.
//!
//! A training corpus is a header-less, magic-less flat `u32` array: nothing in
//! the file says what the ids mean. For a text-only corpus that is harmless. For
//! a speech corpus it is not: every id at or above
//! [`SpeechVocab::audio_base`](boostr::model::SpeechVocab::audio_base) means
//! "codebook C, code K" under ONE layout and something else under any other, and
//! the audio-only loss mask selects positions by that same boundary. Train a
//! corpus packed under one layout with a differently configured layout and every
//! audio id lands in the wrong codebook while the mask selects the wrong
//! positions — with no error anywhere, because both files are just numbers.
//!
//! The fix is a SIDECAR, not a header. Adding a header would change the `.bin`
//! format and break every existing text corpus. Instead the layout is written
//! beside the data: for `foo.bin`, `foo.layout.json` holds a [`CorpusLayout`].
//!
//! The sidecar describes the LAYOUT, not one vocabulary. There is more than one
//! layout — boostr's native [`SpeechVocab`] scheme and the
//! `Multilingual-Expressive-TTS-1.7B` scheme — and a file that can only express
//! one of them cannot report the mismatch between them, which is the only thing
//! it exists to do. So the JSON carries a `layout` tag plus whatever decides
//! what an id MEANS under that layout.
//!
//! A MISSING sidecar is not an error — text corpora predate this and have none.
//! A sidecar that DISAGREES with the configured layout is a hard error, because
//! it is exactly the silent corruption above.

use std::fs::{self, File};
use std::path::{Path, PathBuf};

use boostr::model::{SpeechLayout, SpeechVocab};
use serde::{Deserialize, Serialize};

use crate::prelude::{Error, Result};

/// What a corpus was packed under, as the sidecar stores it.
///
/// Tagged by `layout`, so both [`SpeechLayout`] variants land in one file:
///
/// ```json
/// {"layout":"native","vocab":{ ...SpeechVocab... }}
/// {"layout":"expressive_tts","audio_base":151670,"codebook_size":65536,"vocab_size":217208}
/// ```
///
/// `Native` carries the whole [`SpeechVocab`], because every id in it is part of
/// the layout. `ExpressiveTts` has no `SpeechVocab` at all, so it carries the
/// three numbers that decide what an id means: the first audio id, the size of
/// the codebook that id space indexes, and the row count a checkpoint must have
/// to read the stream. They are RECORDED rather than implied by the tag, so a
/// corpus packed against a checkpoint whose ids move is caught by the numbers
/// instead of trusted because the name still matches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "layout", rename_all = "snake_case")]
pub enum CorpusLayout {
    /// boostr's own id layout, described in full by its [`SpeechVocab`].
    Native {
        /// The vocabulary the stream's ids are positions in.
        vocab: SpeechVocab,
    },
    /// The `Multilingual-Expressive-TTS-1.7B` layout.
    ExpressiveTts {
        /// Id of `<|s_0|>`; audio code `c` is `audio_base + c`.
        audio_base: u32,
        /// Codes in the single audio codebook.
        codebook_size: usize,
        /// Embedding rows a checkpoint must have to read this stream.
        vocab_size: usize,
    },
}

impl CorpusLayout {
    /// The sidecar form of a configured layout.
    pub fn of(layout: &SpeechLayout) -> Self {
        match layout {
            SpeechLayout::Native(vocab) => Self::Native {
                vocab: vocab.clone(),
            },
            SpeechLayout::ExpressiveTts(tts) => Self::ExpressiveTts {
                audio_base: tts.audio_base(),
                codebook_size: tts.codebook_size(),
                vocab_size: tts.vocab_size(),
            },
        }
    }

    /// Short name, matching [`SpeechLayout::name`], for error messages.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Native { .. } => "boostr-native",
            Self::ExpressiveTts { .. } => "expressive-tts-1.7b",
        }
    }

    /// Whether a corpus packed under `self` is readable as `layout`.
    ///
    /// Equality of the recorded description, so a different variant, a moved
    /// audio base, a resized codebook or a different row count all disagree.
    pub fn matches(&self, layout: &SpeechLayout) -> bool {
        *self == Self::of(layout)
    }
}

/// Path of the layout sidecar beside a corpus: `foo.bin` -> `foo.layout.json`.
///
/// The final extension is replaced, so a path with no extension gets the
/// sidecar name appended (`foo` -> `foo.layout.json`).
pub fn layout_sidecar_path(bin_path: &Path) -> PathBuf {
    bin_path.with_extension("layout.json")
}

/// Write a packed token stream plus the layout sidecar that identifies it.
///
/// The `.bin` payload is byte-identical to [`crate::data::write_tokens_to_file`];
/// the only addition is [`layout_sidecar_path`] beside it.
pub fn write_speech_corpus(path: &Path, tokens: &[u32], layout: &SpeechLayout) -> Result<()> {
    crate::data::write_tokens_to_file(tokens, path).map_err(|e| Error::DataError {
        reason: format!("failed to write corpus {}: {e}", path.display()),
    })?;

    let sidecar = layout_sidecar_path(path);
    let file = File::create(&sidecar).map_err(|e| Error::DataError {
        reason: format!("failed to create layout sidecar {}: {e}", sidecar.display()),
    })?;
    serde_json::to_writer_pretty(&file, &CorpusLayout::of(layout)).map_err(|e| {
        Error::DataError {
            reason: format!("failed to write layout sidecar {}: {e}", sidecar.display()),
        }
    })?;
    file.sync_all().map_err(|e| Error::DataError {
        reason: format!("failed to flush layout sidecar {}: {e}", sidecar.display()),
    })?;

    Ok(())
}

/// Read the sidecar beside a `.bin`, if one exists.
///
/// `Ok(None)` means no sidecar — a corpus written before layouts were recorded,
/// or a plain text corpus. A sidecar that exists but cannot be parsed is an
/// error: it was meant to describe this corpus and no longer does.
pub fn read_corpus_layout(bin_path: &Path) -> Result<Option<CorpusLayout>> {
    let sidecar = layout_sidecar_path(bin_path);
    if !sidecar.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&sidecar).map_err(|e| Error::DataError {
        reason: format!("failed to read layout sidecar {}: {e}", sidecar.display()),
    })?;
    Ok(Some(parse_sidecar(&text, &sidecar)?))
}

/// Parse either sidecar form.
///
/// The tagged [`CorpusLayout`] is tried first. The fallback is the PRE-EXISTING
/// on-disk format: a bare [`SpeechVocab`] object with no `layout` key, written
/// when the native layout was the only one. Such a file can only be native, so
/// it loads as [`CorpusLayout::Native`] rather than being rejected — corpora
/// already packed keep working.
fn parse_sidecar(text: &str, sidecar: &Path) -> Result<CorpusLayout> {
    if let Ok(layout) = serde_json::from_str::<CorpusLayout>(text) {
        return Ok(layout);
    }
    let vocab: SpeechVocab = serde_json::from_str(text).map_err(|e| Error::DataError {
        reason: format!(
            "failed to parse layout sidecar {}: it is neither a tagged layout \
             ({{\"layout\":\"native\"|\"expressive_tts\", ...}}) nor the older bare SpeechVocab \
             object: {e}",
            sidecar.display()
        ),
    })?;
    Ok(CorpusLayout::Native { vocab })
}

/// Serialize a recorded layout for an error message, falling back to `Debug`.
fn describe(layout: &CorpusLayout) -> String {
    match serde_json::to_string(layout) {
        Ok(json) => json,
        Err(_) => format!("{layout:?}"),
    }
}

/// Check the sidecar beside `bin_path` against the layout this run is configured
/// with.
///
/// - No sidecar: `Ok(())`. Existing corpora have none.
/// - Sidecar matching `layout`: `Ok(())`.
/// - Sidecar different: hard error naming BOTH layouts. Never a warning — the
///   run that continues past this point trains on ids that mean something else.
pub fn check_corpus_layout(bin_path: &Path, layout: &SpeechLayout) -> Result<()> {
    let Some(packed) = read_corpus_layout(bin_path)? else {
        return Ok(());
    };
    if packed.matches(layout) {
        return Ok(());
    }
    let configured = CorpusLayout::of(layout);
    Err(Error::DataError {
        reason: format!(
            "speech layout mismatch for corpus {}: the sidecar {} records the layout this \
             corpus was PACKED with, and it differs from the layout this run is CONFIGURED \
             with. Every audio id would land in the wrong codebook and the loss mask would \
             select the wrong positions.\n  packed layout:     {} {}\n  configured layout: {} \
             {}\nRepack the corpus with the configured layout, or train under the layout it \
             was packed with.",
            bin_path.display(),
            layout_sidecar_path(bin_path).display(),
            packed.name(),
            describe(&packed),
            configured.name(),
            describe(&configured),
        ),
    })
}
