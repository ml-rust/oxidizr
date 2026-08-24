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
//! beside the data: for `foo.bin`, `foo.layout.json` holds the serialized
//! [`SpeechVocab`].
//!
//! A MISSING sidecar is not an error — text corpora predate this and have none.
//! A sidecar that DISAGREES with the configured layout is a hard error, because
//! it is exactly the silent corruption above.

use std::fs::File;
use std::path::{Path, PathBuf};

use boostr::model::SpeechVocab;

use crate::prelude::{Error, Result};

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
pub fn write_speech_corpus(path: &Path, tokens: &[u32], vocab: &SpeechVocab) -> Result<()> {
    crate::data::write_tokens_to_file(tokens, path).map_err(|e| Error::DataError {
        reason: format!("failed to write corpus {}: {e}", path.display()),
    })?;

    let sidecar = layout_sidecar_path(path);
    let file = File::create(&sidecar).map_err(|e| Error::DataError {
        reason: format!("failed to create layout sidecar {}: {e}", sidecar.display()),
    })?;
    serde_json::to_writer_pretty(&file, vocab).map_err(|e| Error::DataError {
        reason: format!("failed to write layout sidecar {}: {e}", sidecar.display()),
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
pub fn read_corpus_layout(bin_path: &Path) -> Result<Option<SpeechVocab>> {
    let sidecar = layout_sidecar_path(bin_path);
    if !sidecar.exists() {
        return Ok(None);
    }
    let file = File::open(&sidecar).map_err(|e| Error::DataError {
        reason: format!("failed to open layout sidecar {}: {e}", sidecar.display()),
    })?;
    let vocab: SpeechVocab = serde_json::from_reader(&file).map_err(|e| Error::DataError {
        reason: format!("failed to parse layout sidecar {}: {e}", sidecar.display()),
    })?;
    Ok(Some(vocab))
}

/// Serialize a layout for an error message, falling back to its `Debug` form.
fn describe(vocab: &SpeechVocab) -> String {
    match serde_json::to_string(vocab) {
        Ok(json) => json,
        Err(_) => format!("{vocab:?}"),
    }
}

/// Check the sidecar beside `bin_path` against the layout this run is configured
/// with.
///
/// - No sidecar: `Ok(())`. Existing corpora have none.
/// - Sidecar equal to `vocab`: `Ok(())`.
/// - Sidecar different: hard error naming BOTH layouts. Never a warning — the
///   run that continues past this point trains on ids that mean something else.
pub fn check_corpus_layout(bin_path: &Path, vocab: &SpeechVocab) -> Result<()> {
    let Some(packed) = read_corpus_layout(bin_path)? else {
        return Ok(());
    };
    if &packed == vocab {
        return Ok(());
    }
    Err(Error::DataError {
        reason: format!(
            "speech layout mismatch for corpus {}: the sidecar {} records the layout this \
             corpus was PACKED with, and it differs from the layout this run is CONFIGURED \
             with. Every audio id would land in the wrong codebook and the loss mask would \
             select the wrong positions.\n  packed layout:     {}\n  configured layout: {}\n\
             Repack the corpus with the configured layout, or train under the layout it was \
             packed with.",
            bin_path.display(),
            layout_sidecar_path(bin_path).display(),
            describe(&packed),
            describe(vocab),
        ),
    })
}
