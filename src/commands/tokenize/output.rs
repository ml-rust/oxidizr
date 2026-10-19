//! The token file writer of `oxidizr tokenize`.
//!
//! Ids go to `<output>.partial` as little-endian `u32`. The full-file hash
//! and the counts are taken from the same bytes. On commit:
//!
//! 1. The partial file is flushed and synced.
//! 2. A sidecar beside the old output is removed, so no sidecar names the new bytes wrongly.
//! 3. The partial file is renamed over the output, and the directory synced.
//!
//! A writer dropped before commit removes its partial file.

use crate::data::token_meta_path;
use crate::data_hash::{hex, Fnv1a64};
use anyhow::{anyhow, Result};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// What the committed token file holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenTokens {
    /// Tokens in the file.
    pub num_tokens: u64,
    /// The highest id, or `None` for zero tokens.
    pub max_token_id: Option<u32>,
    /// FNV-1a 64 of every byte, as 16 hex digits.
    pub file_hash: String,
}

/// Appends ids to a partial file, then commits it in place of the output.
pub struct TokenWriter {
    output: PathBuf,
    partial: PathBuf,
    file: Option<BufWriter<File>>,
    bytes: Vec<u8>,
    hasher: Fnv1a64,
    num_tokens: u64,
    max_token_id: Option<u32>,
}

/// `out.bin` -> `out.bin.partial`.
pub fn partial_path(output: &Path) -> PathBuf {
    let mut name: OsString = output.as_os_str().to_owned();
    name.push(".partial");
    PathBuf::from(name)
}

impl TokenWriter {
    /// Creates `<output>.partial`, replacing a stale one.
    ///
    /// # Errors
    /// Returns an error naming the partial path when it cannot be created.
    pub fn create(output: &Path) -> Result<Self> {
        let partial = partial_path(output);
        let file = File::create(&partial).map_err(|e| {
            anyhow!(
                "could not create '{}': {e}. Check that the directory of -o exists and is writable.",
                partial.display()
            )
        })?;
        Ok(Self {
            output: output.to_path_buf(),
            partial,
            file: Some(BufWriter::new(file)),
            bytes: Vec::new(),
            hasher: Fnv1a64::new(),
            num_tokens: 0,
            max_token_id: None,
        })
    }

    /// Appends `ids` as little-endian `u32`.
    ///
    /// # Errors
    /// Returns an error naming the partial path when the write fails.
    pub fn append(&mut self, ids: &[u32]) -> Result<()> {
        self.bytes.clear();
        self.bytes.reserve(ids.len() * 4);
        for &id in ids {
            self.bytes.extend_from_slice(&id.to_le_bytes());
        }
        if let Some(&max) = ids.iter().max() {
            self.max_token_id = Some(self.max_token_id.map_or(max, |m| m.max(max)));
        }
        self.hasher.update(&self.bytes);
        self.num_tokens += ids.len() as u64;
        let partial = &self.partial;
        let file = self.file.as_mut().ok_or_else(|| closed(partial))?;
        file.write_all(&self.bytes)
            .map_err(|e| write_error(partial, e))
    }

    /// Syncs the partial file and renames it over the output.
    ///
    /// # Errors
    /// Returns an error naming the failing path. The partial file is removed.
    pub fn commit(mut self) -> Result<WrittenTokens> {
        let file = self.file.take().ok_or_else(|| closed(&self.partial))?;
        let file = file
            .into_inner()
            .map_err(|e| write_error(&self.partial, e.into_error()))?;
        file.sync_all().map_err(|e| write_error(&self.partial, e))?;
        drop(file);

        let sidecar = token_meta_path(&self.output);
        match fs::remove_file(&sidecar) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(anyhow!(
                    "could not remove the old sidecar '{}': {e}. Remove it, then run again.",
                    sidecar.display()
                ))
            }
        }
        fs::rename(&self.partial, &self.output).map_err(|e| {
            anyhow!(
                "could not rename '{}' to '{}': {e}. Check that -o is writable.",
                self.partial.display(),
                self.output.display()
            )
        })?;
        sync_parent(&self.output)?;
        Ok(WrittenTokens {
            num_tokens: self.num_tokens,
            max_token_id: self.max_token_id,
            file_hash: hex(self.hasher.finish()),
        })
    }
}

fn write_error(partial: &Path, e: std::io::Error) -> anyhow::Error {
    anyhow!(
        "could not write '{}': {e}. Check the free space on its disk.",
        partial.display()
    )
}

fn closed(partial: &Path) -> anyhow::Error {
    anyhow!(
        "'{}' is already closed. Run the command again.",
        partial.display()
    )
}

impl Drop for TokenWriter {
    /// Removes the partial file unless [`TokenWriter::commit`] renamed it.
    fn drop(&mut self) {
        drop(self.file.take());
        if self.partial.exists() {
            let _ = fs::remove_file(&self.partial);
        }
    }
}

/// Syncs the directory of `path`, so the rename survives a crash.
fn sync_parent(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let dir = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(|e| anyhow!("could not sync directory '{}': {e}.", dir.display()))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "oxidizr-tokenize-output-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    #[test]
    fn commit_writes_little_endian_ids_and_their_hash() {
        let dir = scratch_dir("commit");
        let output = dir.join("out.bin");
        fs::write(token_meta_path(&output), b"stale").expect("write stale sidecar");
        let mut writer = TokenWriter::create(&output).expect("create");
        writer.append(&[1, 0x0102_0304]).expect("append");
        writer.append(&[7]).expect("append");
        let written = writer.commit().expect("commit");

        let bytes = fs::read(&output).expect("read output");
        assert_eq!(bytes, [1, 0, 0, 0, 4, 3, 2, 1, 7, 0, 0, 0]);
        let mut hasher = Fnv1a64::new();
        hasher.update(&bytes);
        assert_eq!(written.file_hash, hex(hasher.finish()));
        assert_eq!(written.num_tokens, 3);
        assert_eq!(written.max_token_id, Some(0x0102_0304));
        assert!(!partial_path(&output).exists());
        assert!(
            !token_meta_path(&output).exists(),
            "the stale sidecar must go"
        );
        fs::remove_dir_all(&dir).expect("remove scratch dir");
    }

    #[test]
    fn a_dropped_writer_leaves_the_old_output_alone() {
        let dir = scratch_dir("drop");
        let output = dir.join("out.bin");
        fs::write(&output, b"old!").expect("write old output");
        let mut writer = TokenWriter::create(&output).expect("create");
        writer.append(&[9, 9]).expect("append");
        drop(writer);
        assert!(!partial_path(&output).exists());
        assert_eq!(fs::read(&output).expect("read"), b"old!");
        fs::remove_dir_all(&dir).expect("remove scratch dir");
    }
}
