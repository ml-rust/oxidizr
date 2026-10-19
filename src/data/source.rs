//! Token storage behind a loader: whole rows over one or more token arrays.
//!
//! A [`RowStore`] holds its sources in order. Each source keeps its own whole
//! rows of `width` tokens and drops its own tail, so no row spans two
//! sources. Global row `r` counts rows from the start of source 0.
//!
//! Token files hold little-endian `u32` ids. On a little-endian target a file
//! is memory-mapped and read in place. On a big-endian target the file is
//! read into memory once, and each id is decoded from little-endian bytes.

use super::token_meta::{read_token_meta, token_meta_path};
use crate::prelude::{Error, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// The token ids of one source.
enum Tokens {
    /// Ids held in memory: in-memory data, or a decoded file.
    InMemory(Vec<u32>),
    /// A read-only map of a non-empty file of little-endian ids.
    #[cfg(target_endian = "little")]
    Mapped {
        /// The map. [`map_tokens`] checked its start alignment and length.
        mmap: Mmap,
        /// Whole ids in the map.
        len: usize,
    },
}

impl Tokens {
    /// Every whole id of the source.
    fn as_slice(&self) -> &[u32] {
        match self {
            Self::InMemory(tokens) => tokens,
            #[cfg(target_endian = "little")]
            Self::Mapped { mmap, len } => {
                // SAFETY: `map_tokens` checked that the map starts on a u32
                // boundary and holds at least `len * 4` bytes. Every bit
                // pattern is a valid u32, and the target is little-endian, so
                // each id reads as stored. The slice borrows `mmap`, so it
                // cannot outlive the map.
                unsafe { std::slice::from_raw_parts(mmap.as_ptr().cast::<u32>(), *len) }
            }
        }
    }
}

/// One source of a store.
struct Part {
    tokens: Tokens,
    /// Whole rows of the source. Its tail past them is never read.
    rows: usize,
    /// Global row of the source's row 0.
    first_row: usize,
    /// The file the source came from. `None` for in-memory data.
    path: Option<PathBuf>,
    /// `max_token_id` of the file's `.tokens.json` sidecar, when it has one.
    max_token_id: Option<u32>,
}

/// Whole rows of `width` tokens over one or more sources, read in source order.
pub(super) struct RowStore {
    parts: Vec<Part>,
    width: usize,
    total_rows: usize,
    total_tokens: usize,
}

impl RowStore {
    /// A store of one in-memory source.
    pub(super) fn in_memory(tokens: Vec<u32>, width: usize) -> Self {
        Self::in_memory_parts(vec![tokens], width)
    }

    /// A store of in-memory sources, read in order.
    pub(super) fn in_memory_parts(sources: Vec<Vec<u32>>, width: usize) -> Self {
        let parts = sources
            .into_iter()
            .map(|tokens| (Tokens::InMemory(tokens), None, None))
            .collect();
        Self::assemble(parts, width)
    }

    /// A store of the token files at `paths`, read in order.
    ///
    /// A file's `.tokens.json` sidecar, when present, must record the file's
    /// token count. A missing sidecar is not an error.
    ///
    /// # Errors
    /// Returns an error for an empty path list, a file that does not open or
    /// map, a size that is not a multiple of 4, or a sidecar that disagrees.
    pub(super) fn map_files<P: AsRef<Path>>(paths: &[P], width: usize) -> Result<Self> {
        if paths.is_empty() {
            return Err(Error::DataError {
                reason: "no data files given. Pass -d <file.bin>, or set trainer.data.sources."
                    .to_string(),
            });
        }
        let mut parts = Vec::with_capacity(paths.len());
        for path in paths {
            let path = path.as_ref();
            let tokens = read_token_file(path)?;
            let max_token_id = sidecar_max_token_id(path, tokens.as_slice().len())?;
            parts.push((tokens, Some(path.to_path_buf()), max_token_id));
        }
        Ok(Self::assemble(parts, width))
    }

    fn assemble(sources: Vec<(Tokens, Option<PathBuf>, Option<u32>)>, width: usize) -> Self {
        let mut parts = Vec::with_capacity(sources.len());
        let (mut total_rows, mut total_tokens) = (0, 0);
        for (tokens, path, max_token_id) in sources {
            let len = tokens.as_slice().len();
            let rows = len / width;
            parts.push(Part {
                tokens,
                rows,
                first_row: total_rows,
                path,
                max_token_id,
            });
            total_rows += rows;
            total_tokens += len;
        }
        Self {
            parts,
            width,
            total_rows,
            total_tokens,
        }
    }

    /// Whole rows over every source.
    pub(super) fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// Tokens over every source, tails included.
    pub(super) fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Whole rows of each source, in source order.
    pub(super) fn source_rows(&self) -> Vec<usize> {
        self.parts.iter().map(|p| p.rows).collect()
    }

    /// The file of each file-backed source, in source order.
    pub(super) fn paths(&self) -> impl Iterator<Item = &Path> {
        self.parts.iter().filter_map(|p| p.path.as_deref())
    }

    /// The `width` tokens of global row `global_row`.
    ///
    /// # Errors
    /// Returns an error when the row is not below [`total_rows`](Self::total_rows).
    pub(super) fn row(&self, global_row: usize) -> Result<&[u32]> {
        let index = self
            .parts
            .partition_point(|p| p.first_row + p.rows <= global_row);
        let start = self
            .parts
            .get(index)
            .map(|p| (p, (global_row - p.first_row) * self.width));
        start
            .and_then(|(part, start)| part.tokens.as_slice().get(start..start + self.width))
            .ok_or_else(|| Error::DataError {
                reason: format!(
                    "row {global_row} is past the {} whole rows of this data. This is a loader \
                     bug: report it with the data and config.",
                    self.total_rows
                ),
            })
    }

    /// Checks every sidecar's `max_token_id` against `vocab_size`.
    ///
    /// A source without a sidecar is not checked.
    ///
    /// # Errors
    /// Returns an error naming the first file with an id at or above `vocab_size`.
    pub(super) fn check_vocab(&self, vocab_size: usize) -> Result<()> {
        for part in &self.parts {
            let (Some(path), Some(max)) = (&part.path, part.max_token_id) else {
                continue;
            };
            if vocab_size <= max as usize {
                return Err(Error::DataError {
                    reason: format!(
                        "data file {} holds token id {max} (max_token_id in {}), and \
                         model.vocab_size is {vocab_size}: that id has no embedding row. Set \
                         model.vocab_size to at least {}, or train on a file tokenized for this \
                         vocabulary.",
                        path.display(),
                        token_meta_path(path).display(),
                        u64::from(max) + 1
                    ),
                });
            }
        }
        Ok(())
    }
}

/// Reads the token file at `path`: a map on a little-endian target.
fn read_token_file(path: &Path) -> Result<Tokens> {
    let file = File::open(path).map_err(|e| Error::DataError {
        reason: format!("failed to open data file {}: {e}", path.display()),
    })?;
    // SAFETY: the map is read-only. The file must not shrink while a run reads it.
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| Error::DataError {
            reason: format!("failed to mmap data file {}: {e}", path.display()),
        })?
    };
    if mmap.len() % std::mem::size_of::<u32>() != 0 {
        return Err(Error::DataError {
            reason: format!(
                "data file {} is {} bytes, not a multiple of 4: it is not a u32 token \
                 array. Rebuild it as raw u32 tokens.",
                path.display(),
                mmap.len()
            ),
        });
    }
    map_tokens(mmap, path)
}

/// Reads the ids of `mmap` in place. Only the size is checked above, so the
/// start alignment is checked here.
#[cfg(target_endian = "little")]
fn map_tokens(mmap: Mmap, path: &Path) -> Result<Tokens> {
    let len = mmap.len() / std::mem::size_of::<u32>();
    if len == 0 {
        return Ok(Tokens::InMemory(Vec::new()));
    }
    // A map starts on a page boundary, so this holds on every supported OS.
    if mmap.as_ptr().align_offset(std::mem::align_of::<u32>()) != 0 {
        return Err(Error::DataError {
            reason: format!(
                "the map of data file {} does not start on a 4-byte boundary. Report this \
                 with the OS name.",
                path.display()
            ),
        });
    }
    Ok(Tokens::Mapped { mmap, len })
}

/// Decodes the little-endian ids of `mmap` into memory.
#[cfg(target_endian = "big")]
fn map_tokens(mmap: Mmap, _path: &Path) -> Result<Tokens> {
    Ok(Tokens::InMemory(decode_le(&mmap)))
}

/// Decodes little-endian `u32` ids. Trailing bytes short of an id are dropped.
#[cfg(any(target_endian = "big", test))]
fn decode_le(bytes: &[u8]) -> Vec<u32> {
    let (ids, _) = bytes.as_chunks::<4>();
    ids.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

/// The sidecar's `max_token_id`, after checking its token count against `tokens`.
///
/// `Ok(None)` without a sidecar, or for a sidecar of an empty file.
fn sidecar_max_token_id(path: &Path, tokens: usize) -> Result<Option<u32>> {
    let Some(meta) = read_token_meta(path)? else {
        return Ok(None);
    };
    if meta.num_tokens != tokens as u64 {
        return Err(Error::DataError {
            reason: format!(
                "token sidecar {} records num_tokens {}, and data file {} holds {tokens} \
                 tokens: the sidecar belongs to another file. Run `oxidizr tokenize` again \
                 to rewrite both, or remove the sidecar.",
                token_meta_path(path).display(),
                meta.num_tokens,
                path.display()
            ),
        });
    }
    Ok(meta.max_token_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn little_endian_bytes_decode_by_value() {
        let bytes = [1, 0, 0, 0, 0x04, 0x03, 0x02, 0x01, 9];
        assert_eq!(decode_le(&bytes), vec![1, 0x0102_0304]);
    }

    #[test]
    fn each_source_drops_its_own_tail() {
        // Width 3: 7 tokens hold 2 rows, 5 tokens hold 1, 2 tokens hold none.
        let store = RowStore::in_memory_parts(
            vec![(0..7).collect(), (100..105).collect(), vec![200, 201]],
            3,
        );
        assert_eq!(store.source_rows(), vec![2, 1, 0]);
        assert_eq!(store.total_rows(), 3);
        assert_eq!(store.total_tokens(), 14);
        assert_eq!(store.row(0).expect("row 0"), &[0, 1, 2]);
        assert_eq!(store.row(1).expect("row 1"), &[3, 4, 5]);
        assert_eq!(store.row(2).expect("row 2"), &[100, 101, 102]);
        assert!(store.row(3).is_err());
    }

    #[test]
    fn empty_sources_are_skipped() {
        let store = RowStore::in_memory_parts(vec![vec![], (0..4).collect(), vec![]], 2);
        assert_eq!(store.source_rows(), vec![0, 2, 0]);
        assert_eq!(store.row(1).expect("row 1"), &[2, 3]);
        assert!(store.row(2).is_err());
    }

    #[test]
    fn no_paths_is_an_error() {
        let paths: [&Path; 0] = [];
        let Err(err) = RowStore::map_files(&paths, 4) else {
            panic!("an empty path list must be refused");
        };
        assert!(err.to_string().contains("no data files"), "{err}");
    }
}
