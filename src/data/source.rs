//! Token storage behind a loader: an in-memory vector or a memory-mapped file.

use crate::prelude::{Error, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub(super) enum DataSource {
    /// In-memory token array (for small datasets or dummy data)
    InMemory(Vec<u32>),
    /// Memory-mapped file (for large datasets)
    /// Safety: The Mmap is kept alive for the lifetime of the DataSource
    MemoryMapped {
        _mmap: Mmap,            // Keep alive
        tokens: &'static [u32], // Safe because _mmap outlives this reference
        len: usize,
    },
}

impl DataSource {
    /// Memory-maps a `.bin` file of native-endian u32 tokens.
    ///
    /// # Safety
    /// This function uses unsafe code to cast the memory-mapped region to a &'static [u32].
    /// This is safe because:
    /// 1. The Mmap is kept alive in the DataSource
    /// 2. The slice lifetime is tied to the Mmap lifetime via the enum
    /// 3. The file is assumed to be valid u32 data (caller responsibility)
    pub(super) fn map_file(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::DataError {
            reason: format!("failed to open data file {}: {e}", path.display()),
        })?;

        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| Error::DataError {
                reason: format!("failed to mmap data file {}: {e}", path.display()),
            })?
        };

        // Validate alignment and size
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

        let len = mmap.len() / std::mem::size_of::<u32>();

        // SAFETY:
        // - mmap is kept alive in the enum
        // - alignment is verified above
        // - we're casting to &'static which is safe because _mmap outlives this reference
        let tokens: &'static [u32] =
            unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const u32, len) };

        Ok(Self::MemoryMapped {
            _mmap: mmap,
            tokens,
            len,
        })
    }

    /// Total number of tokens.
    pub(super) fn len(&self) -> usize {
        match self {
            Self::InMemory(tokens) => tokens.len(),
            Self::MemoryMapped { len, .. } => *len,
        }
    }

    /// `count` tokens starting at `start`.
    pub(super) fn tokens(&self, start: usize, count: usize) -> &[u32] {
        match self {
            Self::InMemory(tokens) => &tokens[start..start + count],
            Self::MemoryMapped { tokens, .. } => &tokens[start..start + count],
        }
    }
}
