//! Building a [`LitDataLoader`]: opening a source, and the builder methods
//! that configure row order, target masking, and the row range before
//! training starts.

use super::mask::TargetMask;
use super::mixture::Mixture;
use super::row_ranges::RowRanges;
use super::rows::{DataPosition, EpochOrder, RowLayout, RowOrder};
use super::source::RowStore;
use crate::prelude::{DType, Result};
use boostr::model::SpeechLayout;
use boostr::runtime::Runtime;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Row loader supporting both in-memory and memory-mapped datasets.
///
/// # Rows
/// The stream is cut into fixed rows of `seq_len + 1` tokens (see
/// `super::rows`). A batch is `batch_size` rows: inputs are
/// `row[0..seq_len]`, targets are `row[1..seq_len + 1]`. Tokens past the last
/// whole row of a source are never read.
///
/// # Sources
/// [`from_files`](Self::from_files) reads several token files as one
/// stream: the rows of file 0, then the rows of file 1, and so on. Each file
/// drops its own tail, so no row spans two files.
///
/// # Row range
/// A loader reads ascending global row ranges. The default is every whole
/// row. [`with_row_range`](Self::with_row_range) narrows it to one range, for
/// example to split a held-out eval range off the end.
/// [`with_row_ranges`](Self::with_row_ranges) reads several ranges in order.
///
/// # Row order
/// Each epoch hands out every row once. The default is stream order;
/// [`with_row_order`](Self::with_row_order) selects a seeded shuffle.
///
/// # Mixture
/// [`with_mixture`](Self::with_mixture) weighs the sources. Each epoch then
/// holds a fixed number of slots, and each source fills its share of them.
///
/// # Ranks
/// The default is rank 0 of a world of 1. [`with_rank`](Self::with_rank)
/// makes the loader one rank of a data-parallel run. Every rank keeps the
/// whole row range and the same order. Each rank reads its own `batch_size`
/// slots of every global batch.
///
/// # Supported Formats
/// - `.bin` files: raw little-endian `u32` token arrays
/// - In-memory: `Vec<u32>` for small datasets or testing
///
/// # Memory Mapping
/// On a little-endian target files are memory-mapped, so the OS pages the
/// dataset in on demand.
pub struct LitDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// The rows of every source, in source order.
    pub(super) store: RowStore,
    pub(super) batch_size: usize,
    pub(super) seq_len: usize,
    pub(super) device: R::Device,
    /// The global rows this loader reads.
    pub(super) ranges: RowRanges,
    /// Next batch's place in the row order. Every rank holds the same one.
    pub(super) position: DataPosition,
    /// This loader's data-parallel rank, below `world_size`.
    pub(super) rank: usize,
    /// Ranks that split each global batch.
    pub(super) world_size: usize,
    /// Row at each slot of each epoch.
    pub(super) order: EpochOrder,
    /// The weighted mixture of the sources. `None` reads the row ranges as one stream.
    pub(super) mixture: Option<Mixture>,
    /// Target mask rule. `Some` turns on masked loss scoring (audio-only for
    /// speech, continuation-only for prefix-LM).
    pub(super) target_mask: Option<TargetMask>,
    /// The document separator. `Some` turns on document segments and the
    /// separator loss mask, see [`super::host_masks`].
    pub(super) document_mask: Option<u32>,
}

impl<R> LitDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// Create a new data loader from in-memory tokens (for testing or small datasets)
    pub fn new(tokens: Vec<u32>, batch_size: usize, seq_len: usize, device: R::Device) -> Self {
        let store = RowStore::in_memory(tokens, seq_len + 1);
        Self::from_store(store, batch_size, seq_len, device)
    }

    pub(super) fn from_store(
        store: RowStore,
        batch_size: usize,
        seq_len: usize,
        device: R::Device,
    ) -> Self {
        let ranges = RowRanges::whole(store.total_rows());
        Self {
            store,
            batch_size,
            seq_len,
            device,
            ranges,
            position: DataPosition::default(),
            rank: 0,
            world_size: 1,
            order: EpochOrder::new(RowOrder::Stream),
            mixture: None,
            target_mask: None,
            document_mask: None,
        }
    }

    /// Create a data loader from one token file.
    ///
    /// # Arguments
    /// * `path` - Path to .bin file containing raw u32 tokens
    /// * `batch_size` - Rows per batch
    /// * `seq_len` - Sequence length; rows are `seq_len + 1` tokens
    /// * `device` - Target runtime device
    ///
    /// # File Format
    /// The .bin file holds a contiguous array of little-endian u32 tokens.
    /// [`from_files`](Self::from_files) documents the sidecar check.
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        batch_size: usize,
        seq_len: usize,
        device: R::Device,
    ) -> Result<Self> {
        Self::from_files(&[path], batch_size, seq_len, device)
    }

    /// Select how rows are ordered within each epoch.
    pub fn with_row_order(mut self, order: RowOrder) -> Self {
        self.order = EpochOrder::new(order);
        self
    }

    /// Read only the stream rows `[start, end)`.
    ///
    /// Row indices count whole rows of `seq_len + 1` tokens from the stream
    /// start. The position resets to the start of the range.
    ///
    /// # Errors
    /// Returns an error when the range is empty or ends past the last whole row.
    pub fn with_row_range(self, start: usize, end: usize) -> Result<Self> {
        self.with_row_ranges(&[(start, end)])
    }

    /// Enable masked loss scoring with an arbitrary [`TargetMask`] rule.
    ///
    /// [`with_speech_layout`](Self::with_speech_layout) and
    /// [`try_with_speech_layout`](Self::try_with_speech_layout) are thin
    /// wrappers around this for the speech case. Prefix-LM continuation
    /// scoring (`TargetMask::PrefixLm`) has no corpus-layout sidecar to
    /// check, so it goes through this form directly.
    pub fn with_target_mask(mut self, mask: TargetMask) -> Self {
        self.target_mask = Some(mask);
        self
    }

    /// Enable audio-only loss masking with this speech id layout.
    ///
    /// Without it the loader emits `loss_mask: None` and training scores every
    /// position, as it did before speech support existed.
    ///
    /// This form does NOT check the corpus layout sidecar. Prefer
    /// [`try_with_speech_layout`](Self::try_with_speech_layout) for any loader
    /// opened from a file.
    pub fn with_speech_layout(self, layout: SpeechLayout) -> Self {
        self.with_target_mask(TargetMask::SpeechAudio(layout))
    }

    /// Enable audio-only loss masking, first checking the corpus layout sidecar.
    ///
    /// If a data file has a `*.layout.json` sidecar (see
    /// [`crate::speech_corpus`]), it records the layout the corpus was PACKED
    /// with and MUST match `vocab`. Disagreement is a hard error: the ids would
    /// resolve to different codebooks and the mask would select different
    /// positions, with nothing else in the pipeline able to notice. Every
    /// file of a multi-file loader is checked.
    ///
    /// Masking is driven by the layout itself, so EITHER layout can be checked
    /// here: a corpus packed as `expressive-tts-1.7b` is accepted when that is
    /// what the run configures, and rejected when it is not.
    ///
    /// A missing sidecar is not an error — corpora packed before layouts were
    /// recorded, and plain text corpora, have none.
    pub fn try_with_speech_layout(self, layout: SpeechLayout) -> Result<Self> {
        for path in self.store.paths() {
            crate::speech_corpus::check_corpus_layout(path, &layout)?;
        }
        Ok(self.with_speech_layout(layout))
    }

    /// Returns the total number of tokens in the dataset
    pub fn dataset_size(&self) -> usize {
        self.store.total_tokens()
    }

    /// Tokens per row: `seq_len` inputs plus the one extra target.
    pub(super) fn row_width(&self) -> usize {
        self.seq_len + 1
    }

    /// Whole rows over every source, whatever the row range.
    pub fn total_rows(&self) -> usize {
        self.store.total_rows()
    }

    /// The stream rows `(start, end)` this loader reads, end exclusive.
    ///
    /// Under several ranges this spans the start of the first range to the
    /// end of the last. [`row_ranges`](Self::row_ranges) gives each range.
    pub fn row_range(&self) -> (usize, usize) {
        self.ranges.hull()
    }

    /// Row and global batch counts of this loader's row range.
    ///
    /// One global batch is `batch_size` rows from each of `world_size` ranks.
    /// Under a weighted mixture an epoch is its rows per epoch.
    pub fn row_layout(&self) -> Result<RowLayout> {
        RowLayout::for_world(self.epoch_rows(), self.batch_size, self.world_size)
    }

    /// Slots per epoch: the mixture's rows per epoch, else the rows of the row ranges.
    fn epoch_rows(&self) -> usize {
        self.mixture
            .as_ref()
            .map_or(self.ranges.len(), Mixture::rows_per_epoch)
    }

    /// Tokens one epoch reads.
    #[allow(dead_code)]
    pub fn shard_size(&self) -> usize {
        self.epoch_rows() * self.row_width()
    }
}

// Helper to generate fake data for testing "zero to hero" pipeline
pub fn create_dummy_data(vocab_size: usize, total_tokens: usize) -> Vec<u32> {
    let mut data = Vec::with_capacity(total_tokens);
    for _ in 0..total_tokens {
        data.push(rand::random_range(0..vocab_size) as u32);
    }
    data
}

/// Write tokens as little-endian `u32` to a binary file for the data loader.
///
/// # Example
/// ```no_run
/// use oxidizr::data::{create_dummy_data, write_tokens_to_file};
///
/// let tokens = create_dummy_data(128354, 10_000_000); // 10M tokens, Llama 3 + splintr vocab
/// write_tokens_to_file(&tokens, "data/train.bin").expect("Failed to write");
/// ```
#[allow(dead_code)] // Public API for users to create binary token files
pub fn write_tokens_to_file<P: AsRef<Path>>(tokens: &[u32], path: P) -> std::io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for token in tokens {
        writer.write_all(&token.to_le_bytes())?;
    }
    let file = writer.into_inner().map_err(|e| e.into_error())?;
    file.sync_all()
}
