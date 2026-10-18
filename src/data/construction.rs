//! Building a [`LitDataLoader`]: opening a source, and the builder methods
//! that configure row order, target masking, and sharding before training
//! starts.

use super::mask::TargetMask;
use super::rows::{DataPosition, EpochOrder, RowLayout, RowOrder};
use super::source::DataSource;
use crate::prelude::{DType, Result};
use boostr::model::SpeechLayout;
use boostr::runtime::Runtime;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Row loader supporting both in-memory and memory-mapped datasets.
///
/// # Rows
/// The stream is cut into fixed rows of `seq_len + 1` tokens (see
/// [`super::rows`]). A batch is `batch_size` rows: inputs are
/// `row[0..seq_len]`, targets are `row[1..seq_len + 1]`. Tokens past the last
/// whole row of the shard are never read.
///
/// # Row order
/// Each epoch hands out every row once. The default is stream order;
/// [`with_row_order`](Self::with_row_order) selects a seeded shuffle.
///
/// # Supported Formats
/// - `.bin` files: Raw binary token arrays (u32)
/// - In-memory: Vec<u32> for small datasets or testing
///
/// # Memory Mapping
/// Files are memory-mapped, so the OS pages the dataset in on demand.
pub struct LitDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// Data source - either in-memory or memory-mapped
    pub(super) data_source: DataSource,
    pub(super) batch_size: usize,
    pub(super) seq_len: usize,
    pub(super) device: R::Device,
    // Token range this loader reads
    pub(super) shard_start: usize,
    shard_end: usize,
    /// Next batch's place in the row order.
    pub(super) position: DataPosition,
    /// Row at each slot of each epoch.
    pub(super) order: EpochOrder,
    /// Target mask rule. `Some` turns on masked loss scoring (audio-only for
    /// speech, continuation-only for prefix-LM).
    pub(super) target_mask: Option<TargetMask>,
    /// File this loader was opened from, `None` for in-memory data.
    ///
    /// Kept so [`try_with_speech_layout`](Self::try_with_speech_layout) can find
    /// the corpus layout sidecar that sits beside it.
    source_path: Option<PathBuf>,
}

impl<R> LitDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// Create a new data loader from in-memory tokens (for testing or small datasets)
    pub fn new(tokens: Vec<u32>, batch_size: usize, seq_len: usize, device: R::Device) -> Self {
        Self::from_source(DataSource::InMemory(tokens), batch_size, seq_len, device)
    }

    fn from_source(
        data_source: DataSource,
        batch_size: usize,
        seq_len: usize,
        device: R::Device,
    ) -> Self {
        let len = data_source.len();
        Self {
            data_source,
            batch_size,
            seq_len,
            device,
            shard_start: 0,
            shard_end: len,
            position: DataPosition::default(),
            order: EpochOrder::new(RowOrder::Stream),
            target_mask: None,
            source_path: None,
        }
    }

    /// Create a data loader from a memory-mapped binary file
    ///
    /// # Arguments
    /// * `path` - Path to .bin file containing raw u32 tokens
    /// * `batch_size` - Rows per batch
    /// * `seq_len` - Sequence length; rows are `seq_len + 1` tokens
    /// * `device` - Target runtime device
    ///
    /// # File Format
    /// The .bin file should contain a contiguous array of u32 tokens in native endianness.
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        batch_size: usize,
        seq_len: usize,
        device: R::Device,
    ) -> Result<Self> {
        let path_ref = path.as_ref();
        let data_source = DataSource::map_file(path_ref)?;
        let mut loader = Self::from_source(data_source, batch_size, seq_len, device);
        loader.source_path = Some(path_ref.to_path_buf());
        Ok(loader)
    }

    /// Select how rows are ordered within each epoch.
    pub fn with_row_order(mut self, order: RowOrder) -> Self {
        self.order = EpochOrder::new(order);
        self
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
    /// If the data file has a `*.layout.json` sidecar (see
    /// [`crate::speech_corpus`]), it records the layout the corpus was PACKED
    /// with and MUST match `vocab`. Disagreement is a hard error: the ids would
    /// resolve to different codebooks and the mask would select different
    /// positions, with nothing else in the pipeline able to notice.
    ///
    /// Masking is driven by the layout itself, so EITHER layout can be checked
    /// here: a corpus packed as `expressive-tts-1.7b` is accepted when that is
    /// what the run configures, and rejected when it is not.
    ///
    /// A missing sidecar is not an error — corpora packed before layouts were
    /// recorded, and plain text corpora, have none.
    pub fn try_with_speech_layout(self, layout: SpeechLayout) -> Result<Self> {
        if let Some(path) = &self.source_path {
            crate::speech_corpus::check_corpus_layout(path, &layout)?;
        }
        Ok(self.with_speech_layout(layout))
    }

    /// Returns the total number of tokens in the dataset
    pub fn dataset_size(&self) -> usize {
        self.data_source.len()
    }

    /// Tokens per row: `seq_len` inputs plus the one extra target.
    pub(super) fn row_width(&self) -> usize {
        self.seq_len + 1
    }

    /// Row and batch counts of this loader's shard.
    pub fn row_layout(&self) -> Result<RowLayout> {
        RowLayout::new(
            (self.shard_end - self.shard_start) / self.row_width(),
            self.batch_size,
        )
    }

    /// Returns the shard size in tokens for this data loader
    ///
    /// This equals dataset_size.
    #[allow(dead_code)]
    pub fn shard_size(&self) -> usize {
        self.shard_end - self.shard_start
    }
}

// Helper to generate fake data for testing "zero to hero" pipeline
pub fn create_dummy_data(vocab_size: usize, total_tokens: usize) -> Vec<u32> {
    let mut data = Vec::with_capacity(total_tokens);
    for _ in 0..total_tokens {
        data.push((rand::random::<usize>() % vocab_size) as u32);
    }
    data
}

/// Write tokens to a binary file for use with memory-mapped data loader
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
    use std::io::Write;

    let mut file = File::create(path)?;

    // Write tokens as raw bytes (native endianness)
    let bytes = unsafe {
        std::slice::from_raw_parts(tokens.as_ptr() as *const u8, std::mem::size_of_val(tokens))
    };

    file.write_all(bytes)?;
    file.sync_all()?;

    Ok(())
}
