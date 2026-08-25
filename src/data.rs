use crate::prelude::{DType, Error, Result, Tensor};
use boostr::model::{SpeechLayout, SpeechVocab};
use boostr::runtime::Runtime;
use memmap2::Mmap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

// ============================================================================
// Data Loader Trait
// ============================================================================

/// One training batch: inputs, next-token targets, and an optional loss mask.
pub struct Batch<R: Runtime<DType = DType>> {
    /// Token ids `[batch_size, seq_len]`, I64.
    pub inputs: Tensor<R>,
    /// Next-token ids `[batch_size, seq_len]`, I64 (`inputs` shifted by one).
    pub targets: Tensor<R>,
    /// `1.0` at positions whose TARGET is an audio token, else `0.0`.
    /// Shape `[batch_size, seq_len]`, F32. `None` when audio-only masking is off.
    pub loss_mask: Option<Tensor<R>>,
}

/// Build the audio-only loss mask over a raw `[batch_size, seq_len + 1]` token
/// window.
///
/// The mask is derived from the POST-SHIFT targets (`slice[row_start + s + 1]`),
/// never from the inputs (`slice[row_start + s]`). Masking on inputs would ask
/// "is the current token audio" instead of "is the token being predicted audio",
/// shifting every masked-in span by exactly one position on every row.
///
/// Rows are GPT-style packed, so one row can hold several concatenated
/// text->audio examples; the mask is therefore per-token, never a single
/// boundary index per row.
///
/// Shared by `LitDataLoader::next_batch` and `PrefetchDataLoader::prefetch_worker`
/// so the two shift paths cannot diverge.
fn audio_target_mask(
    vocab: &SpeechVocab,
    slice: &[u32],
    batch_size: usize,
    seq_len: usize,
) -> Vec<f32> {
    let mut mask = Vec::with_capacity(batch_size * seq_len);
    for b in 0..batch_size {
        let row_start = b * (seq_len + 1);
        for s in 0..seq_len {
            let target = slice[row_start + s + 1];
            mask.push(if vocab.is_audio(target) { 1.0 } else { 0.0 });
        }
    }
    mask
}

/// Common interface for all data loaders
///
/// This trait allows the trainer to work with different data loader implementations:
/// - `LitDataLoader`: Basic synchronous loader
/// - `PrefetchDataLoader`: Async prefetching for CPU-GPU overlap
#[allow(dead_code)]
pub trait DataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// Fetch the next batch
    fn next_batch(&mut self) -> Result<Batch<R>>;

    /// Total number of tokens in the dataset
    fn dataset_size(&self) -> usize;
}

// ============================================================================
// LitDataLoader (Synchronous)
// ============================================================================

/// Production-grade data loader supporting both in-memory and memory-mapped datasets.
///
/// # Supported Formats
/// - `.bin` files: Raw binary token arrays (u32 or u16)
/// - In-memory: Vec<u32> for small datasets or testing
///
/// # Memory Mapping
/// For large datasets (>1GB), memory mapping is used automatically to avoid loading
/// the entire dataset into RAM. The OS handles paging transparently.
///
/// # Data Sharding
/// For multi-GPU training, use `with_sharding()` to ensure each GPU processes
/// a different portion of the dataset.
pub struct LitDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// Data source - either in-memory or memory-mapped
    data_source: DataSource,
    batch_size: usize,
    seq_len: usize,
    cursor: usize,
    device: R::Device,
    // Sharding support for multi-GPU training
    shard_start: usize,
    shard_end: usize,
    /// Speech id layout. `Some` turns on audio-only loss masking.
    speech_vocab: Option<SpeechVocab>,
    /// File this loader was opened from, `None` for in-memory data.
    ///
    /// Kept so [`try_with_speech_vocab`](Self::try_with_speech_vocab) can find
    /// the corpus layout sidecar that sits beside it.
    source_path: Option<PathBuf>,
}

enum DataSource {
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

impl<R> LitDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// Create a new data loader from in-memory tokens (for testing or small datasets)
    pub fn new(tokens: Vec<u32>, batch_size: usize, seq_len: usize, device: R::Device) -> Self {
        let len = tokens.len();
        let data_source = DataSource::InMemory(tokens);
        Self {
            data_source,
            batch_size,
            seq_len,
            cursor: 0,
            device,
            shard_start: 0,
            shard_end: len,
            speech_vocab: None,
            source_path: None,
        }
    }

    /// Create a data loader from a memory-mapped binary file
    ///
    /// # Arguments
    /// * `path` - Path to .bin file containing raw u32 tokens
    /// * `batch_size` - Micro-batch size
    /// * `seq_len` - Sequence length
    /// * `device` - Target runtime device
    ///
    /// # File Format
    /// The .bin file should contain a contiguous array of u32 tokens in native endianness.
    /// Example creation: `tokens.iter().collect::<Vec<u32>>().write_to_file("data.bin")`
    ///
    /// # Safety
    /// This function uses unsafe code to cast the memory-mapped region to a &'static [u32].
    /// This is safe because:
    /// 1. The Mmap is kept alive in the DataSource struct
    /// 2. The slice lifetime is tied to the Mmap lifetime via the struct
    /// 3. The file is assumed to be valid u32 data (caller responsibility)
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        batch_size: usize,
        seq_len: usize,
        device: R::Device,
    ) -> Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).map_err(|e| Error::DataError {
            reason: format!("failed to open data file {}: {e}", path_ref.display()),
        })?;

        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| Error::DataError {
                reason: format!("failed to mmap data file {}: {e}", path_ref.display()),
            })?
        };

        // Validate alignment and size
        if mmap.len() % std::mem::size_of::<u32>() != 0 {
            return Err(Error::DataError {
                reason: "file size is not a multiple of u32 size".to_string(),
            });
        }

        let len = mmap.len() / std::mem::size_of::<u32>();

        // SAFETY:
        // - mmap is kept alive in the struct
        // - alignment is verified above
        // - we're casting to &'static which is safe because _mmap outlives this reference
        let tokens: &'static [u32] =
            unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const u32, len) };

        let data_source = DataSource::MemoryMapped {
            _mmap: mmap,
            tokens,
            len,
        };

        Ok(Self {
            data_source,
            batch_size,
            seq_len,
            cursor: 0,
            device,
            shard_start: 0,
            shard_end: len,
            speech_vocab: None,
            source_path: Some(path_ref.to_path_buf()),
        })
    }

    /// Enable audio-only loss masking with this speech id layout.
    ///
    /// Without it the loader emits `loss_mask: None` and training scores every
    /// position, as it did before speech support existed.
    ///
    /// This form does NOT check the corpus layout sidecar. Prefer
    /// [`try_with_speech_vocab`](Self::try_with_speech_vocab) for any loader
    /// opened from a file.
    pub fn with_speech_vocab(mut self, vocab: SpeechVocab) -> Self {
        self.speech_vocab = Some(vocab);
        self
    }

    /// Enable audio-only loss masking, first checking the corpus layout sidecar.
    ///
    /// If the data file has a `*.layout.json` sidecar (see
    /// [`crate::speech_corpus`]), it records the layout the corpus was PACKED
    /// with and MUST match `vocab`. Disagreement is a hard error: the ids would
    /// resolve to different codebooks and the mask would select different
    /// positions, with nothing else in the pipeline able to notice.
    ///
    /// Masking is driven by a [`SpeechVocab`], so the layout this checks against
    /// is [`SpeechLayout::Native`]. A corpus packed under any other layout —
    /// `expressive-tts-1.7b`, say — therefore fails here, which is correct: its
    /// audio ids are not the ones `vocab` describes.
    ///
    /// A missing sidecar is not an error — corpora packed before layouts were
    /// recorded, and plain text corpora, have none.
    pub fn try_with_speech_vocab(self, vocab: SpeechVocab) -> Result<Self> {
        if let Some(path) = &self.source_path {
            let layout = SpeechLayout::Native(vocab.clone());
            crate::speech_corpus::check_corpus_layout(path, &layout)?;
        }
        Ok(self.with_speech_vocab(vocab))
    }

    /// Configure data sharding for multi-GPU training
    ///
    /// Each GPU should process a different shard of the dataset to avoid
    /// duplicate work. The shard boundaries are calculated based on rank
    /// and world_size.
    ///
    /// # Arguments
    ///
    /// * `rank` - This GPU's rank (0-indexed)
    /// * `world_size` - Total number of GPUs
    ///
    /// # Example
    ///
    /// ```ignore
    /// // For 4 GPUs, each GPU processes 1/4 of the dataset
    /// let loader = LitDataLoader::from_file(path, batch_size, seq_len, device)?
    ///     .with_sharding(rank, 4);
    /// ```
    #[allow(dead_code)]
    pub fn with_sharding(mut self, rank: usize, world_size: usize) -> Self {
        let total_size = self.dataset_size();
        let shard_size = total_size / world_size;

        self.shard_start = rank * shard_size;
        self.shard_end = if rank == world_size - 1 {
            total_size // Last rank gets any remainder
        } else {
            (rank + 1) * shard_size
        };

        // Reset cursor to shard start
        self.cursor = self.shard_start;

        self
    }

    /// Returns the total number of tokens in the dataset
    pub fn dataset_size(&self) -> usize {
        match &self.data_source {
            DataSource::InMemory(tokens) => tokens.len(),
            DataSource::MemoryMapped { len, .. } => *len,
        }
    }

    /// Get a slice of tokens at the current cursor position
    fn get_tokens(&self, start: usize, count: usize) -> &[u32] {
        match &self.data_source {
            DataSource::InMemory(tokens) => &tokens[start..start + count],
            DataSource::MemoryMapped { tokens, .. } => &tokens[start..start + count],
        }
    }

    /// Fetches the next batch.
    /// Returns a [`Batch`] whose `targets` are `inputs` shifted by 1.
    ///
    /// When sharding is enabled, the cursor wraps within the shard boundaries
    /// rather than the full dataset.
    pub fn next_batch(&mut self) -> Result<Batch<R>> {
        let n_tokens_needed = self.batch_size * (self.seq_len + 1);

        // Check shard boundaries (for multi-GPU training)
        if self.cursor + n_tokens_needed > self.shard_end {
            self.cursor = self.shard_start;
            // TODO: Optional shuffle logic for randomized training
        }

        let current_cursor = self.cursor;
        self.cursor += self.batch_size * self.seq_len; // Move cursor forward

        let slice = self.get_tokens(current_cursor, n_tokens_needed);

        // Token ids are stored as u32 on disk, but numr's indexing ops
        // (embedding lookup, cross-entropy targets) take I64 indices. Convert
        // once here at the data boundary rather than at each use site.
        let ids: Vec<i64> = slice.iter().map(|&t| t as i64).collect();

        // Convert to tensor
        let raw_tensor =
            Tensor::<R>::from_slice(&ids, &[self.batch_size, self.seq_len + 1], &self.device)?;

        // Split into inputs (0..end-1) and targets (1..end).
        // `narrow` returns a strided VIEW; downstream code reshapes these
        // (cross-entropy flattens to [B*S]), which requires a contiguous
        // layout. Materialize here so the loader never hands out a view that
        // breaks a later reshape.
        let inputs = raw_tensor.narrow(1, 0, self.seq_len)?.contiguous()?;
        let targets = raw_tensor.narrow(1, 1, self.seq_len)?.contiguous()?;

        // Built from the same `slice`, over the post-shift target window, so it
        // lines up element-for-element with `targets` above.
        let loss_mask = match &self.speech_vocab {
            Some(vocab) => {
                let mask = audio_target_mask(vocab, slice, self.batch_size, self.seq_len);
                Some(Tensor::<R>::from_slice(
                    &mask,
                    &[self.batch_size, self.seq_len],
                    &self.device,
                )?)
            }
            None => None,
        };

        Ok(Batch {
            inputs,
            targets,
            loss_mask,
        })
    }

    /// Returns the shard size for this data loader
    ///
    /// For single-GPU training, this equals dataset_size.
    /// For multi-GPU, this is the size of this GPU's shard.
    #[allow(dead_code)]
    pub fn shard_size(&self) -> usize {
        self.shard_end - self.shard_start
    }
}

impl<R> DataLoader<R> for LitDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    fn next_batch(&mut self) -> Result<Batch<R>> {
        LitDataLoader::next_batch(self)
    }

    fn dataset_size(&self) -> usize {
        LitDataLoader::dataset_size(self)
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

// ============================================================================
// Async Prefetching Data Pipeline
// ============================================================================

/// A batch ready for training: (inputs, targets, loss mask) on CPU
/// Will be transferred to GPU when consumed
struct PrefetchedBatch {
    /// Input token IDs as raw u32 vec (before GPU transfer)
    inputs: Vec<u32>,
    /// Target token IDs as raw u32 vec
    targets: Vec<u32>,
    /// Audio-only loss mask over the target positions, `None` when masking is off
    loss_mask: Option<Vec<f32>>,
    /// Batch dimensions
    batch_size: usize,
    seq_len: usize,
}

/// Async prefetching data loader for CPU-GPU pipeline overlap
///
/// This loader runs a background thread that prefetches batches while the GPU
/// is training on the current batch. This overlaps CPU I/O with GPU compute.
///
/// # Performance
///
/// Without prefetching:
/// ```text
/// GPU: [Forward/Backward on batch N] → [Wait for CPU] → [Forward/Backward on batch N+1]
/// CPU:                                  [Load batch N+1]
/// ```
///
/// With prefetching:
/// ```text
/// GPU: [Forward/Backward on batch N] → [Forward/Backward on batch N+1] → ...
/// CPU: [Load batch N+1]               → [Load batch N+2]                → ...
/// ```
///
/// Expected speedup: 5-15% for I/O-bound workloads
///
/// # Usage
///
/// ```ignore
/// let base_loader = LitDataLoader::from_file(path, batch_size, seq_len, device)?;
/// let mut prefetch_loader = PrefetchDataLoader::new(base_loader, 2)?; // prefetch 2 batches
///
/// for step in 0..max_steps {
///     let batch = prefetch_loader.next_batch()?;
///     // GPU training on batch.inputs/batch.targets while next batch is being loaded
/// }
///
/// // Background thread is joined automatically when `prefetch_loader` is dropped.
/// ```
#[allow(dead_code)]
pub struct PrefetchDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// Receiver for prefetched batches
    receiver: Receiver<PrefetchedBatch>,
    /// Handle to background prefetch thread
    prefetch_thread: Option<JoinHandle<()>>,
    /// Signal to stop prefetching
    stop_signal: Arc<AtomicBool>,
    /// Target device for tensors
    device: R::Device,
    /// Batch dimensions (for tensor creation)
    batch_size: usize,
    seq_len: usize,
    /// Dataset size (for calculating epochs)
    dataset_size: usize,
}

impl<R> PrefetchDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// Create a new prefetching data loader
    ///
    /// # Arguments
    ///
    /// * `base_loader` - The underlying LitDataLoader to wrap
    /// * `prefetch_count` - Number of batches to prefetch (default: 2)
    ///
    /// # Notes
    ///
    /// Higher prefetch_count uses more CPU memory but provides better overlap.
    /// Typically 2-3 is sufficient for most workloads.
    pub fn new(base_loader: LitDataLoader<R>, prefetch_count: usize) -> Result<Self> {
        let device = base_loader.device.clone();
        let batch_size = base_loader.batch_size;
        let seq_len = base_loader.seq_len;
        let dataset_size = base_loader.dataset_size(); // Store before moving

        // Channel for passing prefetched batches (bounded to limit memory)
        let (sender, receiver) = mpsc::sync_channel(prefetch_count);

        let stop_signal = Arc::new(AtomicBool::new(false));
        let stop_signal_clone = Arc::clone(&stop_signal);

        // Spawn background prefetch thread
        let prefetch_thread = thread::spawn(move || {
            Self::prefetch_worker(base_loader, sender, stop_signal_clone);
        });

        Ok(Self {
            receiver,
            prefetch_thread: Some(prefetch_thread),
            stop_signal,
            device,
            batch_size,
            seq_len,
            dataset_size,
        })
    }

    /// Background worker that continuously prefetches batches
    fn prefetch_worker(
        mut loader: LitDataLoader<R>,
        sender: SyncSender<PrefetchedBatch>,
        stop_signal: Arc<AtomicBool>,
    ) {
        loop {
            if stop_signal.load(Ordering::Relaxed) {
                break;
            }

            // Get raw token data (on CPU, no GPU transfer yet)
            let n_tokens_needed = loader.batch_size * (loader.seq_len + 1);

            // Handle epoch wraparound
            if loader.cursor + n_tokens_needed > loader.shard_end {
                loader.cursor = loader.shard_start;
            }

            let current_cursor = loader.cursor;
            loader.cursor += loader.batch_size * loader.seq_len;

            // Get raw token slice
            let slice = loader.get_tokens(current_cursor, n_tokens_needed);

            // Same helper as the synchronous path, over the same post-shift
            // target window, so the two loaders cannot produce different masks.
            let loss_mask = loader
                .speech_vocab
                .as_ref()
                .map(|vocab| audio_target_mask(vocab, slice, loader.batch_size, loader.seq_len));

            // Prepare inputs and targets (still on CPU as Vec<u32>)
            let mut inputs = Vec::with_capacity(loader.batch_size * loader.seq_len);
            let mut targets = Vec::with_capacity(loader.batch_size * loader.seq_len);

            for b in 0..loader.batch_size {
                let batch_start = b * (loader.seq_len + 1);
                for s in 0..loader.seq_len {
                    inputs.push(slice[batch_start + s]);
                    targets.push(slice[batch_start + s + 1]);
                }
            }

            let batch = PrefetchedBatch {
                inputs,
                targets,
                loss_mask,
                batch_size: loader.batch_size,
                seq_len: loader.seq_len,
            };

            // Send to main thread (blocks if channel is full - backpressure)
            if sender.send(batch).is_err() {
                break; // Channel closed, exit worker
            }
        }
    }

    /// Get the next batch (transfers prefetched data to GPU)
    ///
    /// This is the main method called in the training loop.
    /// It receives a prefetched batch from the background thread and
    /// transfers it to the target device (GPU).
    pub fn next_batch(&mut self) -> Result<Batch<R>> {
        // Receive prefetched batch (blocks if none available)
        let batch = self.receiver.recv().map_err(|_| Error::DataError {
            reason: "prefetch thread stopped".to_string(),
        })?;

        // Token ids are stored as u32, but numr's indexing ops (embedding
        // lookup, cross-entropy targets) take I64 indices.
        let input_ids: Vec<i64> = batch.inputs.iter().map(|&t| t as i64).collect();
        let target_ids: Vec<i64> = batch.targets.iter().map(|&t| t as i64).collect();

        // Transfer to the runtime device (this is the only blocking device upload)
        let inputs =
            Tensor::<R>::from_slice(&input_ids, &[batch.batch_size, batch.seq_len], &self.device)?;
        let targets = Tensor::<R>::from_slice(
            &target_ids,
            &[batch.batch_size, batch.seq_len],
            &self.device,
        )?;

        let loss_mask = match &batch.loss_mask {
            Some(mask) => Some(Tensor::<R>::from_slice(
                mask,
                &[batch.batch_size, batch.seq_len],
                &self.device,
            )?),
            None => None,
        };

        Ok(Batch {
            inputs,
            targets,
            loss_mask,
        })
    }

    /// Get batch size
    #[allow(dead_code)]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get sequence length
    #[allow(dead_code)]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get dataset size
    #[allow(dead_code)]
    pub fn dataset_size(&self) -> usize {
        self.dataset_size
    }
}

impl<R> DataLoader<R> for PrefetchDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    fn next_batch(&mut self) -> Result<Batch<R>> {
        PrefetchDataLoader::next_batch(self)
    }

    fn dataset_size(&self) -> usize {
        self.dataset_size
    }
}

impl<R> Drop for PrefetchDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    fn drop(&mut self) {
        self.stop_signal.store(true, Ordering::Relaxed);

        // Take ownership of the thread handle and join
        if let Some(handle) = self.prefetch_thread.take() {
            // The worker parks in `sender.send` once the bounded channel is
            // full, and the receiver is still alive here (it is dropped only
            // after this function returns), so joining straight away
            // deadlocks. Drain buffered batches until the worker observes the
            // stop signal and exits.
            while !handle.is_finished() {
                match self.receiver.recv_timeout(Duration::from_millis(10)) {
                    Ok(_) => {}
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
            let _ = handle.join();
        }
    }
}

/// Create a prefetching data loader from a file path (convenience function)
///
/// # Example
///
/// ```ignore
/// let mut loader = create_prefetch_loader(
///     "data/train.bin",
///     batch_size,
///     seq_len,
///     device,
///     2,  // prefetch 2 batches
/// )?;
/// ```
#[allow(dead_code)]
pub fn create_prefetch_loader<R, P>(
    path: P,
    batch_size: usize,
    seq_len: usize,
    device: R::Device,
    prefetch_count: usize,
) -> Result<PrefetchDataLoader<R>>
where
    R: Runtime<DType = DType>,
    P: AsRef<Path>,
{
    let base_loader = LitDataLoader::<R>::from_file(path, batch_size, seq_len, device)?;
    PrefetchDataLoader::new(base_loader, prefetch_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::model::{CodecVocab, SpeechVocab, ALL_SPECIAL_TOKENS};
    use boostr::{CpuDevice, CpuRuntime};

    const TEXT_VOCAB: usize = 16;
    /// Must hold all 13 control tokens.
    const CONTROL_REGION: usize = 16;

    /// Toy layout: text ids [0, 16), control ids [16, 32), audio ids [32, 40).
    fn toy_vocab() -> SpeechVocab {
        let codec = CodecVocab::new(2, 4).expect("valid codec");
        SpeechVocab::with_sequential_specials(
            TEXT_VOCAB,
            CONTROL_REGION,
            &ALL_SPECIAL_TOKENS,
            codec,
        )
        .expect("valid vocab")
    }

    fn audio_base() -> u32 {
        (TEXT_VOCAB + CONTROL_REGION) as u32
    }

    fn mask_of(batch: &Batch<CpuRuntime>) -> Vec<f32> {
        batch
            .loss_mask
            .as_ref()
            .expect("loss mask present")
            .contiguous()
            .expect("contiguous mask")
            .to_vec()
    }

    fn loader(tokens: Vec<u32>, batch_size: usize, seq_len: usize) -> LitDataLoader<CpuRuntime> {
        LitDataLoader::<CpuRuntime>::new(tokens, batch_size, seq_len, CpuDevice::new())
            .with_speech_vocab(toy_vocab())
    }

    /// The mask must answer "is the token being PREDICTED audio", so it is built
    /// from the post-shift targets. Building it from `inputs` instead would move
    /// the first `1.0` one position later, which is what this asserts against.
    #[test]
    fn mask_is_derived_from_targets_not_inputs() {
        let base = audio_base();
        // inputs  = [0, 1, 2, base+0, base+1]
        // targets = [1, 2, base+0, base+1, base+2]
        let tokens = vec![0, 1, 2, base, base + 1, base + 2];
        let batch = loader(tokens, 1, 5).next_batch().expect("batch");

        let mask = mask_of(&batch);
        assert_eq!(mask, vec![0.0, 0.0, 1.0, 1.0, 1.0]);

        let first_one = mask
            .iter()
            .position(|v| *v == 1.0)
            .expect("a masked-in position");
        assert_eq!(
            first_one, 2,
            "mask must start where the TARGET is the first audio token (index 2), \
             not where the INPUT is (index 3)"
        );

        // Sanity: the target at that index really is the first audio token.
        let targets: Vec<i64> = batch.targets.contiguous().expect("contiguous").to_vec();
        assert_eq!(targets[first_one], base as i64);
    }

    /// GPT-style packing puts several text->audio examples in one row, so the
    /// mask must be per-token and produce one span per example.
    #[test]
    fn packed_row_with_two_segments_produces_two_mask_spans() {
        let base = audio_base();
        // targets = [1, base, base, 3, 4, base, base, 5]
        let tokens = vec![0, 1, base, base, 3, 4, base, base, 5];
        let batch = loader(tokens, 1, 8).next_batch().expect("batch");

        assert_eq!(
            mask_of(&batch),
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        );
    }

    #[test]
    fn no_speech_vocab_means_no_loss_mask() {
        let base = audio_base();
        let tokens = vec![0, 1, base, base + 1, 2, 3];
        let mut l = LitDataLoader::<CpuRuntime>::new(tokens, 1, 5, CpuDevice::new());
        let batch = l.next_batch().expect("batch");
        assert!(batch.loss_mask.is_none());
        assert_eq!(batch.inputs.shape(), [1usize, 5]);
        assert_eq!(batch.targets.shape(), [1usize, 5]);
    }

    #[test]
    fn prefetch_loader_mask_matches_sync_loader() {
        let base = audio_base();
        let tokens: Vec<u32> = (0..64)
            .map(|i| {
                if i % 3 == 0 {
                    base + (i % 8) as u32
                } else {
                    (i % 16) as u32
                }
            })
            .collect();

        let sync_batch = loader(tokens.clone(), 2, 6)
            .next_batch()
            .expect("sync batch");

        let mut prefetch = PrefetchDataLoader::<CpuRuntime>::new(loader(tokens, 2, 6), 1)
            .expect("prefetch loader");
        let prefetch_batch = prefetch.next_batch().expect("prefetch batch");

        assert_eq!(mask_of(&sync_batch), mask_of(&prefetch_batch));
        let sync_targets: Vec<i64> = sync_batch.targets.contiguous().expect("c").to_vec();
        let prefetch_targets: Vec<i64> = prefetch_batch.targets.contiguous().expect("c").to_vec();
        assert_eq!(sync_targets, prefetch_targets);
    }
}
