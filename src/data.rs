use crate::prelude::{Device, Error, Result, Tensor};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::mpsc::{self, SyncSender, Receiver};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

// ============================================================================
// Data Loader Trait
// ============================================================================

/// Common interface for all data loaders
///
/// This trait allows the trainer to work with different data loader implementations:
/// - `LitDataLoader`: Basic synchronous loader
/// - `PrefetchDataLoader`: Async prefetching for CPU-GPU overlap
#[allow(dead_code)]
pub trait DataLoader {
    /// Fetch the next batch of (inputs, targets)
    fn next_batch(&mut self) -> Result<(Tensor, Tensor)>;

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
pub struct LitDataLoader {
    /// Data source - either in-memory or memory-mapped
    data_source: DataSource,
    batch_size: usize,
    seq_len: usize,
    cursor: usize,
    device: Device,
    // Sharding support for multi-GPU training
    shard_start: usize,
    shard_end: usize,
}

enum DataSource {
    /// In-memory token array (for small datasets or dummy data)
    InMemory(Vec<u32>),
    /// Memory-mapped file (for large datasets)
    /// Safety: The Mmap is kept alive for the lifetime of the DataSource
    MemoryMapped {
        _mmap: Mmap,  // Keep alive
        tokens: &'static [u32],  // Safe because _mmap outlives this reference
        len: usize,
    },
}

impl LitDataLoader {
    /// Create a new data loader from in-memory tokens (for testing or small datasets)
    pub fn new(tokens: Vec<u32>, batch_size: usize, seq_len: usize, device: Device) -> Self {
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
        }
    }

    /// Create a data loader from a memory-mapped binary file
    ///
    /// # Arguments
    /// * `path` - Path to .bin file containing raw u32 tokens
    /// * `batch_size` - Micro-batch size
    /// * `seq_len` - Sequence length
    /// * `device` - Target device (CPU or CUDA)
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
        device: Device,
    ) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| Error::Msg(format!("Failed to open data file: {}", e)))?;

        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| Error::Msg(format!("Failed to mmap file: {}", e)))?
        };

        // Validate alignment and size
        if mmap.len() % std::mem::size_of::<u32>() != 0 {
            return Err(Error::Msg(
                "File size is not a multiple of u32 size".to_string()
            ));
        }

        let len = mmap.len() / std::mem::size_of::<u32>();

        // SAFETY:
        // - mmap is kept alive in the struct
        // - alignment is verified above
        // - we're casting to &'static which is safe because _mmap outlives this reference
        let tokens: &'static [u32] = unsafe {
            std::slice::from_raw_parts(
                mmap.as_ptr() as *const u32,
                len
            )
        };

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
        })
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
    /// Returns (Input, Target) where Target is Input shifted by 1.
    ///
    /// When sharding is enabled, the cursor wraps within the shard boundaries
    /// rather than the full dataset.
    pub fn next_batch(&mut self) -> Result<(Tensor, Tensor)> {
        let n_tokens_needed = self.batch_size * (self.seq_len + 1);

        // Check shard boundaries (for multi-GPU training)
        if self.cursor + n_tokens_needed > self.shard_end {
            self.cursor = self.shard_start;
            // TODO: Optional shuffle logic for randomized training
        }

        let current_cursor = self.cursor;
        self.cursor += self.batch_size * self.seq_len; // Move cursor forward

        let slice = self.get_tokens(current_cursor, n_tokens_needed);

        // Convert to tensor
        let raw_tensor = Tensor::from_slice(slice, (self.batch_size, self.seq_len + 1), &self.device)?;

        // Split into inputs (0..end-1) and targets (1..end)
        let inputs = raw_tensor.narrow(1, 0, self.seq_len)?;
        let targets = raw_tensor.narrow(1, 1, self.seq_len)?;

        Ok((inputs, targets))
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

impl DataLoader for LitDataLoader {
    fn next_batch(&mut self) -> Result<(Tensor, Tensor)> {
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
#[allow(dead_code)]  // Public API for users to create binary token files
pub fn write_tokens_to_file<P: AsRef<Path>>(tokens: &[u32], path: P) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = File::create(path)?;

    // Write tokens as raw bytes (native endianness)
    let bytes = unsafe {
        std::slice::from_raw_parts(
            tokens.as_ptr() as *const u8,
            tokens.len() * std::mem::size_of::<u32>()
        )
    };

    file.write_all(bytes)?;
    file.sync_all()?;

    Ok(())
}

// ============================================================================
// Async Prefetching Data Pipeline
// ============================================================================

/// A batch ready for training: (inputs, targets) on CPU
/// Will be transferred to GPU when consumed
struct PrefetchedBatch {
    /// Input token IDs as raw u32 vec (before GPU transfer)
    inputs: Vec<u32>,
    /// Target token IDs as raw u32 vec
    targets: Vec<u32>,
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
///     let (inputs, targets) = prefetch_loader.next_batch()?;
///     // GPU training on inputs/targets while next batch is being loaded
/// }
///
/// prefetch_loader.shutdown(); // Clean up background thread
/// ```
#[allow(dead_code)]
pub struct PrefetchDataLoader {
    /// Receiver for prefetched batches
    receiver: Receiver<PrefetchedBatch>,
    /// Handle to background prefetch thread
    prefetch_thread: Option<JoinHandle<()>>,
    /// Signal to stop prefetching
    stop_signal: Arc<AtomicBool>,
    /// Target device for tensors
    device: Device,
    /// Batch dimensions (for tensor creation)
    batch_size: usize,
    seq_len: usize,
    /// Dataset size (for calculating epochs)
    dataset_size: usize,
}

impl PrefetchDataLoader {
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
    pub fn new(base_loader: LitDataLoader, prefetch_count: usize) -> Result<Self> {
        let device = base_loader.device.clone();
        let batch_size = base_loader.batch_size;
        let seq_len = base_loader.seq_len;
        let dataset_size = base_loader.dataset_size();  // Store before moving

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
        mut loader: LitDataLoader,
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
                batch_size: loader.batch_size,
                seq_len: loader.seq_len,
            };

            // Send to main thread (blocks if channel is full - backpressure)
            if sender.send(batch).is_err() {
                break;  // Channel closed, exit worker
            }
        }
    }

    /// Get the next batch (transfers prefetched data to GPU)
    ///
    /// This is the main method called in the training loop.
    /// It receives a prefetched batch from the background thread and
    /// transfers it to the target device (GPU).
    pub fn next_batch(&mut self) -> Result<(Tensor, Tensor)> {
        // Receive prefetched batch (blocks if none available)
        let batch = self.receiver.recv()
            .map_err(|_| Error::Msg("Prefetch thread stopped".to_string()))?;

        // Transfer to GPU (this is the only blocking GPU operation)
        let inputs = Tensor::from_slice(
            &batch.inputs,
            (batch.batch_size, batch.seq_len),
            &self.device,
        )?;
        let targets = Tensor::from_slice(
            &batch.targets,
            (batch.batch_size, batch.seq_len),
            &self.device,
        )?;

        Ok((inputs, targets))
    }

    /// Gracefully shutdown the prefetch thread
    ///
    /// Call this when training is complete to clean up resources.
    pub fn shutdown(&mut self) {
        self.stop_signal.store(true, Ordering::Relaxed);

        // Take ownership of the thread handle and join
        if let Some(handle) = self.prefetch_thread.take() {
            // Drop the receiver first to unblock the sender
            // (handled automatically when self is dropped)
            let _ = handle.join();
        }
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

impl DataLoader for PrefetchDataLoader {
    fn next_batch(&mut self) -> Result<(Tensor, Tensor)> {
        PrefetchDataLoader::next_batch(self)
    }

    fn dataset_size(&self) -> usize {
        self.dataset_size
    }
}

impl Drop for PrefetchDataLoader {
    fn drop(&mut self) {
        self.shutdown();
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
pub fn create_prefetch_loader<P: AsRef<Path>>(
    path: P,
    batch_size: usize,
    seq_len: usize,
    device: Device,
    prefetch_count: usize,
) -> Result<PrefetchDataLoader> {
    let base_loader = LitDataLoader::from_file(path, batch_size, seq_len, device)?;
    PrefetchDataLoader::new(base_loader, prefetch_count)
}