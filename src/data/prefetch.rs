//! Background-thread prefetching over a [`LitDataLoader`].

use super::batch::{Batch, DataLoader};
use super::construction::LitDataLoader;
use super::mask::{build_target_mask, scored_count};
use crate::prelude::{DType, Error, Result, Tensor};
use boostr::runtime::Runtime;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// A batch ready for training: (inputs, targets, loss mask) on CPU
/// Will be transferred to GPU when consumed
struct PrefetchedBatch {
    /// Input token IDs as raw u32 vec (before GPU transfer)
    inputs: Vec<u32>,
    /// Target token IDs as raw u32 vec
    targets: Vec<u32>,
    /// Loss mask over the target positions, `None` when masking is off
    loss_mask: Option<Vec<f32>>,
    /// Positions the loss scores, see [`Batch::scored`]
    scored: usize,
    /// Batch dimensions
    batch_size: usize,
    seq_len: usize,
}

/// What the worker sends: a batch, or the loader's error text. The text
/// crosses the thread boundary and becomes a `DataError` on the receiving side.
type Prefetched = std::result::Result<PrefetchedBatch, String>;

/// Async prefetching data loader for CPU-GPU pipeline overlap
///
/// This loader runs a background thread that prefetches batches while the GPU
/// is training on the current batch. This overlaps CPU I/O with GPU compute.
///
/// The worker draws batches from the wrapped loader in order, so the batch
/// sequence equals the wrapped loader's. Set the wrapped loader's rank and
/// seek it before wrapping it. Each rank of a data-parallel run wraps its
/// own loader.
///
/// The worker loads batch `N + 1` while the device trains on batch `N`.
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
    receiver: Receiver<Prefetched>,
    /// Handle to background prefetch thread
    prefetch_thread: Option<JoinHandle<()>>,
    /// Signal to stop prefetching
    stop_signal: Arc<AtomicBool>,
    /// Target device for tensors
    device: R::Device,
    /// Batch dimensions (for tensor creation)
    batch_size: usize,
    seq_len: usize,
    /// Dataset size in tokens
    dataset_size: usize,
}

impl<R> PrefetchDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// Wraps `base_loader` and keeps up to `prefetch_count` batches ready.
    ///
    /// A higher `prefetch_count` holds more batches in host memory.
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
        sender: SyncSender<Prefetched>,
        stop_signal: Arc<AtomicBool>,
    ) {
        while !stop_signal.load(Ordering::Relaxed) {
            let item = Self::prefetch_one(&mut loader).map_err(|e| e.to_string());
            let failed = item.is_err();
            // Blocks while the channel is full (backpressure). A closed
            // channel means the consumer is gone.
            if sender.send(item).is_err() || failed {
                break;
            }
        }
    }

    /// Draws one batch on CPU: the same row window and mask builder as the
    /// synchronous path, so the two loaders cannot produce different batches.
    fn prefetch_one(loader: &mut LitDataLoader<R>) -> Result<PrefetchedBatch> {
        let (batch_size, seq_len) = (loader.batch_size, loader.seq_len);
        let window = loader.next_window()?;

        let loss_mask = loader
            .target_mask
            .as_ref()
            .map(|mask| build_target_mask(mask, &window, batch_size, seq_len));
        let scored = loss_mask
            .as_deref()
            .map_or(batch_size * seq_len, scored_count);

        let mut inputs = Vec::with_capacity(batch_size * seq_len);
        let mut targets = Vec::with_capacity(batch_size * seq_len);
        for row in window.chunks_exact(seq_len + 1) {
            inputs.extend_from_slice(&row[..seq_len]);
            targets.extend_from_slice(&row[1..]);
        }

        Ok(PrefetchedBatch {
            inputs,
            targets,
            loss_mask,
            scored,
            batch_size,
            seq_len,
        })
    }

    /// Get the next batch (transfers prefetched data to GPU)
    ///
    /// This is the main method called in the training loop.
    /// It receives a prefetched batch from the background thread and
    /// transfers it to the target device (GPU).
    pub fn next_batch(&mut self) -> Result<Batch<R>> {
        // Receive prefetched batch (blocks if none available)
        let batch = self
            .receiver
            .recv()
            .map_err(|_| Error::DataError {
                reason: "prefetch thread stopped before sending the next batch. Rerun \
                         with --prefetch 0 to load batches on the training thread and \
                         see the underlying error."
                    .to_string(),
            })?
            .map_err(|reason| Error::DataError { reason })?;

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
            scored: batch.scored,
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
    use crate::data::mask::tests::{audio_base, ids, loader, mask_of, prefix_lm_loader, EOS};
    use crate::data::RowOrder;
    use boostr::CpuRuntime;

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
        assert_eq!(ids(&sync_batch.targets), ids(&prefetch_batch.targets));
    }

    #[test]
    fn prefetch_loader_mask_matches_sync_loader_for_prefix_lm() {
        let tokens = vec![
            0, 1, 4, 5, 6, 7, 2, 12, 13, 14, 15, 16, 17, 18, EOS, 0, 1, 5, 2, 13, EOS,
        ];

        let sync_batch = prefix_lm_loader(tokens.clone(), 1, 20)
            .next_batch()
            .expect("sync batch");

        let mut prefetch =
            PrefetchDataLoader::<CpuRuntime>::new(prefix_lm_loader(tokens, 1, 20), 1)
                .expect("prefetch loader");
        let prefetch_batch = prefetch.next_batch().expect("prefetch batch");

        assert_eq!(mask_of(&sync_batch), mask_of(&prefetch_batch));
    }

    /// Over several shuffled epochs, a seeked prefetch loader yields exactly
    /// the synchronous loader's batches.
    #[test]
    fn prefetch_loader_matches_sync_loader_across_shuffled_epochs_after_seek() {
        // Prompt ids (4..12) every third token, continuation ids (12..20) otherwise.
        let tokens: Vec<u32> = (0..70)
            .map(|i| if i % 3 == 0 { 4 + i % 8 } else { 12 + i % 8 })
            .collect();
        let order = RowOrder::Shuffled { seed: 11 };
        let make = || prefix_lm_loader(tokens.clone(), 2, 4).with_row_order(order);

        let mut sync = make();
        let layout = sync.row_layout().expect("layout");
        let start = layout.advance(Default::default(), 5);
        sync.seek(start).expect("seek");
        let mut base = make();
        base.seek(start).expect("seek");
        let mut prefetch = PrefetchDataLoader::<CpuRuntime>::new(base, 2).expect("prefetch");

        for _ in 0..3 * layout.batches_per_epoch() {
            let a = sync.next_batch().expect("sync batch");
            let b = prefetch.next_batch().expect("prefetch batch");
            assert_eq!(ids(&a.inputs), ids(&b.inputs));
            assert_eq!(ids(&a.targets), ids(&b.targets));
            assert_eq!(mask_of(&a), mask_of(&b));
            assert_eq!(a.scored, b.scored);
        }
    }

    #[test]
    fn prefetch_loader_reports_a_loader_error() {
        // 5 tokens hold one row of 5; a batch needs 2 rows.
        let mut prefetch =
            PrefetchDataLoader::<CpuRuntime>::new(prefix_lm_loader(vec![0; 5], 2, 4), 1)
                .expect("prefetch loader");
        let Err(err) = prefetch.next_batch() else {
            panic!("a shard with fewer rows than batch_size must not yield a batch");
        };
        assert!(err.to_string().contains("fewer than batch_size 2"), "{err}");
    }
}
