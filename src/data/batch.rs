//! Assembling one training batch: converting a row window to device tensors
//! and building its loss mask.

use super::construction::LitDataLoader;
use super::mask::{build_target_mask, scored_count};
use crate::prelude::{DType, Result, Tensor};
use boostr::runtime::Runtime;

/// One training batch: inputs, next-token targets, and an optional loss mask.
pub struct Batch<R: Runtime<DType = DType>> {
    /// Token ids `[batch_size, seq_len]`, I64.
    pub inputs: Tensor<R>,
    /// Next-token ids `[batch_size, seq_len]`, I64 (`inputs` shifted by one).
    pub targets: Tensor<R>,
    /// `1.0` at positions the loss scores, else `0.0`.
    /// Shape `[batch_size, seq_len]`, F32. `None` when masking is off (every
    /// position is scored).
    pub loss_mask: Option<Tensor<R>>,
    /// Number of positions the loss scores, counted from the host copy of
    /// `loss_mask`. `batch_size * seq_len` when masking is off.
    pub scored: usize,
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

impl<R> LitDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// Fetches the next batch.
    /// Returns a [`Batch`] whose `targets` are `inputs` shifted by 1.
    pub fn next_batch(&mut self) -> Result<Batch<R>> {
        let window = self.next_window()?;

        // Token ids are stored as u32 on disk, but numr's indexing ops
        // (embedding lookup, cross-entropy targets) take I64 indices. Convert
        // once here at the data boundary rather than at each use site.
        let ids: Vec<i64> = window.iter().map(|&t| t as i64).collect();

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

        // Built from the same `window`, over the post-shift target window, so it
        // lines up element-for-element with `targets` above.
        let (loss_mask, scored) = match &self.target_mask {
            Some(target_mask) => {
                let mask = build_target_mask(target_mask, &window, self.batch_size, self.seq_len);
                let scored = scored_count(&mask);
                let tensor =
                    Tensor::<R>::from_slice(&mask, &[self.batch_size, self.seq_len], &self.device)?;
                (Some(tensor), scored)
            }
            None => (None, self.batch_size * self.seq_len),
        };

        Ok(Batch {
            inputs,
            targets,
            loss_mask,
            scored,
        })
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
