//! Document segments of one batch: the builder over a token window and the device copy.
//!
//! A separator token ends each document. The segment of a document ends at,
//! and includes, its separator. Every row starts a new segment, so a row
//! never continues the segment of the row before it.

use crate::prelude::{DType, Error, Result, Tensor};
use boostr::ops::PackedSegments;
use boostr::runtime::Runtime;
use std::sync::Arc;

/// Document segments of one batch, on the batch's device.
///
/// Cloning shares one allocation, so a checkpointed layer keeps the
/// segments for its recompute at no copy.
pub struct BatchSegments<R: Runtime> {
    inner: Arc<SegmentParts<R>>,
}

struct SegmentParts<R: Runtime> {
    packed: PackedSegments<R>,
    segment_ids: Tensor<R>,
}

impl<R: Runtime> Clone for BatchSegments<R> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<R: Runtime> BatchSegments<R> {
    /// Segment offsets over the flattened `batch_size * seq_len` input tokens.
    pub fn packed(&self) -> &PackedSegments<R> {
        &self.inner.packed
    }

    /// `[batch_size, seq_len]` I32 segment id of each input token.
    pub fn segment_ids(&self) -> &Tensor<R> {
        &self.inner.segment_ids
    }
}

/// Host copy of the segments of one window, before the upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HostSegments {
    /// Cumulative offsets over the flattened inputs, from 0 to `batch_size * seq_len`.
    offsets: Vec<i32>,
    /// Segment index of each input token, counted over the whole batch.
    ids: Vec<i32>,
    batch_size: usize,
    seq_len: usize,
}

impl HostSegments {
    /// The segments of the inputs of a `[batch_size, seq_len + 1]` `window`.
    ///
    /// # Errors
    /// Returns an error when `batch_size * seq_len` does not fit in `i32`.
    pub(super) fn build(
        window: &[u32],
        batch_size: usize,
        seq_len: usize,
        separator: u32,
    ) -> Result<Self> {
        let to_i32 = |n: usize| {
            i32::try_from(n).map_err(|_| Error::DataError {
                reason: format!(
                    "a batch of {batch_size} x {seq_len} tokens overflows the i32 segment \
                     offsets. Lower trainer.batch_size or model.max_seq_len."
                ),
            })
        };
        // Every row closes one segment, so `batch_size + 1` offsets is the floor.
        let mut offsets = Vec::with_capacity(batch_size + 1);
        offsets.push(0i32);
        let mut ids = Vec::with_capacity(batch_size * seq_len);
        for (b, row) in window
            .chunks_exact(seq_len + 1)
            .take(batch_size)
            .enumerate()
        {
            let row_start = b * seq_len;
            for (s, &token) in row[..seq_len].iter().enumerate() {
                ids.push(to_i32(offsets.len() - 1)?);
                if token == separator {
                    offsets.push(to_i32(row_start + s + 1)?);
                }
            }
            let row_end = to_i32(row_start + seq_len)?;
            if offsets.last() != Some(&row_end) {
                offsets.push(row_end);
            }
        }
        Ok(Self {
            offsets,
            ids,
            batch_size,
            seq_len,
        })
    }

    /// Uploads the offsets and the `[batch_size, seq_len]` ids to `device`.
    ///
    /// # Errors
    /// Returns an error when an upload fails.
    pub(super) fn upload<R: Runtime<DType = DType>>(
        &self,
        device: &R::Device,
    ) -> Result<BatchSegments<R>> {
        let total = self.batch_size * self.seq_len;
        let packed = PackedSegments::<R>::from_offsets(&self.offsets, total, device)?;
        let segment_ids =
            Tensor::<R>::from_slice(&self.ids, &[self.batch_size, self.seq_len], device)?;
        Ok(BatchSegments {
            inner: Arc::new(SegmentParts {
                packed,
                segment_ids,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEP: u32 = 0;

    fn host(window: &[u32], batch_size: usize, seq_len: usize) -> HostSegments {
        HostSegments::build(window, batch_size, seq_len, SEP).expect("segments")
    }

    #[test]
    fn a_separator_ends_its_segment_and_belongs_to_it() {
        // inputs [5, 6, 0, 7, 8], target 9 dropped from the inputs.
        let segs = host(&[5, 6, SEP, 7, 8, 9], 1, 5);
        assert_eq!(segs.offsets, vec![0, 3, 5]);
        assert_eq!(segs.ids, vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn every_row_start_begins_a_segment() {
        // Row 0 inputs [5, 6, 7], row 1 inputs [8, 0, 9]: no separator closes row 0.
        let segs = host(&[5, 6, 7, 1, 8, SEP, 9, 2], 2, 3);
        assert_eq!(segs.offsets, vec![0, 3, 5, 6]);
        assert_eq!(segs.ids, vec![0, 0, 0, 1, 1, 2]);
    }

    #[test]
    fn consecutive_separators_make_one_token_segments() {
        let segs = host(&[SEP, SEP, 5, 6], 1, 3);
        assert_eq!(segs.offsets, vec![0, 1, 2, 3]);
        assert_eq!(segs.ids, vec![0, 1, 2]);
    }

    #[test]
    fn a_row_ending_on_a_separator_adds_no_empty_segment() {
        let segs = host(&[5, SEP, 3, 6, SEP, 4], 2, 2);
        assert_eq!(segs.offsets, vec![0, 2, 4]);
    }

    #[test]
    fn no_separator_means_one_segment_per_row() {
        let segs = host(&[5, 6, 7, 8, 9, 10], 2, 2);
        assert_eq!(segs.offsets, vec![0, 2, 4]);
        assert_eq!(segs.ids, vec![0, 0, 1, 1]);
    }
}
