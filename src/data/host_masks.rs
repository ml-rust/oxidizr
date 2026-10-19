//! The host-side loss mask and document segments both loaders build per window.
//!
//! A position whose input is the document separator is unscored: its target
//! is the next document's first token. That rule ANDs with the target mask.

use super::construction::LitDataLoader;
use super::mask::{build_target_mask, scored_count};
use super::segments::HostSegments;
use crate::config::DocumentMaskRecord;
use crate::prelude::{DType, Result};
use boostr::runtime::Runtime;

/// The host-side loss mask and segments of one window.
pub(super) struct HostMasks {
    /// `None` when neither a target mask nor a document mask is on.
    pub(super) loss_mask: Option<Vec<f32>>,
    /// Positions the loss scores.
    pub(super) scored: usize,
    /// `Some` exactly when the document mask is on.
    pub(super) segments: Option<HostSegments>,
}

impl<R> LitDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// Turns on the document mask with the separator `separator_id`.
    ///
    /// Each batch then carries [`super::BatchSegments`], and its loss mask
    /// leaves out every position whose input is the separator.
    pub fn with_document_mask(mut self, separator_id: u32) -> Self {
        self.document_mask = Some(separator_id);
        self
    }

    /// The document mask a checkpoint meta records. `None` while it is off.
    pub fn document_mask_record(&self) -> Option<DocumentMaskRecord> {
        self.document_mask
            .map(|separator_id| DocumentMaskRecord { separator_id })
    }

    /// The loss mask and segments of `window`. Both loaders build every batch through it.
    ///
    /// The target mask scores by target id. The document mask then clears
    /// each position whose input is the separator. The two combine by AND.
    ///
    /// # Errors
    /// Returns an error when `batch_size * seq_len` overflows the i32 segment offsets.
    pub(super) fn host_masks(&self, window: &[u32]) -> Result<HostMasks> {
        let (batch_size, seq_len) = (self.batch_size, self.seq_len);
        let target = self
            .target_mask
            .as_ref()
            .map(|mask| build_target_mask(mask, window, batch_size, seq_len));
        let Some(separator) = self.document_mask else {
            let scored = target.as_deref().map_or(batch_size * seq_len, scored_count);
            return Ok(HostMasks {
                loss_mask: target,
                scored,
                segments: None,
            });
        };
        let mut mask = target.unwrap_or_else(|| vec![1.0; batch_size * seq_len]);
        let inputs = window
            .chunks_exact(seq_len + 1)
            .flat_map(|row| &row[..seq_len]);
        for (m, &input) in mask.iter_mut().zip(inputs) {
            if input == separator {
                *m = 0.0;
            }
        }
        Ok(HostMasks {
            scored: scored_count(&mask),
            loss_mask: Some(mask),
            segments: Some(HostSegments::build(window, batch_size, seq_len, separator)?),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::config::DocumentMaskRecord;
    use crate::data::mask::tests::{mask_of, prefix_lm_loader, PREFIX_LM_TARGET_BASE};
    use crate::data::{Batch, LitDataLoader, PrefetchDataLoader, RowOrder};
    use boostr::{CpuDevice, CpuRuntime};

    const SEP: u32 = 0;

    fn offsets_of(batch: &Batch<CpuRuntime>) -> Vec<i32> {
        let segments = batch.segments.as_ref().expect("segments present");
        segments.packed().host_cu_seqlens().to_vec()
    }

    fn ids_of(batch: &Batch<CpuRuntime>) -> Vec<i32> {
        let segments = batch.segments.as_ref().expect("segments present");
        segments
            .segment_ids()
            .contiguous()
            .expect("contiguous")
            .to_vec()
    }

    fn doc_loader(
        tokens: Vec<u32>,
        batch_size: usize,
        seq_len: usize,
    ) -> LitDataLoader<CpuRuntime> {
        LitDataLoader::<CpuRuntime>::new(tokens, batch_size, seq_len, CpuDevice::new())
            .with_document_mask(SEP)
    }

    #[test]
    fn a_separator_input_is_unscored_and_scored_is_recounted() {
        let batch = doc_loader(vec![5, 6, SEP, 7, 8, 9], 1, 5)
            .next_batch()
            .expect("batch");
        assert_eq!(mask_of(&batch), vec![1.0, 1.0, 0.0, 1.0, 1.0]);
        assert_eq!(batch.scored, 4);
        assert_eq!(offsets_of(&batch), vec![0, 3, 5]);
        assert_eq!(ids_of(&batch), vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn the_document_mask_ands_with_the_target_mask() {
        let base = PREFIX_LM_TARGET_BASE;
        // targets [base, 0, base + 1, 4]: the prefix-LM mask scores positions 0 and 2.
        let tokens = vec![4, base, SEP, base + 1, 4];
        let batch = prefix_lm_loader(tokens, 1, 4)
            .with_document_mask(SEP)
            .next_batch()
            .expect("batch");
        // Input 2 is the separator, so position 2 drops out.
        assert_eq!(mask_of(&batch), vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(batch.scored, 1);
    }

    #[test]
    fn off_leaves_no_segments() {
        let batch = LitDataLoader::<CpuRuntime>::new(vec![5, SEP, 6], 1, 2, CpuDevice::new())
            .next_batch()
            .expect("batch");
        assert!(batch.segments.is_none() && batch.loss_mask.is_none());
        assert_eq!(batch.scored, 2);
    }

    #[test]
    fn the_prefetch_loader_yields_the_sync_segments() {
        let tokens: Vec<u32> = (0..80)
            .map(|i| if i % 7 == 3 { SEP } else { 1 + i % 9 })
            .collect();
        let make =
            || doc_loader(tokens.clone(), 2, 5).with_row_order(RowOrder::Shuffled { seed: 3 });
        let mut sync = make();
        let batches = 2 * sync.row_layout().expect("layout").batches_per_epoch();
        let mut prefetch = PrefetchDataLoader::<CpuRuntime>::new(make(), 2).expect("prefetch");
        for n in 0..batches {
            let a = sync.next_batch().expect("sync batch");
            let b = prefetch.next_batch().expect("prefetch batch");
            assert_eq!(offsets_of(&a), offsets_of(&b), "batch {n}");
            assert_eq!(ids_of(&a), ids_of(&b), "batch {n}");
            assert_eq!(mask_of(&a), mask_of(&b), "batch {n}");
            assert_eq!(a.scored, b.scored, "batch {n}");
        }
    }

    /// Each rank builds the segments of its own rows. Row starts split
    /// segments, so the rank offsets, shifted, concatenate to the global ones.
    #[test]
    fn rank_segments_concatenate_to_the_global_segments() {
        let tokens: Vec<u32> = (0..96)
            .map(|i| if i % 5 == 2 { SEP } else { 1 + i % 11 })
            .collect();
        let order = RowOrder::Shuffled { seed: 9 };
        let (batch, world, seq_len) = (2, 2, 3);
        let mut global = doc_loader(tokens.clone(), batch * world, seq_len).with_row_order(order);
        let mut ranks: Vec<_> = (0..world)
            .map(|rank| {
                doc_loader(tokens.clone(), batch, seq_len)
                    .with_row_order(order)
                    .with_rank(rank, world)
                    .expect("rank")
            })
            .collect();
        for n in 0..6 {
            let mut joined = vec![0i32];
            let mut scored = 0;
            for (rank, l) in ranks.iter_mut().enumerate() {
                let part = l.next_batch().expect("rank batch");
                let shift = (rank * batch * seq_len) as i32;
                joined.extend(offsets_of(&part)[1..].iter().map(|o| o + shift));
                scored += part.scored;
            }
            let whole = global.next_batch().expect("global batch");
            assert_eq!(joined, offsets_of(&whole), "step {n}");
            assert_eq!(scored, whole.scored, "step {n}");
        }
    }

    #[test]
    fn the_record_names_the_separator() {
        let l = doc_loader(vec![1, 2, 3], 1, 2);
        assert_eq!(
            l.document_mask_record(),
            Some(DocumentMaskRecord { separator_id: SEP })
        );
    }
}
