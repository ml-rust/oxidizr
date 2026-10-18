//! Where a [`LitDataLoader`] stands in its row stream: reading its position,
//! resuming at a saved one, and drawing the raw token window for the next
//! batch.

use super::construction::LitDataLoader;
use super::rows::DataPosition;
use crate::prelude::{DType, Error, Result};
use boostr::runtime::Runtime;

impl<R> LitDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// The next batch's place in the row order.
    pub fn position(&self) -> DataPosition {
        self.position
    }

    /// Continue from `position`, as saved by an earlier run over the same data.
    pub fn seek(&mut self, position: DataPosition) -> Result<()> {
        let layout = self.row_layout()?;
        if position.row > layout.num_rows() {
            return Err(Error::DataError {
                reason: format!(
                    "data position epoch {} row {} is past this shard's {} rows: the \
                     checkpoint was saved over different data or a different \
                     model.max_seq_len. Resume with the original data and config, or \
                     start a fresh run.",
                    position.epoch,
                    position.row,
                    layout.num_rows()
                ),
            });
        }
        self.position = position;
        Ok(())
    }

    /// Draws the next batch's rows as one `[batch_size, seq_len + 1]` window
    /// and advances the position.
    pub(super) fn next_window(&mut self) -> Result<Vec<u32>> {
        let layout = self.row_layout()?;
        let start = layout.batch_start(self.position);
        let width = self.row_width();

        let mut window = Vec::with_capacity(self.batch_size * width);
        for slot in start.row..start.row + self.batch_size {
            let row = self.order.row(&layout, start.epoch, slot);
            let offset = (self.first_row + row) * width;
            window.extend_from_slice(self.data_source.tokens(offset, width));
        }

        self.position = DataPosition {
            epoch: start.epoch,
            row: start.row + self.batch_size,
        };
        Ok(window)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::data::mask::tests::{ids, mask_of, prefix_lm_loader, EOS, PREFIX_LM_TARGET_BASE};
    use crate::data::{Batch, RowOrder};
    use boostr::{CpuDevice, CpuRuntime};
    use std::collections::HashSet;

    pub(in crate::data) const SEQ_LEN: usize = 3;
    pub(in crate::data) const WIDTH: usize = SEQ_LEN + 1;

    /// `rows` rows where every token is its own stream offset, plus a
    /// `tail` shorter than a row.
    pub(in crate::data) fn offsets(rows: usize, tail: usize) -> Vec<u32> {
        (0..(rows * WIDTH + tail) as u32).collect()
    }

    pub(in crate::data) fn shuffled(
        tokens: Vec<u32>,
        batch_size: usize,
        seed: u64,
    ) -> LitDataLoader<CpuRuntime> {
        LitDataLoader::<CpuRuntime>::new(tokens, batch_size, SEQ_LEN, CpuDevice::new())
            .with_row_order(RowOrder::Shuffled { seed })
    }

    /// The row index of each row of a batch built by [`offsets`], checking
    /// that every row is `WIDTH` consecutive tokens starting at a multiple of
    /// `WIDTH`.
    pub(in crate::data) fn rows_of(batch: &Batch<CpuRuntime>) -> Vec<usize> {
        let inputs = ids(&batch.inputs);
        let targets = ids(&batch.targets);
        inputs
            .chunks(SEQ_LEN)
            .zip(targets.chunks(SEQ_LEN))
            .map(|(inp, tgt)| {
                let start = inp[0] as usize;
                assert_eq!(start % WIDTH, 0, "row starts at offset {start}");
                let expected: Vec<i64> = (start..start + WIDTH).map(|t| t as i64).collect();
                assert_eq!(inp, &expected[..SEQ_LEN]);
                assert_eq!(tgt, &expected[1..]);
                start / WIDTH
            })
            .collect()
    }

    pub(in crate::data) fn epoch_rows(
        l: &mut LitDataLoader<CpuRuntime>,
        batches: usize,
    ) -> Vec<usize> {
        (0..batches)
            .flat_map(|_| rows_of(&l.next_batch().expect("batch")))
            .collect()
    }

    #[test]
    fn each_epoch_covers_every_row_once_without_overlap() {
        let mut l = shuffled(offsets(12, 2), 3, 5);
        assert_eq!(l.row_layout().expect("layout").batches_per_epoch(), 4);
        for epoch in 0..3 {
            let mut rows = epoch_rows(&mut l, 4);
            rows.sort_unstable();
            assert_eq!(rows, (0..12).collect::<Vec<_>>(), "epoch {epoch}");
        }
    }

    #[test]
    fn partial_batch_rows_are_dropped_and_never_repeated_within_an_epoch() {
        // 10 rows, batch 3: 3 batches per epoch, 1 row dropped each epoch.
        let mut l = shuffled(offsets(10, 0), 3, 5);
        for _ in 0..3 {
            let rows = epoch_rows(&mut l, 3);
            assert_eq!(rows.len(), 9);
            assert_eq!(rows.iter().collect::<HashSet<_>>().len(), 9);
            assert!(rows.iter().all(|r| *r < 10));
        }
    }

    #[test]
    fn tail_shorter_than_a_row_is_never_read() {
        let mut l = LitDataLoader::<CpuRuntime>::new(offsets(4, 3), 2, SEQ_LEN, CpuDevice::new());
        assert_eq!(l.row_layout().expect("layout").num_rows(), 4);
        let rows = epoch_rows(&mut l, 4);
        assert_eq!(rows, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn stream_order_hands_out_rows_in_order_every_epoch() {
        let mut l = LitDataLoader::<CpuRuntime>::new(offsets(6, 0), 2, SEQ_LEN, CpuDevice::new())
            .with_row_order(RowOrder::Stream);
        assert_eq!(epoch_rows(&mut l, 3), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(epoch_rows(&mut l, 3), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn shuffle_is_fixed_by_seed_and_differs_across_epochs() {
        let mut a = shuffled(offsets(32, 0), 4, 42);
        let mut b = shuffled(offsets(32, 0), 4, 42);
        let a0 = epoch_rows(&mut a, 8);
        let a1 = epoch_rows(&mut a, 8);
        assert_eq!(epoch_rows(&mut b, 8), a0);
        assert_eq!(epoch_rows(&mut b, 8), a1);
        assert_ne!(a0, a1, "each epoch draws a new order");
        assert_ne!(
            a0,
            (0..32).collect::<Vec<_>>(),
            "shuffled is not stream order"
        );

        let mut c = shuffled(offsets(32, 0), 4, 43);
        assert_ne!(epoch_rows(&mut c, 8), a0, "another seed, another order");
    }

    #[test]
    fn resume_mid_epoch_continues_with_the_uninterrupted_rows() {
        let mut uninterrupted = shuffled(offsets(10, 1), 3, 9);
        let layout = uninterrupted.row_layout().expect("layout");
        // Stop after 4 batches: epoch 1, one batch in.
        let stopped_after = 4;
        let _ = epoch_rows(&mut uninterrupted, stopped_after);
        let position = layout.advance(DataPosition::default(), stopped_after);
        assert_eq!(position, uninterrupted.position());
        assert_eq!(position, DataPosition { epoch: 1, row: 3 });

        let mut resumed = shuffled(offsets(10, 1), 3, 9);
        resumed.seek(position).expect("seek within the shard");
        assert_eq!(
            epoch_rows(&mut resumed, 7),
            epoch_rows(&mut uninterrupted, 7)
        );
    }

    #[test]
    fn seek_past_the_shard_is_an_error() {
        let mut l = shuffled(offsets(6, 0), 2, 1);
        let err = l
            .seek(DataPosition { epoch: 0, row: 7 })
            .expect_err("row 7 of 6");
        assert!(
            err.to_string().contains("past this shard's 6 rows"),
            "{err}"
        );
    }

    #[test]
    fn a_row_range_reads_only_its_own_rows() {
        let mut l = shuffled(offsets(12, 2), 2, 5)
            .with_row_range(3, 9)
            .expect("range inside the data");
        assert_eq!(l.row_layout().expect("layout").num_rows(), 6);
        assert_eq!(l.total_rows(), 12);
        for epoch in 0..2 {
            let mut rows = epoch_rows(&mut l, 3);
            rows.sort_unstable();
            assert_eq!(rows, (3..9).collect::<Vec<_>>(), "epoch {epoch}");
        }
    }

    #[test]
    fn a_full_row_range_reads_what_the_default_reads() {
        let mut full = shuffled(offsets(10, 3), 2, 4);
        let mut ranged = shuffled(offsets(10, 3), 2, 4)
            .with_row_range(0, 10)
            .expect("full range");
        assert_eq!(epoch_rows(&mut ranged, 15), epoch_rows(&mut full, 15));
    }

    #[test]
    fn split_ranges_are_disjoint_and_cover_the_rows() {
        let mut train = shuffled(offsets(10, 0), 2, 8)
            .with_row_range(0, 7)
            .expect("train range");
        let mut eval =
            LitDataLoader::<CpuRuntime>::new(offsets(10, 0), 1, SEQ_LEN, CpuDevice::new())
                .with_row_range(7, 10)
                .expect("eval range");
        let train_rows: HashSet<usize> = epoch_rows(&mut train, 9).into_iter().collect();
        let eval_rows: HashSet<usize> = epoch_rows(&mut eval, 3).into_iter().collect();
        assert!(
            train_rows.is_disjoint(&eval_rows),
            "{train_rows:?} {eval_rows:?}"
        );
        assert_eq!(eval_rows, (7..10).collect::<HashSet<usize>>());
        assert!(train_rows.iter().all(|r| *r < 7));
    }

    #[test]
    fn a_row_range_outside_the_data_is_an_error() {
        let l = || LitDataLoader::<CpuRuntime>::new(offsets(6, 0), 2, SEQ_LEN, CpuDevice::new());
        for (start, end) in [(0, 7), (4, 4), (5, 2)] {
            let Err(err) = l().with_row_range(start, end) else {
                panic!("range [{start}, {end}) of 6 rows must be refused");
            };
            assert!(err.to_string().contains("6 whole rows"), "{err}");
        }
    }

    #[test]
    fn target_mask_lines_up_with_shuffled_rows() {
        // Rows of `WIDTH` tokens alternating prompt ids and continuation ids.
        let tokens: Vec<u32> = (0..16 * WIDTH as u32)
            .map(|i| {
                if (i / 2) % 2 == 0 {
                    4 + i % 8
                } else {
                    12 + i % 8
                }
            })
            .collect();
        let mut l =
            prefix_lm_loader(tokens, 4, SEQ_LEN).with_row_order(RowOrder::Shuffled { seed: 3 });
        let scored = |t: &i64| *t >= i64::from(PREFIX_LM_TARGET_BASE) || *t == i64::from(EOS);
        for _ in 0..8 {
            let batch = l.next_batch().expect("batch");
            let expected: Vec<f32> = ids(&batch.targets)
                .iter()
                .map(|t| if scored(t) { 1.0 } else { 0.0 })
                .collect();
            assert!(expected.contains(&0.0) && expected.contains(&1.0));
            assert_eq!(mask_of(&batch), expected);
        }
    }
}
