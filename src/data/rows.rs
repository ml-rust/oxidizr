//! Row geometry of a packed token stream, the per-epoch row order, and the
//! stream position a checkpoint records.
//!
//! A row is `seq_len + 1` consecutive tokens at a fixed offset: row `r` of a
//! shard is tokens `[shard_start + r * (seq_len + 1), shard_start + (r + 1) * (seq_len + 1))`.
//! A batch is `batch_size` rows. Each epoch hands out every row once, in the
//! order [`EpochOrder`] gives.

use crate::prelude::{Error, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Where a row loader stands in its stream.
///
/// `row` counts the slots of epoch `epoch`'s row order already handed out.
/// When fewer than `batch_size` slots remain, the next batch starts at slot 0
/// of epoch `epoch + 1`. `DataPosition::default()` is the stream start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DataPosition {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Slots of this epoch's row order already consumed.
    pub row: usize,
}

/// Row and batch counts of one loader shard.
///
/// Each epoch yields `num_rows / batch_size` batches. The last
/// `num_rows % batch_size` slots of an epoch's order are dropped. Under
/// shuffling those are different rows each epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowLayout {
    num_rows: usize,
    batch_size: usize,
}

impl RowLayout {
    /// Checks that at least one whole batch fits in `num_rows`.
    ///
    /// Row indices are stored as `u32` in the shuffled order, so `num_rows`
    /// must fit in a `u32`.
    pub fn new(num_rows: usize, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::DataError {
                reason: "trainer.batch_size is 0: a batch needs at least one row. \
                         Set trainer.batch_size to a positive number."
                    .to_string(),
            });
        }
        if num_rows < batch_size {
            return Err(Error::DataError {
                reason: format!(
                    "data shard holds {num_rows} whole rows, fewer than batch_size \
                     {batch_size}, so no batch fits. Use more data, a smaller \
                     trainer.batch_size, or a smaller model.max_seq_len."
                ),
            });
        }
        if u32::try_from(num_rows).is_err() {
            return Err(Error::DataError {
                reason: format!(
                    "data shard holds {num_rows} rows, more than the {} a row order can \
                     index. Split the data file, or shard it across more ranks.",
                    u32::MAX
                ),
            });
        }
        Ok(Self {
            num_rows,
            batch_size,
        })
    }

    /// Whole rows in the shard.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Rows per batch.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Whole batches per epoch. The partial batch at an epoch's end is dropped.
    pub fn batches_per_epoch(&self) -> usize {
        self.num_rows / self.batch_size
    }

    /// Where the next batch drawn at `pos` starts.
    pub fn batch_start(&self, pos: DataPosition) -> DataPosition {
        if pos.row + self.batch_size > self.num_rows {
            DataPosition {
                epoch: pos.epoch + 1,
                row: 0,
            }
        } else {
            pos
        }
    }

    /// The position after drawing `batches` batches from `pos`.
    ///
    /// Equal to the loader's position after `batches` calls to `next_batch`.
    pub fn advance(&self, pos: DataPosition, batches: usize) -> DataPosition {
        if batches == 0 {
            return pos;
        }
        let start = self.batch_start(pos);
        let left_in_epoch = (self.num_rows - start.row) / self.batch_size;
        if batches <= left_in_epoch {
            return DataPosition {
                epoch: start.epoch,
                row: start.row + batches * self.batch_size,
            };
        }
        // Past this epoch: `rest >= 1` batches start at the next epoch's slot 0.
        let rest = batches - left_in_epoch;
        let per_epoch = self.batches_per_epoch();
        let full_epochs = (rest - 1) / per_epoch;
        let into_last = (rest - 1) % per_epoch + 1;
        DataPosition {
            epoch: start.epoch + 1 + full_epochs,
            row: into_last * self.batch_size,
        }
    }

    /// Optimizer steps in `num_epochs` epochs at `grad_accum` batches per step.
    ///
    /// A final step that would need batches past the last epoch is not run.
    pub fn total_steps(&self, grad_accum: usize, num_epochs: usize) -> usize {
        self.batches_per_epoch() * num_epochs / grad_accum.max(1)
    }

    /// The position after `steps` optimizer steps from the stream start.
    pub fn position_after_steps(&self, steps: usize, grad_accum: usize) -> DataPosition {
        self.advance(DataPosition::default(), steps * grad_accum)
    }
}

/// How rows are ordered within an epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowOrder {
    /// Row `i` at slot `i`: stream order, the same every epoch.
    Stream,
    /// A permutation seeded from `seed` and the epoch index: reproducible
    /// for a seed, different each epoch.
    Shuffled {
        /// Base seed, mixed with the epoch index per epoch.
        seed: u64,
    },
}

/// The row at each slot of each epoch, with the current epoch's permutation
/// cached.
#[derive(Debug, Clone)]
pub(crate) struct EpochOrder {
    order: RowOrder,
    /// `(epoch, num_rows, permutation)` of the last epoch asked for.
    cached: Option<(usize, usize, Vec<u32>)>,
}

impl EpochOrder {
    pub(crate) fn new(order: RowOrder) -> Self {
        Self {
            order,
            cached: None,
        }
    }

    /// Shard-relative index of the row at `slot` of `epoch`'s order.
    ///
    /// `layout` bounds `slot` below `num_rows` and `num_rows` to `u32`.
    pub(crate) fn row(&mut self, layout: &RowLayout, epoch: usize, slot: usize) -> usize {
        let seed = match self.order {
            RowOrder::Stream => return slot,
            RowOrder::Shuffled { seed } => seed,
        };
        let num_rows = layout.num_rows();
        let fresh = !matches!(&self.cached, Some((e, n, _)) if *e == epoch && *n == num_rows);
        if fresh {
            self.cached = Some((epoch, num_rows, epoch_permutation(num_rows, seed, epoch)));
        }
        match &self.cached {
            Some((_, _, perm)) => perm[slot] as usize,
            None => slot,
        }
    }
}

/// A permutation of `0..num_rows` for `epoch`, seeded from `seed` and `epoch`.
fn epoch_permutation(num_rows: usize, seed: u64, epoch: usize) -> Vec<u32> {
    let mut perm: Vec<u32> = (0..num_rows as u32).collect();
    perm.shuffle(&mut StdRng::seed_from_u64(epoch_seed(seed, epoch)));
    perm
}

/// Mixes the epoch index into the base seed (golden-ratio increment), so
/// each epoch of a run draws an independent permutation.
fn epoch_seed(seed: u64, epoch: usize) -> u64 {
    seed ^ (epoch as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(num_rows: usize, batch_size: usize) -> RowLayout {
        RowLayout::new(num_rows, batch_size).expect("valid layout")
    }

    /// Steps `batches` batches one at a time, as the loader does.
    fn step_by_step(l: &RowLayout, mut pos: DataPosition, batches: usize) -> DataPosition {
        for _ in 0..batches {
            let start = l.batch_start(pos);
            pos = DataPosition {
                epoch: start.epoch,
                row: start.row + l.batch_size(),
            };
        }
        pos
    }

    #[test]
    fn advance_matches_one_batch_at_a_time() {
        for (rows, bs) in [(10, 3), (9, 3), (7, 1), (5, 5), (11, 4)] {
            let l = layout(rows, bs);
            for start_row in 0..=rows {
                let start = DataPosition {
                    epoch: 2,
                    row: start_row,
                };
                for n in 0..(4 * l.batches_per_epoch() + 3) {
                    assert_eq!(
                        l.advance(start, n),
                        step_by_step(&l, start, n),
                        "rows={rows} bs={bs} start_row={start_row} n={n}"
                    );
                }
            }
        }
    }

    #[test]
    fn partial_batch_at_epoch_end_is_dropped() {
        let l = layout(10, 3);
        assert_eq!(l.batches_per_epoch(), 3);
        assert_eq!(l.total_steps(1, 2), 6);
        assert_eq!(
            l.batch_start(DataPosition { epoch: 0, row: 9 }),
            DataPosition { epoch: 1, row: 0 }
        );
    }

    #[test]
    fn total_steps_divides_batches_by_accumulation() {
        let l = layout(12, 2);
        assert_eq!(l.total_steps(2, 3), 9);
        assert_eq!(l.total_steps(4, 1), 1);
    }

    #[test]
    fn too_few_rows_for_one_batch_is_an_error() {
        let err = RowLayout::new(3, 4).expect_err("3 rows cannot fill a batch of 4");
        assert!(err.to_string().contains("fewer than batch_size 4"), "{err}");
        assert!(RowLayout::new(3, 0).is_err());
    }

    #[test]
    fn shuffled_order_is_a_permutation_fixed_by_seed_and_epoch() {
        let l = layout(50, 5);
        let mut a = EpochOrder::new(RowOrder::Shuffled { seed: 7 });
        let mut b = EpochOrder::new(RowOrder::Shuffled { seed: 7 });
        let order = |o: &mut EpochOrder, epoch| -> Vec<usize> {
            (0..50).map(|slot| o.row(&l, epoch, slot)).collect()
        };

        let e0 = order(&mut a, 0);
        let mut sorted = e0.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        assert_ne!(
            e0,
            (0..50).collect::<Vec<_>>(),
            "seeded order is not stream order"
        );

        assert_eq!(order(&mut b, 0), e0, "same seed and epoch, same order");
        assert_ne!(order(&mut a, 1), e0, "a new epoch draws a new order");
        assert_eq!(
            order(&mut a, 0),
            e0,
            "an epoch's order is recomputed exactly"
        );

        let mut c = EpochOrder::new(RowOrder::Shuffled { seed: 8 });
        assert_ne!(
            order(&mut c, 0),
            e0,
            "a different seed draws a different order"
        );
    }

    #[test]
    fn stream_order_is_identity_every_epoch() {
        let l = layout(8, 2);
        let mut o = EpochOrder::new(RowOrder::Stream);
        for epoch in 0..3 {
            let rows: Vec<usize> = (0..8).map(|slot| o.row(&l, epoch, slot)).collect();
            assert_eq!(rows, (0..8).collect::<Vec<_>>());
        }
    }
}
