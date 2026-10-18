//! The data-parallel rank of a [`LitDataLoader`]: which slots of each global
//! batch it reads.

use super::construction::LitDataLoader;
use super::rows::{zero_world_error, DataPosition};
use crate::prelude::{DType, Error, Result};
use boostr::runtime::Runtime;

impl<R> LitDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// Make this loader rank `rank` of `world_size` data-parallel ranks.
    ///
    /// Every rank keeps the whole row range and the same row order. One
    /// global batch is `batch_size * world_size` slots. Rank `rank` reads
    /// slots `[rank * batch_size, (rank + 1) * batch_size)` of it. Rows past
    /// the last whole global batch of an epoch are dropped for every rank.
    /// The position resets to the stream start.
    ///
    /// Rank 0 of a world of 1 is the default and reads every slot.
    ///
    /// # Errors
    /// Returns an error when `world_size` is 0 or `rank` is not below it.
    pub fn with_rank(mut self, rank: usize, world_size: usize) -> Result<Self> {
        if world_size == 0 {
            return Err(zero_world_error());
        }
        if rank >= world_size {
            return Err(Error::DataError {
                reason: format!(
                    "rank {rank} is not below world size {world_size}. Pass a rank in \
                     [0, {world_size})."
                ),
            });
        }
        self.rank = rank;
        self.world_size = world_size;
        self.position = DataPosition::default();
        Ok(self)
    }

    /// This loader's data-parallel rank, below [`world_size`](Self::world_size).
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Ranks that split each global batch. 1 without data parallelism.
    pub fn world_size(&self) -> usize {
        self.world_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::positioning::tests::{epoch_rows, offsets, rows_of, shuffled, SEQ_LEN};
    use crate::data::RowOrder;
    use boostr::{CpuDevice, CpuRuntime};
    use std::collections::HashSet;

    /// `world` loaders of `tokens`, one per rank, each with batch `batch_size`.
    fn rank_loaders(
        tokens: &[u32],
        batch_size: usize,
        world: usize,
        seed: u64,
    ) -> Vec<LitDataLoader<CpuRuntime>> {
        (0..world)
            .map(|rank| {
                shuffled(tokens.to_vec(), batch_size, seed)
                    .with_rank(rank, world)
                    .expect("valid rank")
            })
            .collect()
    }

    /// One step of every rank: their batch rows in rank order.
    fn union_step(ranks: &mut [LitDataLoader<CpuRuntime>]) -> Vec<usize> {
        ranks
            .iter_mut()
            .flat_map(|l| rows_of(&l.next_batch().expect("rank batch")))
            .collect()
    }

    #[test]
    fn rank_windows_concatenate_to_the_global_window() {
        // 23 rows, 2 ranks x 3: 3 global batches per epoch, 5 rows dropped.
        let tokens = offsets(23, 2);
        let mut global = shuffled(tokens.clone(), 6, 13);
        let mut ranks = rank_loaders(&tokens, 3, 2, 13);
        assert_eq!(
            ranks[1].row_layout().expect("layout").batches_per_epoch(),
            3
        );
        for step in 0..10 {
            let expected = rows_of(&global.next_batch().expect("global batch"));
            assert_eq!(union_step(&mut ranks), expected, "step {step}");
            for rank in &ranks {
                assert_eq!(rank.position(), global.position(), "step {step}");
            }
        }
    }

    #[test]
    fn every_rank_drops_the_same_remainder_and_runs_the_same_steps() {
        // 11 rows, 3 ranks x 1: 3 steps per epoch, 2 rows dropped.
        let tokens = offsets(11, 0);
        let mut ranks = rank_loaders(&tokens, 1, 3, 4);
        for epoch in 0..3 {
            let rows: Vec<usize> = (0..3).flat_map(|_| union_step(&mut ranks)).collect();
            assert_eq!(
                rows.iter().collect::<HashSet<_>>().len(),
                9,
                "epoch {epoch}"
            );
            for rank in &ranks {
                assert_eq!(rank.position().epoch, epoch, "epoch {epoch}");
                assert_eq!(rank.position().row, 9, "epoch {epoch}");
            }
        }
    }

    #[test]
    fn world_one_rank_zero_is_the_default_loader() {
        let mut plain = shuffled(offsets(10, 3), 3, 2);
        let mut ranked = shuffled(offsets(10, 3), 3, 2)
            .with_rank(0, 1)
            .expect("rank 0 of 1");
        assert_eq!(ranked.row_layout().ok(), plain.row_layout().ok());
        assert_eq!((ranked.rank(), ranked.world_size()), (0, 1));
        assert_eq!((plain.rank(), plain.world_size()), (0, 1));
        assert_eq!(epoch_rows(&mut ranked, 9), epoch_rows(&mut plain, 9));
        assert_eq!(ranked.position(), plain.position());
    }

    #[test]
    fn rank_outside_the_world_is_an_error() {
        let l = || LitDataLoader::<CpuRuntime>::new(offsets(8, 0), 2, SEQ_LEN, CpuDevice::new());
        let Err(err) = l().with_rank(2, 2) else {
            panic!("rank 2 of 2 must be refused");
        };
        let text = err.to_string();
        assert!(text.contains("rank 2 is not below world size 2"), "{text}");
        let Err(err) = l().with_rank(0, 0) else {
            panic!("world size 0 must be refused");
        };
        assert!(err.to_string().contains("world size is 0"), "{err}");
    }

    #[test]
    fn too_few_rows_for_one_global_batch_is_an_error() {
        let mut l = LitDataLoader::<CpuRuntime>::new(offsets(5, 0), 2, SEQ_LEN, CpuDevice::new())
            .with_rank(1, 3)
            .expect("rank 1 of 3");
        let Err(err) = l.next_batch() else {
            panic!("5 rows cannot fill 3 ranks x 2");
        };
        assert!(
            err.to_string().contains("one global batch of 6 rows"),
            "{err}"
        );
    }

    #[test]
    fn ranks_share_a_row_range() {
        let ranged = |batch_size: usize| {
            LitDataLoader::<CpuRuntime>::new(offsets(20, 1), batch_size, SEQ_LEN, CpuDevice::new())
                .with_row_order(RowOrder::Shuffled { seed: 6 })
                .with_row_range(3, 17)
                .expect("range inside the data")
        };
        let mut ranks: Vec<_> = (0..2)
            .map(|rank| ranged(2).with_rank(rank, 2).expect("valid rank"))
            .collect();
        let mut global = ranged(4);
        assert_eq!(ranks[0].row_range(), (3, 17));
        for step in 0..9 {
            let rows = union_step(&mut ranks);
            assert!(rows.iter().all(|r| (3..17).contains(r)), "{rows:?}");
            let expected = rows_of(&global.next_batch().expect("batch"));
            assert_eq!(rows, expected, "step {step}");
        }
    }
}
