//! A [`LitDataLoader`] over several sources: opening token files in order,
//! reading several row ranges, and the sidecar vocabulary check.

use super::construction::LitDataLoader;
use super::row_ranges::RowRanges;
use super::rows::DataPosition;
use super::source::RowStore;
use crate::prelude::{DType, Result};
use boostr::runtime::Runtime;
use std::path::Path;

impl<R> LitDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// Create a data loader over the token files at `paths`, read in order.
    ///
    /// The stream is file 0's whole rows, then file 1's, and so on. Each file
    /// drops its own tail shorter than a row. A file's `.tokens.json`
    /// sidecar, when present, must record the file's token count.
    ///
    /// # Errors
    /// Returns an error for an empty list, a file that does not map, a size
    /// that is not a multiple of 4, or a sidecar that disagrees with its file.
    pub fn from_files<P: AsRef<Path>>(
        paths: &[P],
        batch_size: usize,
        seq_len: usize,
        device: R::Device,
    ) -> Result<Self> {
        let store = RowStore::map_files(paths, seq_len + 1)?;
        Ok(Self::from_store(store, batch_size, seq_len, device))
    }

    /// Read only the global row ranges `spans`, in order.
    ///
    /// Each range is `(start, end)`, end exclusive. The ranges must ascend
    /// without overlap. Loader rows count across the ranges in order. The
    /// position resets to the start. A weighted mixture is rebuilt over the
    /// new ranges.
    ///
    /// # Errors
    /// Returns an error for an empty list, an empty range, a range past the
    /// last whole row, or ranges out of order, and the errors of
    /// [`with_mixture`](Self::with_mixture) under a mixture.
    pub fn with_row_ranges(mut self, spans: &[(usize, usize)]) -> Result<Self> {
        self.ranges = RowRanges::new(spans, self.total_rows())?;
        self.position = DataPosition::default();
        self.rebuild_mixture()
    }

    /// The global row ranges this loader reads, ascending.
    pub fn row_ranges(&self) -> &[(usize, usize)] {
        self.ranges.spans()
    }

    /// Whole rows of each source, in source order. One entry for in-memory data.
    pub fn source_rows(&self) -> Vec<usize> {
        self.store.source_rows()
    }

    /// Checks each file's sidecar `max_token_id` against `vocab_size`.
    ///
    /// A file without a sidecar is not checked.
    ///
    /// # Errors
    /// Returns an error naming the file, its `max_token_id` and `vocab_size`
    /// when the file holds an id at or above `vocab_size`.
    pub fn check_token_ids(&self, vocab_size: usize) -> Result<()> {
        self.store.check_vocab(vocab_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::positioning::tests::{epoch_rows, rows_of, SEQ_LEN, WIDTH};
    use crate::data::RowOrder;
    use boostr::{CpuDevice, CpuRuntime};
    use std::collections::HashSet;

    /// Sources of `rows[i]` rows plus a `tail`. Every token is its own offset
    /// in the concatenation of the sources' whole rows, so `rows_of` reads the
    /// global row index. Tail tokens are `u32::MAX`, so a read tail fails `rows_of`.
    fn sources(rows: &[usize], tail: usize) -> Vec<Vec<u32>> {
        let mut first = 0;
        rows.iter()
            .map(|&n| {
                let mut part: Vec<u32> = (first..first + (n * WIDTH) as u32).collect();
                part.extend(std::iter::repeat_n(u32::MAX, tail));
                first += (n * WIDTH) as u32;
                part
            })
            .collect()
    }

    fn concat_loader(rows: &[usize], tail: usize, batch: usize) -> LitDataLoader<CpuRuntime> {
        let store = RowStore::in_memory_parts(sources(rows, tail), WIDTH);
        LitDataLoader::from_store(store, batch, SEQ_LEN, CpuDevice::new())
    }

    #[test]
    fn concat_rows_follow_the_sources_in_order() {
        let mut l = concat_loader(&[3, 2, 4], 2, 3);
        assert_eq!(l.source_rows(), vec![3, 2, 4]);
        assert_eq!(l.total_rows(), 9);
        assert_eq!(l.dataset_size(), 9 * WIDTH + 3 * 2);
        assert_eq!(epoch_rows(&mut l, 3), (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn no_row_crosses_a_source_under_a_shuffle() {
        let mut l = concat_loader(&[5, 1, 6], 3, 4).with_row_order(RowOrder::Shuffled { seed: 2 });
        for epoch in 0..3 {
            // `rows_of` checks every row is WIDTH consecutive offsets: no tail, no seam.
            let mut rows = epoch_rows(&mut l, 3);
            rows.sort_unstable();
            assert_eq!(rows, (0..12).collect::<Vec<_>>(), "epoch {epoch}");
        }
    }

    #[test]
    fn one_source_reads_what_the_plain_loader_reads() {
        let tokens: Vec<u32> = (0..(10 * WIDTH + 3) as u32).collect();
        let order = RowOrder::Shuffled { seed: 9 };
        let mut plain =
            LitDataLoader::<CpuRuntime>::new(tokens.clone(), 2, SEQ_LEN, CpuDevice::new())
                .with_row_order(order);
        let store = RowStore::in_memory_parts(vec![tokens], WIDTH);
        let mut parts =
            LitDataLoader::<CpuRuntime>::from_store(store, 2, SEQ_LEN, CpuDevice::new())
                .with_row_order(order);
        assert_eq!(epoch_rows(&mut parts, 15), epoch_rows(&mut plain, 15));
    }

    #[test]
    fn row_ranges_read_only_their_rows_and_stay_disjoint() {
        // Sources of 6 and 4 rows. Train [0, 5) and [6, 9), eval [5, 6) and [9, 10).
        let train_spans = [(0, 5), (6, 9)];
        let eval_spans = [(5, 6), (9, 10)];
        let mut train = concat_loader(&[6, 4], 1, 2)
            .with_row_order(RowOrder::Shuffled { seed: 4 })
            .with_row_ranges(&train_spans)
            .expect("train ranges");
        let mut eval = concat_loader(&[6, 4], 1, 1)
            .with_row_ranges(&eval_spans)
            .expect("eval ranges");
        assert_eq!(train.row_ranges(), &train_spans);
        assert_eq!(train.row_layout().expect("layout").num_rows(), 8);
        let train_rows: HashSet<usize> = epoch_rows(&mut train, 8).into_iter().collect();
        let eval_rows: Vec<usize> = epoch_rows(&mut eval, 2);
        assert_eq!(eval_rows, vec![5, 9]);
        assert_eq!(train_rows.len(), 8);
        assert!(eval_rows.iter().all(|r| !train_rows.contains(r)));
    }

    #[test]
    fn batches_carry_rows_from_several_sources() {
        let mut l = concat_loader(&[1, 1], 0, 2);
        let rows = rows_of(&l.next_batch().expect("batch"));
        assert_eq!(rows, vec![0, 1]);
    }
}
