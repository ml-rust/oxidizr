//! The weighted mixture of a [`LitDataLoader`]: building it over the loader's
//! training rows, and what the run reports and records of it.

use super::record::{MixtureRecord, MixtureSource};
use super::state::Mixture;
use crate::data::construction::LitDataLoader;
use crate::data::row_ranges::RowRanges;
use crate::data::rows::DataPosition;
use crate::prelude::{DType, Result};
use boostr::runtime::Runtime;

impl<R> LitDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    /// Read a weighted mixture of the sources instead of their concatenation.
    ///
    /// `weights[i]` weighs source `i`. Each epoch is `rows_per_epoch` slots,
    /// by default every training row of every source. Source `i` fills its
    /// largest-remainder share of the slots. A source draws its training rows
    /// in passes: each pass reads every row once, in a seeded order under a
    /// shuffled row order. The mixture covers only the loader's row ranges,
    /// so a holdout stays out. A later [`with_row_ranges`](Self::with_row_ranges)
    /// rebuilds the mixture over the new ranges. The position resets to the start.
    ///
    /// # Errors
    /// Returns an error naming the key for a weight count that differs from
    /// the source count, a weight that is not finite and above 0, a weighted
    /// source without training rows, or an epoch of 0 rows.
    pub fn with_mixture(mut self, weights: &[f64], rows_per_epoch: Option<usize>) -> Result<Self> {
        let source_rows = self.source_train_rows()?;
        self.mixture = Some(Mixture::new(weights, rows_per_epoch, source_rows)?);
        self.position = DataPosition::default();
        Ok(self)
    }

    /// The mixture a checkpoint meta records. `None` without weights.
    pub fn mixture_record(&self) -> Option<MixtureRecord> {
        self.mixture.as_ref().map(Mixture::record)
    }

    /// Weight, rows per epoch and training rows of each source. `None` without weights.
    pub fn mixture_sources(&self) -> Option<Vec<MixtureSource>> {
        self.mixture.as_ref().map(Mixture::sources)
    }

    /// Rebuilds the mixture, when there is one, over the current row ranges.
    pub(in crate::data) fn rebuild_mixture(mut self) -> Result<Self> {
        match self.mixture.take() {
            Some(mixture) => {
                let (weights, rows) = mixture.spec();
                self.with_mixture(&weights, rows)
            }
            None => Ok(self),
        }
    }

    /// Each source's global rows inside this loader's row ranges, in source order.
    fn source_train_rows(&self) -> Result<Vec<RowRanges>> {
        let total = self.store.total_rows();
        let mut next = 0;
        self.store
            .source_rows()
            .into_iter()
            .map(|rows| {
                let (first, end) = (next, next + rows);
                next = end;
                let inside: Vec<(usize, usize)> = self
                    .ranges
                    .spans()
                    .iter()
                    .map(|&(start, stop)| (start.max(first), stop.min(end)))
                    .filter(|&(start, stop)| start < stop)
                    .collect();
                if inside.is_empty() {
                    Ok(RowRanges::whole(0))
                } else {
                    RowRanges::new(&inside, total)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::positioning::tests::{epoch_rows, SEQ_LEN, WIDTH};
    use crate::data::source::RowStore;
    use crate::data::RowOrder;
    use boostr::{CpuDevice, CpuRuntime};

    /// Sources of `rows[i]` rows whose tokens are their offset in the
    /// concatenation of every source's rows, so `rows_of` reads global rows.
    fn mixed(rows: &[usize], batch: usize) -> LitDataLoader<CpuRuntime> {
        let mut first = 0;
        let parts = rows
            .iter()
            .map(|&n| {
                let part: Vec<u32> = (first..first + (n * WIDTH) as u32).collect();
                first += (n * WIDTH) as u32;
                part
            })
            .collect();
        let store = RowStore::in_memory_parts(parts, WIDTH);
        LitDataLoader::from_store(store, batch, SEQ_LEN, CpuDevice::new())
    }

    #[test]
    fn the_layout_counts_rows_per_epoch() {
        let l = mixed(&[10, 4], 2)
            .with_mixture(&[1.0, 1.0], Some(20))
            .expect("mixture");
        let layout = l.row_layout().expect("layout");
        assert_eq!(layout.num_rows(), 20);
        assert_eq!(layout.batches_per_epoch(), 10);
        let sources = l.mixture_sources().expect("sources");
        assert_eq!(sources[1].rows_per_epoch, 10);
        assert_eq!(sources[1].repeats(), 2.5);
        assert_eq!(l.mixture_record().expect("record").rows_per_epoch, 20);
        assert!(mixed(&[3], 1).mixture_record().is_none());
    }

    #[test]
    fn a_mixture_reads_only_the_row_ranges() {
        // Sources [0, 8) and [8, 14). Train [0, 6) and [8, 12); the rest is held out.
        let mut l = mixed(&[8, 6], 2)
            .with_mixture(&[1.0, 2.0], None)
            .expect("mixture")
            .with_row_ranges(&[(0, 6), (8, 12)])
            .expect("ranges")
            .with_row_order(RowOrder::Shuffled { seed: 3 });
        assert_eq!(l.row_layout().expect("layout").num_rows(), 10);
        let rows = epoch_rows(&mut l, 15);
        assert!(
            rows.iter().all(|&r| r < 6 || (8..12).contains(&r)),
            "{rows:?}"
        );
        // Quotas 3 and 7 of 10 rows per epoch, over 3 epochs.
        assert_eq!(rows.iter().filter(|&&r| r < 6).count(), 9);
    }

    #[test]
    fn one_weighted_source_reads_what_the_plain_loader_reads() {
        let order = RowOrder::Shuffled { seed: 9 };
        let mut plain = mixed(&[12], 3).with_row_order(order);
        let mut weighted = mixed(&[12], 3)
            .with_mixture(&[5.0], None)
            .expect("mixture")
            .with_row_order(order);
        assert_eq!(epoch_rows(&mut weighted, 16), epoch_rows(&mut plain, 16));
    }

    #[test]
    fn a_seek_mid_epoch_continues_the_uninterrupted_rows() {
        let make = || {
            mixed(&[9, 2], 3)
                .with_mixture(&[1.0, 1.0], Some(12))
                .expect("mixture")
                .with_row_order(RowOrder::Shuffled { seed: 4 })
        };
        let mut full = make();
        let layout = full.row_layout().expect("layout");
        let _ = epoch_rows(&mut full, 5);
        let mut resumed = make();
        resumed
            .seek(layout.advance(DataPosition::default(), 5))
            .expect("seek");
        // 9 batches cross source 1's pass boundaries and two epoch boundaries.
        assert_eq!(epoch_rows(&mut resumed, 9), epoch_rows(&mut full, 9));
    }
}
