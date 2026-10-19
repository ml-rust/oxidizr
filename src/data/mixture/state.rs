//! A weighted mixture of data sources as a virtual row order.
//!
//! An epoch is `rows_per_epoch` slots. Source `i` fills `c_i` of them (see
//! [`super::quota`]), in the order of [`super::schedule`]. The `p`-th slot of
//! source `i` in epoch `e` is its draw `e * c_i + p`, and [`super::draw`]
//! maps the draw to a row. The row at a slot is a pure function of the
//! epoch, the slot, the row order and the sources, so a seek is exact.

use super::draw::SourceDraws;
use super::quota::{quotas, weight_error};
use super::record::{MixtureRecord, MixtureSource};
use super::schedule::{Cursor, EpochSchedule};
use crate::data::row_ranges::RowRanges;
use crate::data::rows::RowOrder;
use crate::prelude::{Error, Result};

/// Most rows one epoch can hold. The schedule keeps a `u16` per row in host memory.
const MAX_ROWS_PER_EPOCH: usize = 1 << 28;

/// The sources, quotas and cached epoch schedule of a weighted mixture.
#[derive(Debug, Clone)]
pub(in crate::data) struct Mixture {
    weights: Vec<f64>,
    /// `trainer.data.rows_per_epoch` as set. `None`: every training row of every source.
    requested_rows: Option<usize>,
    /// Slots per epoch.
    rows_per_epoch: usize,
    /// Slots of each source per epoch.
    quotas: Vec<usize>,
    sources: Vec<SourceDraws>,
    /// `(order, key epoch, schedule)` of the last epoch asked for. Stream
    /// order keys every epoch as 0: its schedule never changes.
    schedule: Option<(RowOrder, usize, EpochSchedule)>,
    cursor: Cursor,
}

fn data_error(reason: String) -> Error {
    Error::DataError { reason }
}

impl Mixture {
    /// A mixture of the sources whose training rows are `source_rows`, under `weights`.
    ///
    /// `rows_per_epoch` defaults to the training rows of every source.
    ///
    /// # Errors
    /// Returns an error naming the key for a weight count that differs from
    /// the source count, a weight that is not finite and above 0, a weighted
    /// source without training rows, or an epoch of 0 rows or more than a
    /// row order can index.
    pub(in crate::data) fn new(
        weights: &[f64],
        rows_per_epoch: Option<usize>,
        source_rows: Vec<RowRanges>,
    ) -> Result<Self> {
        if weights.len() != source_rows.len() {
            return Err(data_error(format!(
                "trainer.data.sources sets {} weight(s), and the loader reads {} source(s). Set \
                 one weight per source in trainer.data.sources, and remove the -d flags.",
                weights.len(),
                source_rows.len()
            )));
        }
        if weights.len() > usize::from(u16::MAX) + 1 {
            return Err(data_error(format!(
                "trainer.data.sources lists {} weighted sources, more than the {} a mixture can \
                 index. Merge some sources into one file.",
                weights.len(),
                usize::from(u16::MAX) + 1
            )));
        }
        for (i, (&weight, rows)) in weights.iter().zip(&source_rows).enumerate() {
            if let Some(reason) = weight_error(i, weight) {
                return Err(data_error(reason));
            }
            if rows.len() == 0 {
                return Err(data_error(format!(
                    "trainer.data.sources[{i}] holds no training rows, and its weight {weight} \
                     asks for rows. Remove the source, or use a file of at least one row of \
                     model.max_seq_len + 1 tokens outside the holdout."
                )));
            }
            if u32::try_from(rows.len()).is_err() {
                return Err(data_error(format!(
                    "trainer.data.sources[{i}] holds {} training rows, more than the {} a row \
                     order can index. Split the file.",
                    rows.len(),
                    u32::MAX
                )));
            }
        }
        let total: usize = source_rows.iter().map(RowRanges::len).sum();
        let rows = rows_per_epoch.unwrap_or(total);
        if rows == 0 || rows > MAX_ROWS_PER_EPOCH {
            return Err(data_error(format!(
                "trainer.data.rows_per_epoch is {rows}: an epoch needs at least 1 row and at \
                 most {MAX_ROWS_PER_EPOCH} rows, because the schedule keeps 2 bytes per row. \
                 Set trainer.data.rows_per_epoch inside that range."
            )));
        }
        Ok(Self {
            weights: weights.to_vec(),
            requested_rows: rows_per_epoch,
            rows_per_epoch: rows,
            quotas: quotas(weights, rows),
            sources: source_rows.into_iter().map(SourceDraws::new).collect(),
            schedule: None,
            cursor: Cursor::default(),
        })
    }

    /// The weights and the `rows_per_epoch` this mixture was built from.
    pub(in crate::data) fn spec(&self) -> (Vec<f64>, Option<usize>) {
        (self.weights.clone(), self.requested_rows)
    }

    /// Slots per epoch.
    pub(in crate::data) fn rows_per_epoch(&self) -> usize {
        self.rows_per_epoch
    }

    /// What a checkpoint meta records of this mixture.
    pub(in crate::data) fn record(&self) -> MixtureRecord {
        MixtureRecord {
            weights: self.weights.clone(),
            rows_per_epoch: self.rows_per_epoch,
        }
    }

    /// Weight, quota and training rows of each source, in source order.
    pub(in crate::data) fn sources(&self) -> Vec<MixtureSource> {
        self.weights
            .iter()
            .zip(&self.quotas)
            .zip(&self.sources)
            .map(|((&weight, &rows_per_epoch), draws)| MixtureSource {
                weight,
                rows_per_epoch,
                train_rows: draws.len(),
            })
            .collect()
    }

    /// The global row at `slot` of `epoch` under `order`.
    ///
    /// # Errors
    /// Returns an error when `slot` is not below [`rows_per_epoch`](Self::rows_per_epoch).
    pub(in crate::data) fn global_row(
        &mut self,
        order: RowOrder,
        epoch: usize,
        slot: usize,
    ) -> Result<usize> {
        let key = match order {
            RowOrder::Stream => 0,
            RowOrder::Shuffled { .. } => epoch,
        };
        let fresh = !matches!(&self.schedule, Some((o, e, _)) if *o == order && *e == key);
        if fresh {
            let schedule = EpochSchedule::build(&self.quotas, order, key);
            self.schedule = Some((order, key, schedule));
            self.cursor = Cursor::default();
        }
        let located = match &self.schedule {
            Some((_, _, schedule)) => schedule.locate(&mut self.cursor, slot),
            None => None,
        };
        let row = located.and_then(|(source, before)| {
            let quota = *self.quotas.get(source)? as u128;
            let draw = epoch as u128 * quota + before as u128;
            self.sources
                .get_mut(source)?
                .global_row(order, source, draw)
        });
        row.ok_or_else(|| {
            data_error(format!(
                "mixture slot {slot} of epoch {epoch} is past the {} rows of an epoch. This is a \
                 loader bug: report it with the data and config.",
                self.rows_per_epoch
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranges(spans: &[(usize, usize)]) -> Vec<RowRanges> {
        spans
            .iter()
            .map(|&span| RowRanges::new(&[span], 1000).expect("range"))
            .collect()
    }

    fn epoch(m: &mut Mixture, order: RowOrder, e: usize) -> Vec<usize> {
        (0..m.rows_per_epoch())
            .map(|slot| m.global_row(order, e, slot).expect("row"))
            .collect()
    }

    #[test]
    fn each_epoch_reads_each_quota_from_its_source() {
        // Source 0 is rows [0, 10), source 1 is rows [100, 104).
        let mut m =
            Mixture::new(&[1.0, 3.0], Some(10), ranges(&[(0, 10), (100, 104)])).expect("mixture");
        let q: Vec<usize> = m.sources().iter().map(|s| s.rows_per_epoch).collect();
        assert_eq!(q, vec![3, 7]);
        for order in [RowOrder::Stream, RowOrder::Shuffled { seed: 1 }] {
            for e in 0..4 {
                let rows = epoch(&mut m, order, e);
                assert_eq!(rows.iter().filter(|&&r| r < 10).count(), 3, "{order:?} {e}");
                assert_eq!(
                    rows.iter().filter(|&&r| r >= 100).count(),
                    7,
                    "{order:?} {e}"
                );
            }
        }
    }

    #[test]
    fn rows_per_epoch_above_the_schedule_cap_is_an_error() {
        let err = Mixture::new(
            &[1.0, 1.0],
            Some(MAX_ROWS_PER_EPOCH + 1),
            ranges(&[(0, 10), (100, 104)]),
        )
        .expect_err("over the cap");
        let text = err.to_string();
        assert!(text.contains("trainer.data.rows_per_epoch"), "{text}");
        assert!(text.contains(&MAX_ROWS_PER_EPOCH.to_string()), "{text}");
    }

    #[test]
    fn an_upsampled_source_repeats_every_row_once_per_pass() {
        let order = RowOrder::Shuffled { seed: 2 };
        let mut m =
            Mixture::new(&[1.0, 1.0], Some(16), ranges(&[(0, 3), (50, 58)])).expect("mixture");
        // Source 0 draws 8 rows per epoch over 3 rows: 24 draws over 3 epochs is 8 passes.
        let small: Vec<usize> = (0..3)
            .flat_map(|e| epoch(&mut m, order, e))
            .filter(|&r| r < 3)
            .collect();
        assert_eq!(small.len(), 24);
        for pass in small.chunks(3) {
            let mut p = pass.to_vec();
            p.sort_unstable();
            assert_eq!(p, vec![0, 1, 2]);
        }
    }

    #[test]
    fn a_lookup_out_of_order_equals_the_sequential_read() {
        let order = RowOrder::Shuffled { seed: 6 };
        let spans = [(0, 700), (700, 703), (800, 990)];
        let make = || Mixture::new(&[2.0, 1.0, 1.0], Some(9000), ranges(&spans));
        let mut seq = make().expect("mixture");
        let all: Vec<Vec<usize>> = (0..2).map(|e| epoch(&mut seq, order, e)).collect();
        let mut jumpy = make().expect("mixture");
        for (e, slot) in [
            (1, 8999),
            (0, 4096),
            (1, 0),
            (0, 4095),
            (0, 12),
            (1, 5000),
            (0, 8191),
        ] {
            assert_eq!(jumpy.global_row(order, e, slot).expect("row"), all[e][slot]);
        }
        assert!(jumpy.global_row(order, 0, 9000).is_err());
    }

    #[test]
    fn one_source_of_one_pass_per_epoch_reads_the_plain_order() {
        let order = RowOrder::Shuffled { seed: 5 };
        let mut m = Mixture::new(&[0.3], None, ranges(&[(4, 24)])).expect("mixture");
        for e in 0..3 {
            let plain: Vec<usize> = crate::data::rows::epoch_permutation(20, 5, e)
                .iter()
                .map(|&r| 4 + r as usize)
                .collect();
            assert_eq!(epoch(&mut m, order, e), plain);
        }
        assert_eq!(
            epoch(&mut m, RowOrder::Stream, 1),
            (4..24).collect::<Vec<_>>()
        );
    }

    #[test]
    fn bad_mixtures_are_refused_naming_the_key() {
        let err = |w: &[f64], rows: Option<usize>, r: Vec<RowRanges>| {
            Mixture::new(w, rows, r).expect_err("refused").to_string()
        };
        let text = err(&[1.0], None, ranges(&[(0, 4), (4, 8)]));
        assert!(
            text.contains("1 weight(s)") && text.contains("2 source(s)"),
            "{text}"
        );
        let text = err(&[1.0, -2.0], None, ranges(&[(0, 4), (4, 8)]));
        assert!(
            text.contains("trainer.data.sources[1].weight is -2"),
            "{text}"
        );
        let text = err(&[1.0, 1.0], Some(0), ranges(&[(0, 4), (4, 8)]));
        assert!(text.contains("trainer.data.rows_per_epoch is 0"), "{text}");
        let text = err(
            &[1.0, 1.0],
            None,
            vec![RowRanges::whole(4), RowRanges::whole(0)],
        );
        assert!(
            text.contains("trainer.data.sources[1] holds no training rows"),
            "{text}"
        );
    }
}
