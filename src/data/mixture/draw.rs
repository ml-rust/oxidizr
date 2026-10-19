//! The row behind each draw from one source of a weighted mixture.
//!
//! Source `i` of `n_i` training rows numbers its draws over the whole run.
//! Draw `k` is pass `k / n_i`, index `k % n_i`. Under a shuffled row order the
//! row is entry `k % n_i` of a permutation seeded from the run seed, the
//! source index and the pass. Under stream order the row is the index itself.
//! A pass ends only after every row of the source, so an upsampled source
//! repeats each row once per pass.

use super::super::row_ranges::RowRanges;
use super::super::rows::{epoch_permutation, RowOrder};

/// Mixed with the source index into the run seed. Source 0 keeps the run seed,
/// so a one-source mixture of one pass per epoch reads the plain row order.
const SOURCE_SALT: u64 = 0xBB67_AE85_84CA_A73B;

/// One source's training rows and its cached pass permutations.
#[derive(Debug, Clone)]
pub(super) struct SourceDraws {
    /// Global rows of the source's training rows.
    rows: RowRanges,
    /// Seed of the cached permutations.
    seed: Option<u64>,
    /// `(pass, permutation)` of the last two passes asked for. A batch can
    /// span the end of one pass and the start of the next.
    cache: [Option<(usize, Vec<u32>)>; 2],
}

impl SourceDraws {
    /// The draws over `rows`. Their count must fit in a `u32`.
    pub(super) fn new(rows: RowRanges) -> Self {
        Self {
            rows,
            seed: None,
            cache: [None, None],
        }
    }

    /// Training rows of the source.
    pub(super) fn len(&self) -> usize {
        self.rows.len()
    }

    /// The global row of draw `draw` of source `index` under `order`.
    ///
    /// `None` for a source without rows.
    pub(super) fn global_row(
        &mut self,
        order: RowOrder,
        index: usize,
        draw: u128,
    ) -> Option<usize> {
        let n = self.rows.len() as u128;
        if n == 0 {
            return None;
        }
        let pass = usize::try_from(draw / n).ok()?;
        let at = (draw % n) as usize;
        let row = match order {
            RowOrder::Stream => at,
            RowOrder::Shuffled { seed } => self.permuted(source_seed(seed, index), pass, at)?,
        };
        Some(self.rows.global_row(row))
    }

    /// Entry `at` of the permutation of `pass` under `seed`.
    fn permuted(&mut self, seed: u64, pass: usize, at: usize) -> Option<usize> {
        if self.seed != Some(seed) {
            self.cache = [None, None];
            self.seed = Some(seed);
        }
        let hit = self
            .cache
            .iter()
            .position(|c| matches!(c, Some((p, _)) if *p == pass));
        let slot = match hit {
            Some(slot) => slot,
            None => {
                // Fill an empty entry, else replace the lower pass.
                let slot = match &self.cache {
                    [None, _] => 0,
                    [_, None] => 1,
                    [Some((a, _)), Some((b, _))] => usize::from(a > b),
                };
                self.cache[slot] = Some((pass, epoch_permutation(self.rows.len(), seed, pass)));
                slot
            }
        };
        let (_, perm) = self.cache[slot].as_ref()?;
        perm.get(at).map(|&row| row as usize)
    }
}

/// The permutation seed of source `index` under the run seed `seed`.
fn source_seed(seed: u64, index: usize) -> u64 {
    seed ^ (index as u64).wrapping_mul(SOURCE_SALT)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draws(start: usize, end: usize) -> SourceDraws {
        SourceDraws::new(RowRanges::new(&[(start, end)], 100).expect("range"))
    }

    #[test]
    fn stream_order_cycles_through_the_rows() {
        let mut d = draws(10, 13);
        let rows: Vec<usize> = (0..7)
            .map(|k| d.global_row(RowOrder::Stream, 1, k).expect("row"))
            .collect();
        assert_eq!(rows, vec![10, 11, 12, 10, 11, 12, 10]);
    }

    #[test]
    fn each_pass_is_a_seeded_permutation_and_passes_differ() {
        let order = RowOrder::Shuffled { seed: 4 };
        let mut d = draws(20, 60);
        let pass = |d: &mut SourceDraws, p: u128| -> Vec<usize> {
            (p * 40..(p + 1) * 40)
                .map(|k| d.global_row(order, 2, k).expect("row"))
                .collect()
        };
        let p0 = pass(&mut d, 0);
        let mut sorted = p0.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (20..60).collect::<Vec<_>>());
        let p1 = pass(&mut d, 1);
        assert_ne!(p0, p1, "each pass draws a new order");
        assert_eq!(
            pass(&mut draws(20, 60), 1),
            p1,
            "a pass is recomputed exactly"
        );
        assert_ne!(p0, {
            let mut other = draws(20, 60);
            (0..40)
                .map(|k| other.global_row(order, 3, k).expect("row"))
                .collect::<Vec<_>>()
        });
    }

    #[test]
    fn draws_across_a_pass_boundary_match_fresh_lookups() {
        let order = RowOrder::Shuffled { seed: 9 };
        let mut d = draws(0, 5);
        // Alternate passes 0, 1 and 2, as rank strides and seeks can.
        for k in [4u128, 5, 3, 6, 11, 2, 10, 14, 0] {
            let fresh = draws(0, 5).global_row(order, 0, k);
            assert_eq!(d.global_row(order, 0, k), fresh, "draw {k}");
        }
    }

    #[test]
    fn source_zero_reads_the_plain_epoch_permutation() {
        let mut d = draws(0, 30);
        for pass in 0..3 {
            let perm = epoch_permutation(30, 7, pass);
            for (at, &row) in perm.iter().enumerate() {
                let k = (pass * 30 + at) as u128;
                let got = d.global_row(RowOrder::Shuffled { seed: 7 }, 0, k);
                assert_eq!(got, Some(row as usize));
            }
        }
    }

    #[test]
    fn a_source_without_rows_has_no_draw() {
        let mut d = SourceDraws::new(RowRanges::whole(0));
        assert_eq!(d.len(), 0);
        assert_eq!(d.global_row(RowOrder::Stream, 0, 0), None);
    }
}
