//! Which source supplies each slot of one epoch of a weighted mixture.
//!
//! Source `i` fills `c_i` slots of each epoch (see [`super::quota`]).
//! - Shuffled row order: a seeded permutation of the multiset holding `i`
//!   `c_i` times. The seed mixes the run seed and the epoch index.
//! - Stream order: smooth weighted round-robin over the quotas. It spreads
//!   each source evenly over the epoch and is the same every epoch.
//!
//! A schedule stores one `u16` source id per slot, and the count of each
//! source before every [`BLOCK`]-th slot. A lookup at any slot scans at most
//! `BLOCK` slots, so a seek lands mid-epoch exactly.

use super::super::rows::{epoch_seed, RowOrder};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Slots between two stored prefix counts.
pub(super) const BLOCK: usize = 4096;

/// Mixed into the run seed, so the schedule shuffle and the row shuffles differ.
const SCHEDULE_SALT: u64 = 0x6A09_E667_F3BC_C908;

/// The source of each slot of one epoch.
#[derive(Debug, Clone)]
pub(super) struct EpochSchedule {
    /// Source id of each slot.
    sources: Vec<u16>,
    /// `prefix[b * width + i]`: slots of source `i` in `[0, b * BLOCK)`.
    prefix: Vec<u32>,
    /// Sources in the mixture.
    width: usize,
}

/// A lookup position in one [`EpochSchedule`]: the next slot and the count
/// of each source before it.
#[derive(Debug, Clone, Default)]
pub(super) struct Cursor {
    slot: usize,
    counts: Vec<u32>,
}

impl EpochSchedule {
    /// The schedule of `epoch` under `order` for sources of `quotas` slots.
    ///
    /// The source count must fit in a `u16`, and each quota in a `u32`.
    pub(super) fn build(quotas: &[usize], order: RowOrder, epoch: usize) -> Self {
        let sources = match order {
            RowOrder::Stream => round_robin(quotas),
            RowOrder::Shuffled { seed } => shuffled(quotas, seed, epoch),
        };
        let width = quotas.len();
        let blocks = sources.len().div_ceil(BLOCK).max(1);
        let mut prefix = Vec::with_capacity(blocks * width);
        let mut counts = vec![0u32; width];
        for chunk in sources.chunks(BLOCK) {
            prefix.extend_from_slice(&counts);
            for &id in chunk {
                counts[id as usize] += 1;
            }
        }
        if prefix.is_empty() {
            prefix.extend_from_slice(&counts);
        }
        Self {
            sources,
            prefix,
            width,
        }
    }

    /// Slots in the epoch.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.sources.len()
    }

    /// The source of `slot`, and the slots of that source before `slot` in this epoch.
    ///
    /// `cursor` makes consecutive slots O(1). Any other slot rescans from the
    /// stored count before it. `None` when `slot` is past the epoch.
    pub(super) fn locate(&self, cursor: &mut Cursor, slot: usize) -> Option<(usize, usize)> {
        let source = *self.sources.get(slot)? as usize;
        if cursor.counts.len() != self.width || cursor.slot > slot || slot - cursor.slot >= BLOCK {
            let block = slot / BLOCK;
            cursor.slot = block * BLOCK;
            cursor.counts.clear();
            let stored = self
                .prefix
                .get(block * self.width..(block + 1) * self.width)?;
            cursor.counts.extend_from_slice(stored);
        }
        for &id in self.sources.get(cursor.slot..slot)? {
            *cursor.counts.get_mut(id as usize)? += 1;
        }
        let count = cursor.counts.get_mut(source)?;
        let before = *count as usize;
        *count += 1;
        cursor.slot = slot + 1;
        Some((source, before))
    }
}

/// Smooth weighted round-robin: each slot adds every quota to its source's
/// credit and picks the largest credit, lower index first on a tie. The pick
/// pays back the epoch length. Over one epoch source `i` is picked `c_i` times.
fn round_robin(quotas: &[usize]) -> Vec<u16> {
    let total: usize = quotas.iter().sum();
    let mut credit = vec![0i64; quotas.len()];
    let mut sources = Vec::with_capacity(total);
    for _ in 0..total {
        let mut pick = 0;
        for (i, &quota) in quotas.iter().enumerate() {
            credit[i] += quota as i64;
            if credit[i] > credit[pick] {
                pick = i;
            }
        }
        credit[pick] -= total as i64;
        sources.push(pick as u16);
    }
    sources
}

/// The multiset of source `i` repeated `c_i` times, in a seeded order for `epoch`.
fn shuffled(quotas: &[usize], seed: u64, epoch: usize) -> Vec<u16> {
    let mut sources: Vec<u16> = quotas
        .iter()
        .enumerate()
        .flat_map(|(i, &quota)| std::iter::repeat_n(i as u16, quota))
        .collect();
    sources.shuffle(&mut StdRng::seed_from_u64(epoch_seed(
        seed ^ SCHEDULE_SALT,
        epoch,
    )));
    sources
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slots(s: &EpochSchedule) -> Vec<usize> {
        let mut cursor = Cursor::default();
        (0..s.len())
            .map(|slot| s.locate(&mut cursor, slot).expect("slot").0)
            .collect()
    }

    fn counts(s: &EpochSchedule, width: usize) -> Vec<usize> {
        let mut c = vec![0; width];
        for i in slots(s) {
            c[i] += 1;
        }
        c
    }

    #[test]
    fn every_epoch_fills_each_quota_exactly() {
        let quotas = [5000, 3, 4100, 1];
        for order in [RowOrder::Stream, RowOrder::Shuffled { seed: 3 }] {
            for epoch in 0..3 {
                let s = EpochSchedule::build(&quotas, order, epoch);
                assert_eq!(s.len(), 9104);
                assert_eq!(counts(&s, 4), quotas.to_vec(), "{order:?} epoch {epoch}");
            }
        }
    }

    #[test]
    fn stream_order_interleaves_evenly_and_never_changes() {
        let s = EpochSchedule::build(&[2, 1], RowOrder::Stream, 0);
        assert_eq!(slots(&s), vec![0, 1, 0]);
        let s = EpochSchedule::build(&[3, 1, 0], RowOrder::Stream, 0);
        assert_eq!(slots(&s), vec![0, 0, 1, 0]);
        let a = EpochSchedule::build(&[7, 5, 2], RowOrder::Stream, 0);
        let b = EpochSchedule::build(&[7, 5, 2], RowOrder::Stream, 9);
        assert_eq!(slots(&a), slots(&b));
    }

    #[test]
    fn a_seed_fixes_the_shuffle_and_each_epoch_draws_another() {
        let quotas = [40, 30, 30];
        let order = RowOrder::Shuffled { seed: 11 };
        let e0 = slots(&EpochSchedule::build(&quotas, order, 0));
        assert_eq!(e0, slots(&EpochSchedule::build(&quotas, order, 0)));
        assert_ne!(e0, slots(&EpochSchedule::build(&quotas, order, 1)));
        let other = RowOrder::Shuffled { seed: 12 };
        assert_ne!(e0, slots(&EpochSchedule::build(&quotas, other, 0)));
    }

    #[test]
    fn a_lookup_at_any_slot_counts_the_slots_before_it() {
        let s = EpochSchedule::build(&[6000, 2500, 900], RowOrder::Shuffled { seed: 5 }, 2);
        let all = slots(&s);
        let mut cursor = Cursor::default();
        for slot in [9399, 0, 4095, 4096, 8191, 8192, 5000, 5001, 17] {
            let want = all[..slot].iter().filter(|&&i| i == all[slot]).count();
            assert_eq!(
                s.locate(&mut cursor, slot),
                Some((all[slot], want)),
                "slot {slot}"
            );
        }
        assert_eq!(s.locate(&mut cursor, 9400), None);
    }
}
