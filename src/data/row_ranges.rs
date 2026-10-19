//! The global rows one loader reads: ascending, disjoint row ranges of its store.
//!
//! Loader row `r` is the `r`-th row across the ranges in order. With one
//! range `[start, end)`, loader row `r` is global row `start + r`.

use crate::prelude::{Error, Result};

/// Ascending, disjoint, non-empty global row ranges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RowRanges {
    /// `(start, end)` global rows, end exclusive.
    spans: Vec<(usize, usize)>,
    /// Loader rows before each span.
    before: Vec<usize>,
    /// Loader rows over every span.
    len: usize,
}

impl RowRanges {
    /// Every row of a store of `total` rows.
    pub(super) fn whole(total: usize) -> Self {
        if total == 0 {
            return Self {
                spans: Vec::new(),
                before: Vec::new(),
                len: 0,
            };
        }
        Self {
            spans: vec![(0, total)],
            before: vec![0],
            len: total,
        }
    }

    /// The ranges `spans` of a store of `total` rows.
    ///
    /// # Errors
    /// Returns an error for an empty list, an empty range, a range past
    /// `total`, or ranges out of order or overlapping.
    pub(super) fn new(spans: &[(usize, usize)], total: usize) -> Result<Self> {
        if spans.is_empty() {
            return Err(Error::DataError {
                reason: "an empty list of row ranges reads no rows. Pass at least one range."
                    .to_string(),
            });
        }
        let mut before = Vec::with_capacity(spans.len());
        let (mut len, mut prev_end) = (0, 0);
        for (i, &(start, end)) in spans.iter().enumerate() {
            if start >= end || end > total {
                return Err(Error::DataError {
                    reason: format!(
                        "row range [{start}, {end}) is not a non-empty range inside the {total} \
                         whole rows of this data. Pick a range inside [0, {total})."
                    ),
                });
            }
            if i > 0 && start < prev_end {
                return Err(Error::DataError {
                    reason: format!(
                        "row range [{start}, {end}) starts before row {prev_end}, the end of \
                         the range before it. Pass ranges in ascending order without overlap."
                    ),
                });
            }
            before.push(len);
            len += end - start;
            prev_end = end;
        }
        Ok(Self {
            spans: spans.to_vec(),
            before,
            len,
        })
    }

    /// Loader rows over every range.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// The ranges, ascending.
    pub(super) fn spans(&self) -> &[(usize, usize)] {
        &self.spans
    }

    /// From the start of the first range to the end of the last. `(0, 0)` when empty.
    pub(super) fn hull(&self) -> (usize, usize) {
        match (self.spans.first(), self.spans.last()) {
            (Some(&(start, _)), Some(&(_, end))) => (start, end),
            _ => (0, 0),
        }
    }

    /// The global row of loader row `row`.
    ///
    /// A row at or past [`len`](Self::len) maps past the last range, so the
    /// store refuses it.
    pub(super) fn global_row(&self, row: usize) -> usize {
        let index = self.before.partition_point(|&b| b <= row).saturating_sub(1);
        match (self.spans.get(index), self.before.get(index)) {
            (Some(&(start, _)), Some(&before)) => start + (row - before),
            _ => row,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_range_offsets_by_its_start() {
        let r = RowRanges::new(&[(3, 9)], 12).expect("range");
        assert_eq!(r.len(), 6);
        assert_eq!(r.hull(), (3, 9));
        let rows: Vec<usize> = (0..6).map(|i| r.global_row(i)).collect();
        assert_eq!(rows, vec![3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn several_ranges_concatenate_in_order() {
        let r = RowRanges::new(&[(0, 2), (5, 6), (8, 10)], 10).expect("ranges");
        assert_eq!(r.len(), 5);
        let rows: Vec<usize> = (0..5).map(|i| r.global_row(i)).collect();
        assert_eq!(rows, vec![0, 1, 5, 8, 9]);
        assert_eq!(r.hull(), (0, 10));
        assert!(
            r.global_row(5) >= 10,
            "past the end maps past the last range"
        );
    }

    #[test]
    fn whole_reads_every_row() {
        let r = RowRanges::whole(4);
        assert_eq!(r.spans(), &[(0, 4)]);
        assert_eq!(r.global_row(3), 3);
        assert_eq!(RowRanges::whole(0).len(), 0);
        assert_eq!(RowRanges::whole(0).hull(), (0, 0));
    }

    #[test]
    fn bad_ranges_are_refused() {
        assert!(RowRanges::new(&[], 10).is_err());
        let err = RowRanges::new(&[(0, 11)], 10).expect_err("past the end");
        assert!(err.to_string().contains("10 whole rows"), "{err}");
        assert!(RowRanges::new(&[(4, 4)], 10).is_err());
        let err = RowRanges::new(&[(0, 5), (4, 8)], 10).expect_err("overlap");
        assert!(err.to_string().contains("ascending order"), "{err}");
        assert!(RowRanges::new(&[(6, 8), (0, 2)], 10).is_err());
    }
}
