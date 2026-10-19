//! The per-epoch row quota of each source of a weighted mixture.
//!
//! Source `i` of weight `w_i` supplies `c_i` rows of each epoch of `L` rows.
//! `c_i` is the largest-remainder rounding of `w_i / sum(w) * L`: every
//! source gets the floor of its share, then the rows left go one each to the
//! largest remainders. A tie goes to the lower source index. `sum(c_i) == L`.

/// The error for weight `weight` of source `index`, or `None` for a valid weight.
///
/// A valid weight is finite and above 0.
pub(in crate::data) fn weight_error(index: usize, weight: f64) -> Option<String> {
    (!(weight.is_finite() && weight > 0.0)).then(|| {
        format!(
            "trainer.data.sources[{index}].weight is {weight}: a weight must be a finite number \
             above 0. Set it to the source's share, for example 1.0."
        )
    })
}

/// The quota of each source of `weights` over an epoch of `rows` rows.
///
/// Each weight must pass [`weight_error`].
pub(super) fn quotas(weights: &[f64], rows: usize) -> Vec<usize> {
    let total: f64 = weights.iter().sum();
    let shares: Vec<f64> = weights.iter().map(|w| w / total * rows as f64).collect();
    let mut counts: Vec<usize> = shares
        .iter()
        .map(|s| (s.floor() as usize).min(rows))
        .collect();
    // Remainder order: largest first, lower index first on a tie.
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by(|&a, &b| {
        let (ra, rb) = (shares[a] - shares[a].floor(), shares[b] - shares[b].floor());
        rb.total_cmp(&ra).then(a.cmp(&b))
    });
    let assigned: usize = counts.iter().sum();
    if assigned <= rows {
        // At most one extra row per source: the remainders sum below the source count.
        for &i in order.iter().cycle().take(rows - assigned) {
            counts[i] += 1;
        }
    } else {
        // Rounding error in the shares: take rows back from the smallest remainders.
        let mut excess = assigned - rows;
        for &i in order.iter().rev().cycle() {
            if excess == 0 {
                break;
            }
            if counts[i] > 0 {
                counts[i] -= 1;
                excess -= 1;
            }
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotas_sum_to_the_epoch_and_follow_the_weights() {
        assert_eq!(quotas(&[3.0, 1.0], 100), vec![75, 25]);
        assert_eq!(quotas(&[1.0, 1.0, 2.0], 8), vec![2, 2, 4]);
        for rows in [1, 7, 10, 99, 1000, 12_345] {
            let q = quotas(&[0.7, 0.2, 0.1, 1e-3], rows);
            assert_eq!(q.iter().sum::<usize>(), rows, "rows={rows}");
        }
    }

    #[test]
    fn remainders_go_to_the_largest_then_the_lower_index() {
        // Shares 3.33.. each: one extra row, to source 0.
        assert_eq!(quotas(&[1.0, 1.0, 1.0], 10), vec![4, 3, 3]);
        // Shares 3.33.. each: two extra rows, to sources 0 and 1.
        assert_eq!(quotas(&[1.0, 1.0, 1.0], 11), vec![4, 4, 3]);
        // Shares 1.2, 2.4, 3.6, 4.8: remainders .2 .4 .6 .8, two extra rows.
        assert_eq!(quotas(&[1.0, 2.0, 3.0, 4.0], 12), vec![1, 2, 4, 5]);
        // Shares 0.5 each over 1 row: the tie goes to source 0.
        assert_eq!(quotas(&[2.0, 2.0], 1), vec![1, 0]);
    }

    #[test]
    fn only_finite_positive_weights_pass() {
        assert_eq!(weight_error(0, 0.5), None);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = weight_error(2, bad).expect("a bad weight");
            assert!(err.contains("trainer.data.sources[2].weight"), "{err}");
        }
    }

    #[test]
    fn one_source_takes_every_row() {
        assert_eq!(quotas(&[0.25], 37), vec![37]);
    }
}
