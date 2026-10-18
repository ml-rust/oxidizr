//! Cross-entropy weighting across the micro-batches of one optimizer step.
//!
//! A masked run scores a different number of tokens in each micro-batch. A
//! per-micro-batch mean then gives a token in a sparse micro-batch more weight
//! than a token in a dense one. [`StepWeighting::ScoredTokens`] gives every
//! scored token of the step the weight `1 / total`.
//!
//! An unmasked run keeps the per-micro-batch mean. Every micro-batch scores
//! `batch_size * seq_len` tokens, so both forms are equal in exact arithmetic.
//! Their float rounding differs, and the mean form keeps the bits of earlier
//! runs and their resumes.

use crate::data::Batch;
use crate::prelude::{cross_entropy, DType, Error, Result, Tensor};
use boostr::autograd::{var_mul_scalar, Var};
use boostr::nn::loss::cross_entropy_loss_masked_sum;
use boostr::runtime::{Runtime, RuntimeClient};
use boostr::{
    ActivationOps, BinaryOps, IndexingOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};

/// How one step weights the cross-entropy of each micro-batch.
///
/// The trainer scales every micro-batch loss by `1 / micro_batches`.
/// [`StepWeighting::ce_term`] returns the cross-entropy before that scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepWeighting {
    /// No loss mask: the mean over each micro-batch.
    MicroBatchMean {
        /// Micro-batches in the step.
        micro_batches: usize,
    },
    /// Loss mask: the masked sum of each micro-batch, times
    /// `micro_batches / total_scored`.
    ScoredTokens {
        /// Micro-batches in the step.
        micro_batches: usize,
        /// Scored positions over every micro-batch of the step.
        total_scored: usize,
    },
}

impl StepWeighting {
    /// Picks the weighting for the micro-batches of one step.
    ///
    /// Every batch must carry a loss mask, or none may. A loader either masks
    /// every batch or no batch, so a mix is an error.
    pub fn for_batches<R: Runtime<DType = DType>>(batches: &[Batch<R>]) -> Result<Self> {
        let micro_batches = batches.len();
        let masked = batches.iter().filter(|b| b.loss_mask.is_some()).count();
        if masked == 0 {
            return Ok(Self::MicroBatchMean { micro_batches });
        }
        if masked != micro_batches {
            return Err(Error::TrainingError {
                reason: format!(
                    "{masked} of {micro_batches} micro-batches in one step carry a loss mask; \
                     the loader must mask every batch or none"
                ),
            });
        }
        let total_scored = batches.iter().map(|b| b.scored).sum();
        Ok(Self::ScoredTokens {
            micro_batches,
            total_scored,
        })
    }

    /// The weighting of this rank's step over `world` data-parallel ranks.
    ///
    /// `micro_batches` becomes the count over every rank. A masked step takes
    /// `global_scored`, its scored count summed over every rank. An unmasked
    /// step takes `None`. Each micro-batch loss is then scaled by
    /// `1 / (micro_batches * world)`, so the sum over ranks is the step mean.
    ///
    /// # Errors
    /// Returns an error when `world` is 0, when `global_scored` does not
    /// match the step's form, or when the micro-batch count overflows.
    pub fn over_ranks(self, world: usize, global_scored: Option<usize>) -> Result<Self> {
        if world == 0 {
            return Err(Error::TrainingError {
                reason: "a data-parallel step needs at least one rank, got world size 0"
                    .to_string(),
            });
        }
        let micro_batches =
            self.micro_batches()
                .checked_mul(world)
                .ok_or_else(|| Error::TrainingError {
                    reason: format!(
                        "{} micro-batches times {world} ranks overflows a count",
                        self.micro_batches()
                    ),
                })?;
        match (self, global_scored) {
            (Self::MicroBatchMean { .. }, None) => Ok(Self::MicroBatchMean { micro_batches }),
            (Self::ScoredTokens { .. }, Some(total_scored)) => Ok(Self::ScoredTokens {
                micro_batches,
                total_scored,
            }),
            (Self::MicroBatchMean { .. }, Some(_)) => Err(Error::TrainingError {
                reason: "an unmasked step got a global scored count. Pass None".to_string(),
            }),
            (Self::ScoredTokens { .. }, None) => Err(Error::TrainingError {
                reason: "a masked step needs the scored count summed over every rank".to_string(),
            }),
        }
    }

    /// Micro-batches in the step. The trainer scales each loss by its inverse.
    pub fn micro_batches(&self) -> usize {
        match self {
            Self::MicroBatchMean { micro_batches } | Self::ScoredTokens { micro_batches, .. } => {
                *micro_batches
            }
        }
    }

    /// Scored tokens of the step. `None` for an unmasked step, which scores every token.
    ///
    /// After [`over_ranks`](Self::over_ranks) the count covers every rank.
    pub fn scored_tokens(&self) -> Option<usize> {
        match self {
            Self::MicroBatchMean { .. } => None,
            Self::ScoredTokens { total_scored, .. } => Some(*total_scored),
        }
    }

    /// Cross-entropy term of one micro-batch, before the `1 / micro_batches`
    /// scale.
    ///
    /// - `logits`: `[N, V]`, `targets`: `[N]`, `mask`: `[N]` or `None`
    /// - `scored`: the batch's [`Batch::scored`]
    ///
    /// Returns `None` when the micro-batch scores no position. It then adds
    /// no cross-entropy term and no NaN.
    pub fn ce_term<R, C>(
        &self,
        client: &C,
        logits: &Var<R>,
        targets: &Tensor<R>,
        mask: Option<&Tensor<R>>,
        scored: usize,
    ) -> Result<Option<Var<R>>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + IndexingOps<R>
            + TypeConversionOps<R>,
        R::Client: ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + IndexingOps<R>
            + TypeConversionOps<R>,
    {
        match (self, mask) {
            (Self::MicroBatchMean { .. }, None) => {
                Ok(Some(cross_entropy(client, logits, targets)?))
            }
            (
                Self::ScoredTokens {
                    micro_batches,
                    total_scored,
                },
                Some(mask),
            ) => {
                if scored == 0 || *total_scored == 0 {
                    return Ok(None);
                }
                let sum = cross_entropy_loss_masked_sum(client, logits, targets, mask)?;
                let scale = *micro_batches as f64 / *total_scored as f64;
                Ok(Some(var_mul_scalar(&sum, scale, client)?))
            }
            (Self::MicroBatchMean { .. }, Some(_)) => Err(Error::TrainingError {
                reason: "a masked micro-batch reached an unmasked step; the loader must mask \
                         every batch or none"
                    .to_string(),
            }),
            (Self::ScoredTokens { .. }, None) => Err(Error::TrainingError {
                reason: "an unmasked micro-batch reached a masked step; the loader must mask \
                         every batch or none"
                    .to_string(),
            }),
        }
    }

    /// Reported loss of a step in which no micro-batch produced a loss term.
    ///
    /// A masked step with no scored token and no auxiliary loss reports `0`.
    /// An unmasked step always produces a term, so reaching this is an error.
    pub fn loss_without_terms(&self) -> Result<f64> {
        match self {
            Self::ScoredTokens { .. } => Ok(0.0),
            Self::MicroBatchMean { .. } => Err(Error::TrainingError {
                reason: "No loss accumulated".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::cross_entropy_masked;
    use boostr::autograd::{backward_wrt, var_narrow};
    use boostr::nn::var_contiguous;
    use boostr::{CpuClient, CpuDevice, CpuRuntime};

    const V: usize = 3;
    /// Six rows of logits: the trainable parameter of every test.
    const LOGITS: [f64; 18] = [
        2.0, 1.0, 0.1, //
        0.1, 2.0, 1.0, //
        -1.0, 0.5, 3.0, //
        0.7, -0.2, 0.3, //
        1.5, -0.5, 0.0, //
        -0.3, 0.9, 0.4, //
    ];
    const TARGETS: [i64; 6] = [0, 1, 2, 1, 0, 2];
    /// Scored rows of the merged batch for [`uneven`].
    const UNEVEN_MERGED: [f64; 6] = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        (CpuClient::new(device.clone()), device)
    }

    fn param(device: &CpuDevice) -> Var<CpuRuntime> {
        let t = Tensor::from_slice(&LOGITS, &[6, V], device).expect("logits");
        Var::new(t, true)
    }

    /// One micro-batch: rows `start..start + rows` of the parameter.
    struct Part {
        start: usize,
        rows: usize,
        mask: Option<Vec<f64>>,
    }

    fn masked(start: usize, mask: &[f64]) -> Part {
        Part {
            start,
            rows: mask.len(),
            mask: Some(mask.to_vec()),
        }
    }

    fn unmasked(start: usize, rows: usize) -> Part {
        Part {
            start,
            rows,
            mask: None,
        }
    }

    /// Micro-batch 0 scores 1 of 2 rows, micro-batch 1 scores 3 of 4.
    fn uneven() -> [Part; 2] {
        [masked(0, &[1.0, 0.0]), masked(2, &[1.0, 1.0, 0.0, 1.0])]
    }

    fn batch(part: &Part, device: &CpuDevice) -> Batch<CpuRuntime> {
        let rows = part.rows;
        let targets = &TARGETS[part.start..part.start + rows];
        let ids = Tensor::from_slice(targets, &[1, rows], device).expect("ids");
        let loss_mask = part
            .mask
            .as_ref()
            .map(|m| Tensor::from_slice(m, &[1, rows], device).expect("mask"));
        let scored = part
            .mask
            .as_ref()
            .map_or(rows, |m| m.iter().filter(|v| **v != 0.0).count());
        Batch {
            inputs: ids.clone(),
            targets: ids,
            loss_mask,
            scored,
        }
    }

    fn rows_of(w: &Var<CpuRuntime>, part: &Part) -> Var<CpuRuntime> {
        let narrowed = var_narrow(w, 0, part.start, part.rows).expect("narrow");
        var_contiguous(&narrowed).expect("contiguous")
    }

    /// Adds the gradient of `loss` w.r.t. `w` into `acc`.
    fn add_grad(acc: &mut [f64], loss: &Var<CpuRuntime>, w: &Var<CpuRuntime>, c: &CpuClient) {
        let grads = backward_wrt(loss, &[w.id()], c).expect("backward");
        let g: Vec<f64> = grads.get(w.id()).expect("gradient").to_vec();
        for (a, v) in acc.iter_mut().zip(g) {
            *a += v;
        }
    }

    fn scalar(v: &Var<CpuRuntime>) -> f64 {
        v.tensor().to_vec::<f64>()[0]
    }

    /// The trainer's accumulation through [`StepWeighting`]: one backward per
    /// micro-batch, summed. Returns the gradient and the summed scaled loss.
    fn accumulate(parts: &[Part]) -> (Vec<f64>, f64) {
        accumulate_weighted(parts, |batches, _| {
            StepWeighting::for_batches(batches).expect("weighting")
        })
    }

    /// [`accumulate`] with `weighting_of(batches, index)` weighting part `index`.
    fn accumulate_weighted(
        parts: &[Part],
        weighting_of: impl Fn(&[Batch<CpuRuntime>], usize) -> StepWeighting,
    ) -> (Vec<f64>, f64) {
        let (client, device) = setup();
        let w = param(&device);
        let batches: Vec<_> = parts.iter().map(|p| batch(p, &device)).collect();

        let mut grad = vec![0.0f64; LOGITS.len()];
        let mut loss = 0.0f64;
        for (index, (part, b)) in parts.iter().zip(&batches).enumerate() {
            let weighting = weighting_of(&batches, index);
            let scale = 1.0 / weighting.micro_batches() as f64;
            let logits = rows_of(&w, part);
            let targets = b.targets.reshape(&[part.rows]).expect("targets");
            let mask = b
                .loss_mask
                .as_ref()
                .map(|m| m.reshape(&[part.rows]).expect("mask"));
            let term = weighting
                .ce_term(&client, &logits, &targets, mask.as_ref(), b.scored)
                .expect("ce term");
            let Some(term) = term else { continue };
            let scaled = var_mul_scalar(&term, scale, &client).expect("scale");
            loss += scalar(&scaled);
            add_grad(&mut grad, &scaled, &w, &client);
        }
        (grad, loss)
    }

    /// One batch holding all six rows, under the masked mean.
    fn merged(mask: &[f64]) -> (Vec<f64>, f64) {
        let (client, device) = setup();
        let w = param(&device);
        let targets = Tensor::from_slice(&TARGETS, &[6], &device).expect("targets");
        let mask = Tensor::from_slice(mask, &[6], &device).expect("mask");
        let loss = cross_entropy_masked(&client, &w, &targets, &mask).expect("loss");
        let mut grad = vec![0.0f64; LOGITS.len()];
        add_grad(&mut grad, &loss, &w, &client);
        (grad, scalar(&loss))
    }

    fn assert_close(got: &[f64], want: &[f64]) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-12, "gradient[{i}]: {g} != {w}");
        }
    }

    #[test]
    fn uneven_mask_counts_match_one_merged_batch() {
        let (grad, loss) = accumulate(&uneven());
        let (want_grad, want_loss) = merged(&UNEVEN_MERGED);
        assert_close(&grad, &want_grad);
        assert!(
            (loss - want_loss).abs() < 1e-12,
            "loss {loss} != {want_loss}"
        );
    }

    /// Two ranks with one micro-batch each, weighted over both ranks, sum to
    /// the merged batch. The ranks score 1 of 2 and 3 of 4 rows.
    #[test]
    fn ranks_weighted_over_the_world_match_one_merged_batch() {
        let (grad, loss) = accumulate_weighted(&uneven(), |batches, index| {
            let global = batches.iter().map(|b| b.scored).sum();
            let local = StepWeighting::for_batches(&batches[index..=index]).expect("weighting");
            local.over_ranks(2, Some(global)).expect("over ranks")
        });
        let (want_grad, want_loss) = merged(&UNEVEN_MERGED);
        assert_close(&grad, &want_grad);
        assert!(
            (loss - want_loss).abs() < 1e-12,
            "loss {loss} != {want_loss}"
        );
    }

    #[test]
    fn over_ranks_scales_the_count_and_checks_the_form() {
        let mean = StepWeighting::MicroBatchMean { micro_batches: 2 };
        assert_eq!(
            mean.over_ranks(3, None).expect("unmasked"),
            StepWeighting::MicroBatchMean { micro_batches: 6 }
        );
        assert!(mean.over_ranks(3, Some(4)).is_err());
        assert!(mean.over_ranks(0, None).is_err());
        let masked = StepWeighting::ScoredTokens {
            micro_batches: 2,
            total_scored: 5,
        };
        assert_eq!(
            masked.over_ranks(2, Some(11)).expect("masked"),
            StepWeighting::ScoredTokens {
                micro_batches: 4,
                total_scored: 11
            }
        );
        assert!(masked.over_ranks(2, None).is_err());
        assert_eq!(mean.over_ranks(1, None).expect("world 1"), mean);
    }

    /// The per-micro-batch mean is the weighting this module replaces. It
    /// must differ from the merged batch here, or the test above proves nothing.
    #[test]
    fn per_micro_batch_mean_differs_from_the_merged_batch() {
        let (client, device) = setup();
        let w = param(&device);
        let mut grad = vec![0.0f64; LOGITS.len()];
        for part in uneven() {
            let b = batch(&part, &device);
            let targets = b.targets.reshape(&[part.rows]).expect("targets");
            let mask = b
                .loss_mask
                .expect("mask")
                .reshape(&[part.rows])
                .expect("mask");
            let mean =
                cross_entropy_masked(&client, &rows_of(&w, &part), &targets, &mask).expect("mean");
            let scaled = var_mul_scalar(&mean, 0.5, &client).expect("scale");
            add_grad(&mut grad, &scaled, &w, &client);
        }
        let (want, _) = merged(&UNEVEN_MERGED);
        let max_diff = grad
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0, f64::max);
        assert!(
            max_diff > 1e-3,
            "mean weighting matched the merged batch: {max_diff}"
        );
    }

    /// Unmasked steps keep the old path: the `cross_entropy` mean, then
    /// `1 / micro_batches`. Loss and gradient must match it bit for bit.
    #[test]
    fn unmasked_step_is_bit_identical_to_the_mean_over_accum_path() {
        let parts = [unmasked(0, 3), unmasked(3, 3)];
        let (grad, loss) = accumulate(&parts);

        let (client, device) = setup();
        let w = param(&device);
        let mut want_grad = vec![0.0f64; LOGITS.len()];
        let mut want_loss = 0.0f64;
        for part in &parts {
            let end = part.start + part.rows;
            let targets =
                Tensor::from_slice(&TARGETS[part.start..end], &[part.rows], &device).expect("t");
            let mean = cross_entropy(&client, &rows_of(&w, part), &targets).expect("mean");
            let scaled = var_mul_scalar(&mean, 1.0 / 2.0, &client).expect("scale");
            want_loss += scalar(&scaled);
            add_grad(&mut want_grad, &scaled, &w, &client);
        }

        assert_eq!(
            loss.to_bits(),
            want_loss.to_bits(),
            "loss {loss} != {want_loss}"
        );
        for (i, (g, w)) in grad.iter().zip(&want_grad).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "gradient[{i}]: {g} != {w}");
        }
    }

    #[test]
    fn a_zero_count_micro_batch_adds_no_term_and_no_nan() {
        let parts = [masked(0, &[0.0, 0.0]), masked(2, &[1.0, 1.0, 0.0, 1.0])];
        let (grad, loss) = accumulate(&parts);
        let (want_grad, want_loss) = merged(&[0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);

        assert!(loss.is_finite() && grad.iter().all(|g| g.is_finite()));
        assert_close(&grad, &want_grad);
        assert!(
            (loss - want_loss).abs() < 1e-12,
            "loss {loss} != {want_loss}"
        );
        assert!(
            grad[..2 * V].iter().all(|g| *g == 0.0),
            "unscored rows got a gradient"
        );
    }

    #[test]
    fn a_step_with_no_scored_token_reports_zero() {
        let parts = [masked(0, &[0.0, 0.0]), masked(2, &[0.0; 4])];
        let (grad, loss) = accumulate(&parts);
        assert_eq!(loss, 0.0);
        assert!(grad.iter().all(|g| *g == 0.0));

        let (_, device) = setup();
        let batches: Vec<_> = parts.iter().map(|p| batch(p, &device)).collect();
        let weighting = StepWeighting::for_batches(&batches).expect("weighting");
        let want = StepWeighting::ScoredTokens {
            micro_batches: 2,
            total_scored: 0,
        };
        assert_eq!(weighting, want);
        assert_eq!(weighting.loss_without_terms().expect("zero loss"), 0.0);
    }

    #[test]
    fn mixed_masked_and_unmasked_batches_are_an_error() {
        let (_, device) = setup();
        let parts = [masked(0, &[1.0, 0.0]), unmasked(2, 4)];
        let batches: Vec<_> = parts.iter().map(|p| batch(p, &device)).collect();
        let err = StepWeighting::for_batches(&batches).expect_err("mixed step");
        assert!(err.to_string().contains("1 of 2 micro-batches"), "{err}");
    }

    #[test]
    fn unmasked_steps_have_no_empty_loss() {
        let weighting = StepWeighting::MicroBatchMean { micro_batches: 2 };
        assert!(weighting.loss_without_terms().is_err());
    }

    #[test]
    fn only_a_masked_step_reports_its_global_scored_count() {
        let mean = StepWeighting::MicroBatchMean { micro_batches: 2 };
        assert_eq!(mean.scored_tokens(), None);
        let local = StepWeighting::ScoredTokens {
            micro_batches: 2,
            total_scored: 5,
        };
        assert_eq!(local.scored_tokens(), Some(5));
        let global = local.over_ranks(2, Some(11)).expect("over ranks");
        assert_eq!(global.scored_tokens(), Some(11));
    }
}
