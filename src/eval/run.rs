//! One eval pass: the forward-only cross-entropy over the eval batches.

use crate::data::{DataPosition, LitDataLoader};
use crate::model::LitGPT;
use crate::prelude::{DType, Error, Result, Tensor};
use boostr::autograd::{no_grad, var_reshape};
use boostr::nn::loss::cross_entropy_loss_masked_sum;
use boostr::ops::TensorOps;
use boostr::readback::scalar_f32;
use boostr::runtime::{Runtime, RuntimeClient};
use boostr::tensor::TensorId;
use std::collections::HashMap;

/// The summed loss of one eval pass and the tokens it scored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalTotals {
    /// Cross-entropy summed over every scored token of every eval batch.
    pub loss_sum: f64,
    /// Scored tokens over every eval batch.
    pub tokens: usize,
    pub batches: usize,
}

impl EvalTotals {
    /// Mean cross-entropy per scored token.
    pub fn loss(&self) -> f64 {
        self.loss_sum / self.tokens as f64
    }

    /// `exp(loss)`.
    pub fn perplexity(&self) -> f64 {
        self.loss().exp()
    }
}

/// Scores `batches` batches of `loader` from its first row under the weights in `params`.
///
/// The pass runs under `no_grad`, so it records no graph and touches no
/// optimizer state. Every pass starts at the first row of the loader's range,
/// so two passes over the same weights see the same batches. The loss is the
/// masked cross-entropy sum of training over the scored tokens. A batch
/// without a mask scores every position. Auxiliary MoE losses stay out.
///
/// # Errors
/// Returns an error when a batch or the forward pass fails, or when no eval
/// batch scores a token.
pub fn evaluate<R>(
    client: &R::Client,
    model: &LitGPT<R>,
    params: &HashMap<TensorId, Tensor<R>>,
    loader: &mut LitDataLoader<R>,
    batches: usize,
) -> Result<EvalTotals>
where
    R: Runtime<DType = DType>,
    R::Client: RuntimeClient<R>
        + TensorOps<R>
        + boostr::ScalarOps<R>
        + boostr::BinaryOps<R>
        + boostr::CompareOps<R>
        + boostr::ConvOps<R>
        + boostr::FlashAttentionOps<R>
        + boostr::IndexingOps<R>
        + boostr::NormalizationOps<R>
        + boostr::QuantMatmulOps<R>
        + boostr::RoPEOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::DequantOps<R>
        + boostr::GatedDeltaNetOps<R>,
{
    loader.seek(DataPosition::default())?;
    let totals = no_grad(|| -> Result<EvalTotals> {
        // The current weights. Leaves keep their flag, but no op under `no_grad` records a graph.
        let eval_model = model.rebuild_with_parameters(params)?;
        let mut totals = EvalTotals {
            loss_sum: 0.0,
            tokens: 0,
            batches,
        };
        for _ in 0..batches {
            let batch = loader.next_batch()?;
            if batch.scored == 0 {
                continue;
            }
            let output = eval_model.forward(client, &batch.inputs)?;
            let (b, s, v) = output.logits.tensor().dims3()?;
            let logits = var_reshape(&output.logits, &[b * s, v])?;
            let targets = batch.targets.reshape(&[b * s])?;
            let mask = match &batch.loss_mask {
                Some(mask) => mask.reshape(&[b * s])?,
                None => Tensor::<R>::ones(&[b * s], DType::F32, batch.targets.device())?,
            };
            let sum = cross_entropy_loss_masked_sum(client, &logits, &targets, &mask)?;
            totals.loss_sum += f64::from(scalar_f32(client, sum.tensor())?);
            totals.tokens += batch.scored;
        }
        Ok(totals)
    })?;
    if totals.tokens == 0 {
        return Err(Error::TrainingError {
            reason: format!(
                "the {batches} eval batches score no token under the loss mask, so the eval \
                 loss is undefined. Raise trainer.eval.batches, or eval on rows the mask scores."
            ),
        });
    }
    Ok(totals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loss_is_the_sum_over_scored_tokens_and_perplexity_its_exp() {
        let totals = EvalTotals {
            loss_sum: 30.0,
            tokens: 12,
            batches: 3,
        };
        assert_eq!(totals.loss(), 2.5);
        assert_eq!(totals.perplexity().to_bits(), 2.5f64.exp().to_bits());
    }
}
