//! Training FLOPs per token and model FLOPs utilization (MFU).
//!
//! The estimate follows the PaLM appendix B form: `6N + 12 * L * W * T`.
//!
//! - `N` counts the weights that multiply activations: every parameter of
//!   rank 2 or more. Norm scales, biases and the SSM `A`/`D` vectors are rank 1
//!   and stay out.
//! - The input embedding is a row gather, not a matmul, so it stays out. The
//!   output head is a matmul and counts once. A tied head counts the shared
//!   matrix once, as the head.
//! - A routed MoE expert counts `experts_per_tok / num_experts` of its
//!   weights, the share of tokens it sees on average. The router and the
//!   shared expert see every token and count in full.
//! - A trainable weight costs 6 FLOPs per token: 2 forward, 2 for the input
//!   gradient, 2 for the weight gradient. A frozen weight, such as a LoRA base
//!   weight, costs 4: it has no weight gradient.
//! - The attention term `12 * L * W * T` covers the score and value matmuls,
//!   forward and backward. `L` is the attention layers, `W = num_heads *
//!   head_dim`, and `T` is `max_seq_len`, capped at `sliding_window`.
//!
//! Limits: the count ignores the causal mask, the softmax, norms and
//! activations, the MLA latent width, the Mamba scan and the Gated DeltaNet
//! recurrence. Mamba and Gated DeltaNet layers add only their weights.
//! Activation recompute adds no FLOPs here. MFU from this count is therefore
//! a lower bound of the work the device does.

use crate::config::OxidizrModelConfig;
use crate::model::LitGPT;
use crate::prelude::{DType, Result};
use boostr::runtime::Runtime;
use std::collections::BTreeSet;

/// One parameter tensor as the FLOP count reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatmulParam<'a> {
    /// Name as `LitGPT::named_parameters` gives it.
    pub name: &'a str,
    pub shape: &'a [usize],
    pub trainable: bool,
}

/// The config facts the FLOP count needs beyond the parameter shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlopsModel {
    pub tie_word_embeddings: bool,
    /// Routed experts per MoE layer. `None` without MoE.
    pub num_experts: Option<usize>,
    /// Routed experts each token visits. `None` reads as 1.
    pub experts_per_tok: Option<usize>,
    /// Layers that run attention: not Mamba, not Gated DeltaNet.
    pub attention_layers: usize,
    /// `num_heads * head_dim`.
    pub attention_width: usize,
    /// Keys each query attends to: `max_seq_len`, capped at `sliding_window`.
    pub attention_span: usize,
}

impl FlopsModel {
    /// The facts of `cfg`.
    pub fn of_config(cfg: &OxidizrModelConfig) -> Self {
        let mut other: BTreeSet<usize> = BTreeSet::new();
        for list in [&cfg.mamba_layers, &cfg.gdn_layers].into_iter().flatten() {
            other.extend(list.iter().copied().filter(|&i| i < cfg.num_layers));
        }
        let head_dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_heads.max(1));
        let span = match cfg.sliding_window {
            Some(window) => cfg.max_seq_len.min(window),
            None => cfg.max_seq_len,
        };
        Self {
            tie_word_embeddings: cfg.tie_word_embeddings,
            num_experts: cfg.num_experts,
            experts_per_tok: cfg.experts_per_tok,
            attention_layers: cfg.num_layers - other.len(),
            attention_width: cfg.num_heads * head_dim,
            attention_span: span,
        }
    }

    /// Share of tokens one routed expert sees.
    fn expert_share(&self) -> f64 {
        match self.num_experts {
            Some(experts) if experts > 0 => {
                self.experts_per_tok.unwrap_or(1).min(experts) as f64 / experts as f64
            }
            _ => 1.0,
        }
    }
}

/// Training FLOPs per token of `params` under `model`. See the module docs.
pub fn flops_per_token(params: &[MatmulParam<'_>], model: &FlopsModel) -> f64 {
    let expert_share = model.expert_share();
    let mut weights = 0.0;
    for param in params.iter().filter(|p| p.shape.len() >= 2) {
        if param.name.ends_with("embed_tokens.weight") && !model.tie_word_embeddings {
            continue;
        }
        let share = if param.name.contains(".moe.experts.") {
            expert_share
        } else {
            1.0
        };
        let per_weight = if param.trainable { 6.0 } else { 4.0 };
        let elems: usize = param.shape.iter().product();
        weights += per_weight * elems as f64 * share;
    }
    let attention = 12.0
        * model.attention_layers as f64
        * model.attention_width as f64
        * model.attention_span as f64;
    weights + attention
}

/// Training FLOPs per token of `model`, built from `cfg`.
///
/// # Errors
/// Returns an error when the model cannot list its parameters.
pub fn model_flops_per_token<R>(model: &LitGPT<R>, cfg: &OxidizrModelConfig) -> Result<f64>
where
    R: Runtime<DType = DType>,
{
    let named = model.named_parameters()?;
    let params: Vec<MatmulParam<'_>> = named
        .iter()
        .map(|(name, _, var)| MatmulParam {
            name,
            shape: var.tensor().shape(),
            trainable: var.requires_grad(),
        })
        .collect();
    Ok(flops_per_token(&params, &FlopsModel::of_config(cfg)))
}

/// Model FLOPs utilization: achieved FLOP/s over the peak of every device.
///
/// `tokens_per_sec` counts every rank. `peak_tflops` is one device's peak.
/// Returns 0 when the world or the peak is 0.
pub fn mfu(flops_per_token: f64, tokens_per_sec: f64, world: usize, peak_tflops: f64) -> f64 {
    let peak = world as f64 * peak_tflops * 1e12;
    if peak > 0.0 {
        flops_per_token * tokens_per_sec / peak
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p<'a>(name: &'a str, shape: &'a [usize], trainable: bool) -> MatmulParam<'a> {
        MatmulParam {
            name,
            shape,
            trainable,
        }
    }

    fn model(tied: bool) -> FlopsModel {
        FlopsModel {
            tie_word_embeddings: tied,
            num_experts: Some(4),
            experts_per_tok: Some(1),
            attention_layers: 2,
            attention_width: 16,
            attention_span: 8,
        }
    }

    /// Hand count:
    /// q_proj 256 * 6 = 1536, gate_proj 512 * 6 = 3072, frozen v_proj 256 * 4 = 1024,
    /// head 384 * 6 = 2304, one expert 512 * 6 / 4 = 768, attention 12 * 2 * 16 * 8 = 3072.
    /// Total 11776. The embedding gather and the rank-1 norm add nothing.
    const HAND_COUNT: f64 = 11776.0;

    #[test]
    fn an_untied_model_counts_the_head_and_skips_the_embedding() {
        let params = [
            p("embed_tokens.weight", &[24, 16], true),
            p("layers.0.self_attn.q_proj.weight", &[16, 16], true),
            p("layers.0.mlp.gate_proj.weight", &[32, 16], true),
            p("layers.0.self_attn.v_proj.weight", &[16, 16], false),
            p("layers.1.moe.experts.0.up_proj.weight", &[32, 16], true),
            p("norm.weight", &[16], true),
            p("lm_head.weight", &[24, 16], true),
        ];
        assert_eq!(flops_per_token(&params, &model(false)), HAND_COUNT);
    }

    #[test]
    fn a_tied_model_counts_the_shared_matrix_once_as_the_head() {
        let params = [
            p("embed_tokens.weight", &[24, 16], true),
            p("layers.0.self_attn.q_proj.weight", &[16, 16], true),
            p("layers.0.mlp.gate_proj.weight", &[32, 16], true),
            p("layers.0.self_attn.v_proj.weight", &[16, 16], false),
            p("layers.1.moe.experts.0.up_proj.weight", &[32, 16], true),
            p("norm.weight", &[16], true),
        ];
        assert_eq!(flops_per_token(&params, &model(true)), HAND_COUNT);
    }

    #[test]
    fn mfu_divides_by_the_peak_of_every_device() {
        // 6e9 FLOPs per token at 100 tokens/s over 2 devices of 0.6 TFLOP/s.
        assert!((mfu(6e9, 100.0, 2, 0.6) - 0.5).abs() < 1e-12);
        assert_eq!(mfu(6e9, 100.0, 0, 0.6), 0.0);
    }

    #[test]
    fn the_config_facts_skip_ssm_layers_and_cap_the_span() {
        let mut cfg = crate::config::LitConfig::nano_baseline().model;
        cfg.num_layers = 6;
        cfg.mamba_layers = Some(vec![0, 1]);
        cfg.gdn_layers = Some(vec![1, 2, 9]);
        cfg.max_seq_len = 512;
        cfg.sliding_window = Some(128);
        let facts = FlopsModel::of_config(&cfg);
        assert_eq!(
            facts.attention_layers, 3,
            "layers 0, 1 and 2 are not attention"
        );
        assert_eq!(facts.attention_span, 128);
        assert_eq!(facts.attention_width, cfg.hidden_size);
    }
}
