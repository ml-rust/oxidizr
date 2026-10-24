//! [`Block::forward_checkpointed`]: one layer as one activation-checkpoint segment.
//!
//! The segment keeps only its inputs. The backward pass runs the unchanged
//! [`Block::forward`] again on an alias of the block. The alias keeps every
//! `TensorId`, so the gradients land on the model's own parameters.

use crate::data::BatchSegments;
use crate::model::{Block, BlockOutput, BlockType, FfnLayer};
use crate::prelude::{DType, Error, Result};
use boostr::autograd::{checkpoint_many_with_client, checkpoint_with_client, Var};
use boostr::nn::segment_masks::SegmentMasks;
use boostr::{NumrError, NumrResult};

/// The auxiliary losses the plain [`Block::forward`] returns.
enum AuxLosses {
    /// `aux_loss` and `z_loss` are `None`.
    Absent,
    /// Both are constant zeros: an MLA block over a dense MLP.
    ConstantZero,
    /// Both come from the MoE router and depend on the parameters.
    Routed,
}

impl<R> Block<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// The auxiliary losses of this arm, read from its FFN.
    fn aux_losses(&self) -> AuxLosses {
        let routed_or = |ffn: &FfnLayer<R>, dense: AuxLosses| match ffn {
            FfnLayer::Mlp(_) => dense,
            FfnLayer::Moe(_) => AuxLosses::Routed,
        };
        match &self.block_type {
            BlockType::Mamba { .. } | BlockType::Mamba2 { .. } | BlockType::Mamba3 { .. } => {
                AuxLosses::Absent
            }
            BlockType::Attention { ffn, .. } => routed_or(ffn, AuxLosses::ConstantZero),
            BlockType::LegacyAttention { ffn, .. } => routed_or(ffn, AuxLosses::Absent),
            BlockType::Gdn(layer) => routed_or(&layer.ffn, AuxLosses::Absent),
        }
    }

    /// [`Block::forward`] as one checkpoint segment.
    ///
    /// The segment inputs are `x` and every trainable parameter that requires
    /// grad. The forward keeps no activation inside the layer. The backward
    /// runs the layer again. On CPU the result equals the plain forward bit for
    /// bit, auxiliary losses included. CUDA agrees within a relative 1e-5.
    ///
    /// `segments` and `masks` are plain data, not segment inputs. The closure
    /// keeps a shared handle to each, so the recompute masks the same documents.
    ///
    /// # Errors
    ///
    /// The errors of [`Block::forward`], prefixed `checkpointed layer {layer_idx}`.
    pub(in crate::model) fn forward_checkpointed<C>(
        &self,
        client: &C,
        x: &Var<R>,
        layer_idx: usize,
        segments: Option<&BatchSegments<R>>,
        masks: Option<&SegmentMasks<R>>,
    ) -> Result<BlockOutput<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::ConvOps<R>
            + boostr::QuantMatmulOps<R>
            + boostr::RoPEOps<R>
            + boostr::NormalizationOps<R>
            + boostr::TypeConversionOps<R>
            + boostr::GatedDeltaNetOps<R>
            + 'static,
        R::Client: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::ConvOps<R>
            + boostr::FlashAttentionOps<R>
            + boostr::ops::VarLenAttentionOps<R>
            + boostr::TypeConversionOps<R>
            + boostr::DequantOps<R>
            + boostr::GatedDeltaNetOps<R>,
    {
        let params = self.trainable_parameters();
        let mut inputs: Vec<&Var<R>> = vec![x];
        inputs.extend(params.iter().map(|p| p.1).filter(|var| var.requires_grad()));
        let owned = self.alias();
        // `BatchSegments` clones share one `Arc`, so the closure holds no copy.
        let segments = segments.cloned();
        let masks = masks.cloned();
        match self.aux_losses() {
            AuxLosses::Routed => {
                let outputs = checkpoint_many_with_client(
                    move |xs: &[Var<R>], c: &C| {
                        let out = owned
                            .forward(c, first(xs, layer_idx)?, segments.as_ref(), masks.as_ref())
                            .map_err(|e| segment_error(layer_idx, e))?;
                        match (out.aux_loss, out.z_loss) {
                            (Some(aux), Some(z)) => Ok(vec![out.hidden, aux, z]),
                            _ => Err(NumrError::Internal(format!(
                                "checkpointed layer {layer_idx}: the MoE block returned no \
                                 auxiliary loss"
                            ))),
                        }
                    },
                    &inputs,
                    client,
                )
                .map_err(Error::Numr)?;
                let [hidden, aux_loss, z_loss]: [Var<R>; 3] =
                    outputs
                        .try_into()
                        .map_err(|got: Vec<Var<R>>| Error::ModelError {
                            reason: format!(
                                "checkpointed layer {layer_idx}: expected 3 outputs, got {}",
                                got.len()
                            ),
                        })?;
                Ok(BlockOutput {
                    hidden,
                    aux_loss: Some(aux_loss),
                    z_loss: Some(z_loss),
                })
            }
            dense => {
                let hidden = checkpoint_with_client(
                    move |xs: &[Var<R>], c: &C| {
                        owned
                            .forward(c, first(xs, layer_idx)?, segments.as_ref(), masks.as_ref())
                            .map(|out| out.hidden)
                            .map_err(|e| segment_error(layer_idx, e))
                    },
                    &inputs,
                    client,
                )
                .map_err(Error::Numr)?;
                let (aux_loss, z_loss) = match dense {
                    AuxLosses::ConstantZero => {
                        let device = x.tensor().device();
                        (
                            Some(FfnLayer::<R>::zero_loss(device)?),
                            Some(FfnLayer::<R>::zero_loss(device)?),
                        )
                    }
                    _ => (None, None),
                };
                Ok(BlockOutput {
                    hidden,
                    aux_loss,
                    z_loss,
                })
            }
        }
    }
}

/// The layer input, the first segment input.
fn first<R: boostr::Runtime>(xs: &[Var<R>], layer_idx: usize) -> NumrResult<&Var<R>> {
    xs.first().ok_or_else(|| {
        NumrError::Internal(format!("checkpointed layer {layer_idx}: no segment input"))
    })
}

/// A layer error as the numr error a checkpoint closure returns. The text
/// keeps the original message, so an out-of-memory error still reads as one.
fn segment_error(layer_idx: usize, e: Error) -> NumrError {
    NumrError::Backend(format!("checkpointed layer {layer_idx}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{self, Model, BATCH, SEQ};
    use crate::config::{ActivationCheckpointing, OxidizrModelConfig};
    use crate::data::BatchSegments;
    use crate::prelude::Tensor;
    use boostr::autograd::{
        backward, backward_wrt, var_add, var_mul, var_mul_scalar, var_reshape, var_sum, Var,
    };
    use boostr::{CpuClient, CpuRuntime};
    use std::collections::BTreeMap;

    /// Everything a step reads from one forward and backward.
    #[derive(Debug, PartialEq)]
    struct Outcome {
        logits: Vec<u32>,
        loss: Vec<u32>,
        /// Bits and `requires_grad` of each `aux_losses` entry.
        aux: Vec<(Vec<u32>, bool)>,
        /// Bits and `requires_grad` of each `z_losses` entry.
        z: Vec<(Vec<u32>, bool)>,
        /// Every trainable parameter's gradient, `None` when absent.
        grads: BTreeMap<String, Option<Vec<u32>>>,
    }

    fn bits(t: &Tensor<CpuRuntime>) -> Vec<u32> {
        let values: Vec<f32> = t.contiguous().expect("contiguous").to_vec();
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// The sum of every element of `v`.
    fn total(client: &CpuClient, v: &Var<CpuRuntime>) -> Var<CpuRuntime> {
        let flat = var_reshape(v, &[v.tensor().numel()]).expect("flatten");
        var_sum(&flat, &[0], false, client).expect("sum")
    }

    /// `sum(logits * w)` with fixed `w`, plus scaled auxiliary losses.
    fn loss(client: &CpuClient, out: &crate::model::ModelOutput<CpuRuntime>) -> Var<CpuRuntime> {
        let t = out.logits.tensor();
        let w: Vec<f32> = (0..t.numel())
            .map(|i| ((i * 37 + 11) % 101) as f32 / 101.0 - 0.5)
            .collect();
        let w = Tensor::<CpuRuntime>::from_slice(&w, t.shape(), t.device()).expect("weights");
        let weighted = var_mul(&out.logits, &Var::new(w, false), client).expect("mul");
        let mut loss = total(client, &weighted);
        let terms = out
            .aux_losses
            .iter()
            .map(|v| (v, 0.01))
            .chain(out.z_losses.iter().map(|v| (v, 0.001)));
        for (term, scale) in terms {
            let scaled = var_mul_scalar(&total(client, term), scale, client).expect("scale");
            loss = var_add(&loss, &scaled, client).expect("add");
        }
        loss
    }

    fn run(model: &Model, policy: &ActivationCheckpointing) -> Outcome {
        let (device, _) = fixtures::setup();
        run_on(model, policy, &fixtures::token_ids(&device), None)
    }

    /// [`run`] over `ids`, with the document `segments` when given.
    fn run_on(
        model: &Model,
        policy: &ActivationCheckpointing,
        ids: &Tensor<CpuRuntime>,
        segments: Option<&BatchSegments<CpuRuntime>>,
    ) -> Outcome {
        let (_, client) = fixtures::setup();
        let out = model
            .forward_with(&client, ids, policy, segments)
            .expect("forward");
        assert_eq!(out.logits.shape()[..2], [BATCH, SEQ]);
        let loss = loss(&client, &out);
        let grads = backward(&loss, &client).expect("backward");
        let flagged = |vars: &[Var<CpuRuntime>]| -> Vec<(Vec<u32>, bool)> {
            vars.iter()
                .map(|v| (bits(v.tensor()), v.requires_grad()))
                .collect()
        };
        Outcome {
            logits: bits(out.logits.tensor()),
            loss: bits(loss.tensor()),
            aux: flagged(&out.aux_losses[..]),
            z: flagged(&out.z_losses[..]),
            grads: model
                .named_trainable_parameters()
                .expect("named parameters")
                .into_iter()
                .map(|(name, id, _)| (name, grads.get(id).map(bits)))
                .collect(),
        }
    }

    /// Each quantity of `got` equals `want` bit for bit. The first mismatch
    /// names the diverging quantity.
    fn assert_bitwise(want: &Outcome, got: &Outcome, label: &str) {
        assert_eq!(got.logits, want.logits, "{label}: logits differ");
        assert_eq!(got.loss, want.loss, "{label}: loss differs");
        assert_eq!(got.aux, want.aux, "{label}: aux losses differ");
        assert_eq!(got.z, want.z, "{label}: z losses differ");
        assert_eq!(
            got.grads.keys().collect::<Vec<_>>(),
            want.grads.keys().collect::<Vec<_>>(),
            "{label}: trainable parameter names differ"
        );
        for (name, grad) in &want.grads {
            // A routed expert no token reaches has no gradient in either run.
            if !name.contains(".experts.") {
                assert!(grad.is_some(), "{label}: {name} gets no gradient");
            }
            assert_eq!(
                &got.grads[name], grad,
                "{label}: gradient of {name} differs"
            );
        }
    }

    /// `policy` over `cfg` matches the plain forward bit for bit.
    fn assert_matches_plain(
        cfg: &OxidizrModelConfig,
        policy: ActivationCheckpointing,
        label: &str,
    ) {
        let model = fixtures::build(cfg);
        let plain = run(&model, &ActivationCheckpointing::None);
        let checkpointed = run(&model, &policy);
        assert_bitwise(&plain, &checkpointed, label);
    }

    fn assert_all_matches_plain(cfg: &OxidizrModelConfig, label: &str) {
        assert_matches_plain(cfg, ActivationCheckpointing::All, label);
    }

    #[test]
    fn mamba1_recompute_is_bitwise() {
        assert_all_matches_plain(&fixtures::mamba1(), "mamba1");
    }

    #[test]
    fn mamba2_recompute_is_bitwise() {
        assert_all_matches_plain(&fixtures::mamba2(), "mamba2");
    }

    #[test]
    fn mamba3_recompute_is_bitwise() {
        assert_all_matches_plain(&fixtures::mamba3(), "mamba3");
    }

    /// The MLA arm reports constant zero losses over a dense MLP. The
    /// checkpointed run reports the same untracked zeros.
    #[test]
    fn mla_mlp_recompute_is_bitwise() {
        let cfg = fixtures::mla_mlp();
        assert_all_matches_plain(&cfg, "mla+mlp");
        let out = run(&fixtures::build(&cfg), &ActivationCheckpointing::All);
        assert_eq!(out.aux.len(), cfg.num_layers);
        assert!(out.aux.iter().chain(&out.z).all(|(_, tracked)| !tracked));
    }

    #[test]
    fn mla_moe_recompute_is_bitwise() {
        let cfg = fixtures::mla_moe();
        assert_all_matches_plain(&cfg, "mla+moe");
        let out = run(&fixtures::build(&cfg), &ActivationCheckpointing::All);
        assert_eq!(out.aux.len(), cfg.num_layers);
        assert!(out.aux.iter().chain(&out.z).all(|(_, tracked)| *tracked));
    }

    #[test]
    fn legacy_mlp_recompute_is_bitwise() {
        let cfg = fixtures::legacy_mlp();
        assert_all_matches_plain(&cfg, "legacy+mlp");
        let out = run(&fixtures::build(&cfg), &ActivationCheckpointing::All);
        assert!(out.aux.is_empty() && out.z.is_empty());
    }

    #[test]
    fn legacy_moe_recompute_is_bitwise() {
        assert_all_matches_plain(&fixtures::legacy_moe(), "legacy+moe");
    }

    #[test]
    fn gdn_moe_recompute_is_bitwise() {
        assert_all_matches_plain(&fixtures::gdn_moe(), "gdn+moe");
    }

    /// Under a document mask every attention arm recomputes bit for bit,
    /// and the mask changes the logits.
    #[test]
    fn document_masked_recompute_is_bitwise() {
        let (device, _) = fixtures::setup();
        let (ids, segments) = fixtures::segmented_batch(&device);
        let arms = [
            (fixtures::legacy_mlp(), "legacy+mlp"),
            (fixtures::legacy_moe(), "legacy+moe"),
            (fixtures::mla_mlp(), "mla+mlp"),
            (fixtures::mla_moe(), "mla+moe"),
        ];
        for (cfg, label) in arms {
            let model = fixtures::build(&cfg);
            let plain = run_on(
                &model,
                &ActivationCheckpointing::None,
                &ids,
                Some(&segments),
            );
            let recomputed = run_on(&model, &ActivationCheckpointing::All, &ids, Some(&segments));
            assert_bitwise(&plain, &recomputed, label);
            let unmasked = run_on(&model, &ActivationCheckpointing::None, &ids, None);
            assert_ne!(
                plain.logits, unmasked.logits,
                "{label}: the mask changed nothing"
            );
        }
    }

    /// `{layers: [1]}` on three layers checkpoints the middle layer alone.
    #[test]
    fn one_listed_layer_of_three_is_bitwise() {
        let cfg = OxidizrModelConfig {
            num_layers: 3,
            ..fixtures::legacy_mlp()
        };
        assert_matches_plain(&cfg, ActivationCheckpointing::Layers(vec![1]), "layers [1]");
    }

    /// Under a frozen LoRA base the checkpointed run trains the adapters
    /// alone: a frozen parameter gets no gradient.
    #[test]
    fn a_frozen_lora_base_gets_no_gradient_under_recompute() {
        let cfg = fixtures::legacy_lora_frozen();
        assert_all_matches_plain(&cfg, "legacy+lora");

        let model = fixtures::build(&cfg);
        let (device, client) = fixtures::setup();
        let out = model
            .forward_with(
                &client,
                &fixtures::token_ids(&device),
                &ActivationCheckpointing::All,
                None,
            )
            .expect("forward");
        // The trainer asks backward for the trainable ids alone. This test asks the same.
        let trainable: Vec<boostr::autograd::TensorId> = model
            .trainable_parameters()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let grads = backward_wrt(&loss(&client, &out), &trainable, &client).expect("backward");
        for id in &trainable {
            assert!(grads.get(*id).is_some(), "trainable {id:?} got no gradient");
        }
        let frozen: Vec<(String, boostr::autograd::TensorId)> = model
            .named_parameters()
            .expect("named parameters")
            .into_iter()
            .filter(|(_, _, var)| !var.requires_grad())
            .map(|(name, id, _)| (name, id))
            .collect();
        assert!(
            frozen
                .iter()
                .any(|(name, _)| name == "layers.0.self_attn.q_proj.weight"),
            "{frozen:?}"
        );
        for (name, id) in frozen {
            assert!(grads.get(id).is_none(), "frozen {name} got a gradient");
        }
    }
}
