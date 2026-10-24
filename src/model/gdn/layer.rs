//! [`GdnLayer`]: one Gated DeltaNet layer of the trainable model.
//!
//! The layer mirrors the attention arms of `Block`: `input_layernorm`, the
//! mixer, the residual add, `post_attention_layernorm`, the FFN (MLP or
//! MoE) and the residual add. The mixer is boostr's `GdnBlock`, which takes
//! the normed input and returns its `out_proj` output.
//!
//! Checkpoint names are in `names`, LoRA freeze and merge in `lora`.

use super::mixer::{self, MixerWithIds};
use crate::config::{LoraConfig, OxidizrModelConfig};
use crate::model::{
    Block, BlockOutput, FfnLayer, FrozenParams, LitGPT, MlpWeightsWithIds, MoeWeightsWithIds,
    WeightWithId,
};
use crate::prelude::{DType, Result, RmsNorm, Tensor, VarBuilder};
use boostr::autograd::{TensorId, Var};
use boostr::model::config::{GdnConfig, ModelConfig};
use boostr::model::hybrid::GdnBlock;
use boostr::nn::segment_masks::SegmentMasks;
use std::collections::HashMap;

/// Name segment of the mixer under `layers.N`, as Qwen3.5 checkpoints name it.
pub(super) const MIXER: &str = "linear_attn";

/// `layers.N.linear_attn`, the mixer prefix errors name.
pub(super) fn mixer_prefix(layer_idx: usize) -> String {
    format!("layers.{layer_idx}.{MIXER}")
}

/// One Gated DeltaNet layer: norm, mixer, residual, norm, FFN, residual.
pub(in crate::model) struct GdnLayer<R: boostr::Runtime> {
    /// The GDN mixer.
    pub(in crate::model) block: GdnBlock<R>,
    /// Pre-norm on the mixer input.
    pub(in crate::model) input_layernorm: RmsNorm<R>,
    /// MLP or MoE, chosen by the model config as on the attention layers.
    pub(in crate::model) ffn: FfnLayer<R>,
    /// Pre-norm on the FFN input.
    pub(in crate::model) post_attention_layernorm: RmsNorm<R>,
}

/// Every tensor of a [`GdnLayer`] with its stable id, for
/// [`GdnLayer::with_ids`].
pub(in crate::model) struct GdnLayerWithIds<R: boostr::Runtime> {
    /// The mixer's shape.
    pub(in crate::model) config: GdnConfig,
    pub(in crate::model) mixer: MixerWithIds<R>,
    /// Exactly one of `mlp` and `moe` is `Some`, chosen by the model config.
    pub(in crate::model) mlp: Option<MlpWeightsWithIds<R>>,
    pub(in crate::model) moe: Option<MoeWeightsWithIds<R>>,
    pub(in crate::model) input_layernorm: WeightWithId<R>,
    pub(in crate::model) post_attention_layernorm: WeightWithId<R>,
}

impl<R> GdnLayer<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// A fresh layer from `vb` (`layers.N`). A tensor `vb` holds is taken
    /// as is, so a checkpoint load runs through here too.
    ///
    /// # Errors
    ///
    /// - `gdn.hidden_size` differs from `cfg.hidden_size`.
    /// - The errors of the mixer init, the FFN and the norms.
    pub(in crate::model) fn new<C>(
        cfg: &ModelConfig,
        gdn: &GdnConfig,
        layer_idx: usize,
        mut vb: VarBuilder<'_, R>,
        client: &C,
        dtype: DType,
        lora: Option<&LoraConfig>,
    ) -> Result<Self>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        if gdn.hidden_size != cfg.hidden_size {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "{}: gdn hidden_size {} differs from the model hidden_size {}",
                    mixer_prefix(layer_idx),
                    gdn.hidden_size,
                    cfg.hidden_size
                ),
            });
        }
        let eps = cfg.rms_norm_eps as f32;
        let hidden = cfg.hidden_size;
        let ffn_prefix = if cfg.moe.is_some() { "moe" } else { "mlp" };
        let block = {
            let mut mixer_vb = vb.pp(MIXER);
            mixer::init(
                gdn,
                &mut mixer_vb,
                dtype,
                client,
                lora,
                &mixer_prefix(layer_idx),
            )?
        };
        Ok(Self {
            block,
            ffn: FfnLayer::new(cfg, vb.pp(ffn_prefix), client, dtype, lora)?,
            input_layernorm: Block::<R>::take_rms_norm(
                vb.pp("input_layernorm"),
                hidden,
                eps,
                dtype,
                client,
            )?,
            post_attention_layernorm: Block::<R>::take_rms_norm(
                vb.pp("post_attention_layernorm"),
                hidden,
                eps,
                dtype,
                client,
            )?,
        })
    }

    /// The layer over `weights`, keeping every id. `frozen` decides each flag.
    ///
    /// The mixer computes `-exp(A_log)` once at build, on the default client
    /// of `device`. No training forward reads that value.
    ///
    /// # Errors
    ///
    /// The errors of the mixer build, the FFN and the norm shape checks.
    pub(in crate::model) fn with_ids(
        cfg: &ModelConfig,
        weights: GdnLayerWithIds<R>,
        device: &R::Device,
        frozen: &FrozenParams,
        layer_idx: usize,
    ) -> Result<Self>
    where
        R::Client: boostr::UnaryOps<R>,
    {
        let GdnLayerWithIds {
            config,
            mixer,
            mlp,
            moe,
            input_layernorm,
            post_attention_layernorm,
        } = weights;
        let client = R::default_client(device);
        let eps = cfg.rms_norm_eps as f32;
        let hidden = cfg.hidden_size;
        let prefix = mixer_prefix(layer_idx);
        Ok(Self {
            block: mixer::build(&client, &config, mixer, &prefix, frozen)?,
            ffn: FfnLayer::with_ids(cfg, mlp, moe, frozen)?,
            input_layernorm: Block::<R>::rms_norm_with_id(
                "input_layernorm.weight",
                input_layernorm.0,
                input_layernorm.1,
                hidden,
                eps,
                frozen,
            )?,
            post_attention_layernorm: Block::<R>::rms_norm_with_id(
                "post_attention_layernorm.weight",
                post_attention_layernorm.0,
                post_attention_layernorm.1,
                hidden,
                eps,
                frozen,
            )?,
        })
    }

    /// Every tensor with its id, each trainable one read from `params`.
    ///
    /// # Errors
    ///
    /// A trainable tensor absent from `params`, or an adapted projection
    /// while `cfg.lora` is `None`.
    pub(in crate::model) fn snapshot(
        &self,
        cfg: &OxidizrModelConfig,
        params: &HashMap<TensorId, Tensor<R>>,
        layer_idx: usize,
    ) -> Result<GdnLayerWithIds<R>> {
        let alpha = cfg.lora.as_ref().map(|lora| lora.alpha);
        let (mlp, moe) = LitGPT::<R>::ffn_weights_with_ids(cfg, &self.ffn, params)?;
        Ok(GdnLayerWithIds {
            config: self.block.config().clone(),
            mixer: mixer::snapshot(&self.block, params, alpha, &mixer_prefix(layer_idx))?,
            mlp,
            moe,
            input_layernorm: LitGPT::<R>::tensor_with_stable_id(
                self.input_layernorm.weight(),
                params,
            )?,
            post_attention_layernorm: LitGPT::<R>::tensor_with_stable_id(
                self.post_attention_layernorm.weight(),
                params,
            )?,
        })
    }

    /// `norm -> mixer -> + x -> norm -> ffn -> + residual`.
    ///
    /// The mixer runs `GdnBlock::forward_train`. Each call starts from a
    /// zero conv window and a zero recurrent state. With `masks`, the conv
    /// window and the state also reset at each document start. Under
    /// `trainer.activation_checkpointing`, `Block::forward_checkpointed` runs
    /// this whole call as one recompute segment.
    ///
    /// A dense MLP reports no auxiliary loss, as on the attention layers.
    ///
    /// # Errors
    ///
    /// The errors of the norms, the mixer, the FFN and the residual adds.
    pub(in crate::model) fn forward<C>(
        &self,
        client: &C,
        x: &Var<R>,
        masks: Option<&SegmentMasks<R>>,
    ) -> Result<BlockOutput<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::CompareOps<R>
            + boostr::NormalizationOps<R>
            + boostr::QuantMatmulOps<R>
            + boostr::TypeConversionOps<R>
            + boostr::GatedDeltaNetOps<R>,
        R::Client: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::CompareOps<R>
            + boostr::TypeConversionOps<R>
            + boostr::DequantOps<R>
            + boostr::GatedDeltaNetOps<R>,
    {
        let h = self.input_layernorm.forward(client, x)?;
        let h = self.block.forward_train(client, &h, masks)?;
        let x = boostr::autograd::var_add(&h, x, client).map_err(boostr::error::Error::Numr)?;

        let h = self.post_attention_layernorm.forward(client, &x)?;
        let (h, aux_loss, z_loss) = self.ffn.forward(client, &h)?;
        let hidden =
            boostr::autograd::var_add(&h, &x, client).map_err(boostr::error::Error::Numr)?;
        let (aux_loss, z_loss) = match self.ffn {
            FfnLayer::Mlp(_) => (None, None),
            FfnLayer::Moe(_) => (Some(aux_loss), Some(z_loss)),
        };
        Ok(BlockOutput {
            hidden,
            aux_loss,
            z_loss,
        })
    }

    /// The same layer over the same parameters, each `TensorId` and flag kept.
    pub(in crate::model) fn alias(&self) -> Self {
        Self {
            block: self.block.alias(),
            input_layernorm: self.input_layernorm.alias(),
            ffn: self.ffn.alias(),
            post_attention_layernorm: self.post_attention_layernorm.alias(),
        }
    }

    /// Trainable parameters in `named_parameters` order.
    pub(in crate::model) fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        let mut params = self.input_layernorm.trainable_parameters();
        params.extend(self.block.trainable_parameters());
        params.extend(self.post_attention_layernorm.trainable_parameters());
        params.extend(self.ffn.trainable_parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::GdnLayer;
    use crate::config::{GdnSpec, LitConfig, LoraConfig, OxidizrModelConfig};
    use crate::model::{BlockType, LitGPT};
    use crate::prelude::{Tensor, VarBuilder, VarMap};
    use boostr::autograd::{backward, var_add, var_mul, var_sum, TensorId, Var};
    use boostr::model::hybrid::GdnBlock;
    use boostr::{CpuClient, CpuDevice, CpuRuntime};
    use std::collections::HashMap;

    type Model = LitGPT<CpuRuntime>;

    const BATCH: usize = 2;
    const SEQ: usize = 8;
    const HIDDEN: usize = 16;
    const GDN_TARGETS: [&str; 5] = [
        "in_proj_qkv",
        "in_proj_z",
        "alpha_proj",
        "beta_proj",
        "out_proj",
    ];

    /// Two layers: GDN layer 0, dense GQA attention layer 1. Chunks of 4
    /// over 8 tokens exercise the chunked recurrence.
    fn cfg(lora: Option<LoraConfig>) -> OxidizrModelConfig {
        let mut cfg = LitConfig::nano_baseline().model;
        cfg.name = "gdn-test".to_string();
        cfg.hidden_size = HIDDEN;
        cfg.num_layers = 2;
        cfg.num_heads = 4;
        cfg.kv_heads = Some(2);
        cfg.intermediate_size = Some(32);
        cfg.vocab_size = 12;
        cfg.tokenizer_vocab = "custom".to_string();
        cfg.max_seq_len = SEQ;
        cfg.gdn_layers = Some(vec![0]);
        cfg.gdn = Some(GdnSpec {
            state_size: 4,
            key_heads: 2,
            value_heads: 4,
            conv_kernel: 4,
            chunk_size: 4,
            rms_eps: 1e-6,
        });
        cfg.lora = lora;
        cfg
    }

    fn lora(freeze_base: bool) -> LoraConfig {
        LoraConfig {
            rank: 2,
            alpha: 4.0,
            target_modules: GDN_TARGETS.iter().map(|t| t.to_string()).collect(),
            freeze_base,
            train_modules: Vec::new(),
        }
    }

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    fn build(cfg: &OxidizrModelConfig, seed: u64) -> Model {
        let (device, client) = setup();
        let mut map = VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut map, &device).with_seed(seed);
        LitGPT::new(cfg, &mut vb, &client).expect("the GDN hybrid builds")
    }

    fn gdn_layer(model: &Model) -> &GdnLayer<CpuRuntime> {
        match &model.layers[0].block_type {
            BlockType::Gdn(layer) => layer,
            _ => panic!("layer 0 must be a GDN layer"),
        }
    }

    fn bits(t: &Tensor<CpuRuntime>) -> Vec<u32> {
        let values: Vec<f32> = t.contiguous().expect("contiguous").to_vec();
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// Deterministic values in `[-0.5, 0.5)`.
    fn values(len: usize, salt: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i * 37 + salt * 11) % 101) as f32 / 101.0 - 0.5)
            .collect()
    }

    fn input(device: &CpuDevice) -> Var<CpuRuntime> {
        let shape = [BATCH, SEQ, HIDDEN];
        let x = Tensor::<CpuRuntime>::from_slice(&values(BATCH * SEQ * HIDDEN, 1), &shape, device)
            .expect("input");
        Var::new(x, true)
    }

    /// `sum(y * w)` with a fixed `w`, so every output gets its own gradient.
    fn weighted_loss(client: &CpuClient, y: &Var<CpuRuntime>) -> Var<CpuRuntime> {
        let t = y.tensor();
        let w = Tensor::<CpuRuntime>::from_slice(&values(t.numel(), 2), t.shape(), t.device())
            .expect("weights");
        let prod = var_mul(y, &Var::new(w, false), client).expect("mul");
        var_sum(&prod, &[0, 1, 2], false, client).expect("sum")
    }

    fn token_ids(device: &CpuDevice) -> Tensor<CpuRuntime> {
        let ids: Vec<i64> = (0..BATCH * SEQ).map(|i| (i * 5 % 12) as i64).collect();
        Tensor::<CpuRuntime>::from_slice(&ids, &[BATCH, SEQ], device).expect("token ids")
    }

    fn logits(model: &Model) -> Vec<u32> {
        let (device, client) = setup();
        let output = model
            .forward(&client, &token_ids(&device))
            .expect("forward");
        bits(output.logits.tensor())
    }

    fn trainable_map(model: &Model) -> HashMap<TensorId, Tensor<CpuRuntime>> {
        model
            .trainable_parameters()
            .into_iter()
            .map(|(id, var)| (id, var.tensor().clone()))
            .collect()
    }

    fn names(model: &Model) -> Vec<String> {
        let params = model.named_parameters().expect("named parameters");
        params.into_iter().map(|(name, _, _)| name).collect()
    }

    fn trainable_names(model: &Model) -> Vec<String> {
        let params = model
            .named_trainable_parameters()
            .expect("named parameters");
        params.into_iter().map(|(name, _, _)| name).collect()
    }

    /// The mixer of a rebuilt model, the one every training step runs,
    /// equals boostr's `GdnBlock::load` over the same tensors: forward and
    /// every gradient, bit for bit.
    #[test]
    fn the_mixer_matches_a_boostr_block_loaded_from_its_tensors() {
        let (device, client) = setup();
        let fresh = build(&cfg(None), 0x6d01);
        let model = fresh
            .rebuild_with_parameters(&trainable_map(&fresh))
            .expect("rebuild");
        let layer = gdn_layer(&model);
        let mut map = VarMap::<CpuRuntime>::new();
        for (name, var) in layer.block.named_parameters() {
            map.insert(name, var.tensor().clone());
        }
        let mut vb = VarBuilder::new(&mut map, &device);
        let reference = GdnBlock::load(layer.block.config(), &mut vb, &client, true)
            .expect("boostr loads the mixer's tensors");

        let x = input(&device);
        let ours = layer
            .block
            .forward_train(&client, &x, None)
            .expect("forward");
        let want = reference.forward_train(&client, &x, None).expect("forward");
        assert_eq!(bits(ours.tensor()), bits(want.tensor()));

        let grads = backward(&weighted_loss(&client, &ours), &client).expect("backward");
        let want_grads = backward(&weighted_loss(&client, &want), &client).expect("backward");
        let x_grad = grads.get(x.id()).expect("x gets a gradient");
        let want_x_grad = want_grads.get(x.id()).expect("x gets a gradient");
        assert_eq!(bits(x_grad), bits(want_x_grad));
        let pairs = layer
            .block
            .named_parameters()
            .into_iter()
            .zip(reference.named_parameters());
        for ((name, var), (want_name, want_var)) in pairs {
            assert_eq!(name, want_name);
            let got = grads
                .get(var.id())
                .unwrap_or_else(|| panic!("{name}: no gradient"));
            let want = want_grads.get(want_var.id()).expect("reference gradient");
            assert_eq!(bits(got), bits(want), "{name}: gradient differs");
        }
    }

    /// The layer is `norm -> mixer -> + x -> norm -> MLP -> + residual`, and
    /// a dense MLP reports no auxiliary loss.
    #[test]
    fn the_layer_wraps_the_mixer_like_an_attention_layer() {
        let (device, client) = setup();
        let model = build(&cfg(None), 0x6d02);
        let layer = gdn_layer(&model);
        let x = input(&device);

        let h = layer.input_layernorm.forward(&client, &x).expect("norm");
        let h = layer.block.forward_train(&client, &h, None).expect("mixer");
        let mid = var_add(&h, &x, &client).expect("residual");
        let h = layer
            .post_attention_layernorm
            .forward(&client, &mid)
            .expect("norm");
        let (h, _, _) = layer.ffn.forward(&client, &h).expect("ffn");
        let want = var_add(&h, &mid, &client).expect("residual");

        let got = layer.forward(&client, &x, None).expect("layer forward");
        assert_eq!(bits(got.hidden.tensor()), bits(want.tensor()));
        assert!(got.aux_loss.is_none() && got.z_loss.is_none());
    }

    /// The alias keeps every id and flag, frozen LoRA bases included, and
    /// gives the same output bits.
    #[test]
    fn the_alias_keeps_every_id_flag_and_output() {
        let (device, client) = setup();
        let model = build(&cfg(Some(lora(true))), 0x6d07);
        let layer = gdn_layer(&model);
        let alias = layer.alias();
        let flags = |l: &GdnLayer<CpuRuntime>| -> Vec<(TensorId, bool)> {
            l.block
                .named_parameters()
                .into_iter()
                .map(|(_, var)| (var.id(), var.requires_grad()))
                .chain(
                    l.trainable_parameters()
                        .into_iter()
                        .map(|(id, var)| (id, var.requires_grad())),
                )
                .collect()
        };
        assert_eq!(flags(&alias), flags(layer));
        assert!(flags(layer).iter().any(|(_, trains)| !trains));
        let x = input(&device);
        let got = alias.forward(&client, &x, None).expect("alias forward");
        let want = layer.forward(&client, &x, None).expect("layer forward");
        assert_eq!(bits(got.hidden.tensor()), bits(want.hidden.tensor()));
    }

    /// The per-step rebuild keeps every id, name and output.
    #[test]
    fn a_rebuild_keeps_every_id_name_and_output() {
        let model = build(&cfg(None), 0x6d03);
        let rebuilt = model
            .rebuild_with_parameters(&trainable_map(&model))
            .expect("rebuild");
        let ids = |m: &Model| -> Vec<TensorId> {
            m.trainable_parameters()
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        };
        assert_eq!(ids(&rebuilt), ids(&model));
        assert_eq!(names(&rebuilt), names(&model));
        assert_eq!(logits(&rebuilt), logits(&model));
        for name in [
            "layers.0.input_layernorm.weight",
            "layers.0.linear_attn.in_proj_qkv.weight",
            "layers.0.linear_attn.alpha_proj.weight",
            "layers.0.linear_attn.beta_proj.weight",
            "layers.0.linear_attn.conv1d.weight",
            "layers.0.linear_attn.A_log",
            "layers.0.linear_attn.dt_bias",
            "layers.0.linear_attn.norm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.0.mlp.gate_proj.weight",
        ] {
            assert!(trainable_names(&model).iter().any(|n| n == name), "{name}");
        }
    }

    /// `freeze_base: true` trains the mixer adapters alone, the gradient
    /// reaches every `lora_b`, and the rebuild keeps the set.
    #[test]
    fn mixer_adapters_train_alone_under_freeze_base() {
        let (device, client) = setup();
        let model = build(&cfg(Some(lora(true))), 0x6d04);
        let mut want: Vec<String> = GDN_TARGETS
            .iter()
            .flat_map(|t| ["lora_a", "lora_b"].map(|f| format!("layers.0.linear_attn.{t}.{f}")))
            .collect();
        want.sort();
        let mut got = trainable_names(&model);
        got.sort();
        assert_eq!(got, want);
        assert!(names(&model)
            .iter()
            .any(|n| n == "layers.0.linear_attn.in_proj_qkv.weight"));

        let rebuilt = model
            .rebuild_with_parameters(&trainable_map(&model))
            .expect("rebuild");
        let ids = |m: &Model| -> Vec<TensorId> {
            m.trainable_parameters()
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        };
        assert_eq!(ids(&rebuilt), ids(&model));

        let output = model
            .forward(&client, &token_ids(&device))
            .expect("forward");
        let loss = weighted_loss(&client, &output.logits);
        let grads = backward(&loss, &client).expect("backward");
        for (name, id, _) in model.named_trainable_parameters().expect("named") {
            if name.ends_with(".lora_b") {
                let grad = grads
                    .get(id)
                    .unwrap_or_else(|| panic!("{name}: no gradient"));
                let values: Vec<f32> = grad.contiguous().expect("contiguous").to_vec();
                assert!(values.iter().any(|v| *v != 0.0), "{name}: zero gradient");
            }
        }
    }

    /// `freeze_base: false` freezes only the adapted bases.
    #[test]
    fn freeze_base_off_freezes_only_the_adapted_mixer_bases() {
        let model = build(&cfg(Some(lora(false))), 0x6d05);
        let trainable = trainable_names(&model);
        for target in GDN_TARGETS {
            let base = format!("layers.0.linear_attn.{target}.weight");
            assert!(!trainable.contains(&base), "{base} must be frozen");
        }
        for name in ["layers.0.linear_attn.A_log", "layers.0.mlp.up_proj.weight"] {
            assert!(trainable.iter().any(|n| n == name), "{name} must train");
        }
    }

    /// A merge folds every mixer adapter, keeps the output, and leaves the
    /// plain parameter names.
    #[test]
    fn merging_the_mixer_adapters_keeps_the_output() {
        let (device, client) = setup();
        let model = build(&cfg(Some(lora(true))), 0x6d06);
        let mut params = trainable_map(&model);
        for (name, id, var) in model.named_trainable_parameters().expect("named") {
            if name.ends_with(".lora_b") {
                let t = var.tensor();
                let filled =
                    Tensor::<CpuRuntime>::from_slice(&values(t.numel(), 3), t.shape(), &device)
                        .expect("filled lora_b");
                params.insert(id, filled);
            }
        }
        let mut model = model.rebuild_with_parameters(&params).expect("rebuild");
        let before = model
            .forward(&client, &token_ids(&device))
            .expect("forward")
            .logits;
        let merged = model.merge_lora_adapters(&client).expect("merge");
        assert_eq!(merged, GDN_TARGETS.len());
        assert!(names(&model).iter().all(|n| !n.contains(".lora_")));
        let after = model
            .forward(&client, &token_ids(&device))
            .expect("forward")
            .logits;
        let before: Vec<f32> = before.tensor().contiguous().expect("contiguous").to_vec();
        let after: Vec<f32> = after.tensor().contiguous().expect("contiguous").to_vec();
        let worst = before
            .iter()
            .zip(&after)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "merge moved the logits by {worst}");
    }
}
