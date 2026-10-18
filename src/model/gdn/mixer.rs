//! The GDN mixer: boostr's [`GdnBlock`], built from tensors with their
//! stable ids.
//!
//! Every construction goes through [`GdnBlock::with_ids`]. Its gate pair
//! stays separate (`alpha_proj`, `beta_proj`), and its decay is `A_log`. A
//! projection with a LoRA adapter reports `<name>.base.weight`, `lora_a` and
//! `lora_b`. A plain one reports `<name>.weight`.

use crate::config::{LoraConfig, GDN_LORA_TARGETS};
use crate::model::{
    take_or_init_tensor, AttnProjWithIds, CausalSelfAttention, FrozenParams, LitGPT, LoraWithIds,
    MaybeLora, WeightWithId, INITIALIZED_TRAINABLE,
};
use crate::prelude::{DType, Init, Result, Tensor, VarBuilder};
use boostr::autograd::{TensorId, Var};
use boostr::model::config::GdnConfig;
use boostr::model::hybrid::{GdnBlock, GdnWeightsWithIds};
use std::collections::HashMap;

/// The mixer's projections, in [`GdnBlock::named_parameters`] order. These
/// are also the LoRA target names.
pub(in crate::model) const PROJECTIONS: [&str; 5] = GDN_LORA_TARGETS;

/// `[out_features, in_features]` of each projection, in [`PROJECTIONS`] order.
fn projection_shapes(cfg: &GdnConfig) -> [[usize; 2]; 5] {
    let (hidden, heads) = (cfg.hidden_size, cfg.value_heads);
    [
        [cfg.qkv_dim(), hidden],
        [cfg.value_dim(), hidden],
        [heads, hidden],
        [heads, hidden],
        [hidden, cfg.value_dim()],
    ]
}

/// Every mixer tensor with its stable id. A projection carries its adapter
/// factors when it has one.
pub(in crate::model) struct MixerWithIds<R: boostr::Runtime> {
    /// In [`PROJECTIONS`] order.
    pub(in crate::model) projections: [AttnProjWithIds<R>; 5],
    /// `conv1d.weight`, `[qkv_dim, 1, conv_kernel]`.
    pub(in crate::model) conv1d: WeightWithId<R>,
    /// `A_log`, `[value_heads]`.
    pub(in crate::model) a_log: WeightWithId<R>,
    /// `dt_bias`, `[value_heads]`.
    pub(in crate::model) dt_bias: WeightWithId<R>,
    /// `norm.weight`, `[state_size]`.
    pub(in crate::model) norm: WeightWithId<R>,
}

/// [`build_with`] that keeps each projection as built.
///
/// # Errors
///
/// As [`build_with`].
pub(in crate::model) fn build<R, C>(
    client: &C,
    cfg: &GdnConfig,
    weights: MixerWithIds<R>,
    prefix: &str,
    frozen: &FrozenParams,
) -> Result<GdnBlock<R>>
where
    R: boostr::Runtime<DType = DType>,
    C: boostr::UnaryOps<R>,
{
    build_with(client, cfg, weights, prefix, frozen, Ok)
}

/// The mixer over `weights`, keeping every id. `frozen` decides each flag.
/// `finish` takes each projection before the mixer does, in
/// [`PROJECTIONS`] order. A merge folds the adapter there.
///
/// `prefix` (`layers.N.linear_attn`) names the layer in errors.
///
/// # Errors
///
/// - A shape that disagrees with `cfg`, or a half-frozen adapter, as
///   `CausalSelfAttention::projection_with_ids` reports.
/// - A half-frozen raw group: boostr gives `conv1d.weight`, `A_log`,
///   `dt_bias` and `norm.weight` one flag.
/// - The errors of `finish` and of [`GdnBlock::with_ids`].
pub(in crate::model) fn build_with<R, C>(
    client: &C,
    cfg: &GdnConfig,
    weights: MixerWithIds<R>,
    prefix: &str,
    frozen: &FrozenParams,
    mut finish: impl FnMut(MaybeLora<R>) -> Result<MaybeLora<R>>,
) -> Result<GdnBlock<R>>
where
    R: boostr::Runtime<DType = DType>,
    C: boostr::UnaryOps<R>,
{
    let MixerWithIds {
        projections,
        conv1d,
        a_log,
        dt_bias,
        norm,
    } = weights;
    let [qkv, z, alpha, beta, out] = projections;
    let [qkv_shape, z_shape, alpha_shape, beta_shape, out_shape] = projection_shapes(cfg);
    let mut project = |index: usize, proj: AttnProjWithIds<R>, shape: [usize; 2]| {
        let name = format!("{prefix}.{}", PROJECTIONS[index]);
        let built = CausalSelfAttention::<R>::projection_with_ids(&name, proj, &shape, frozen)?;
        finish(built)
    };
    let attn_qkv = project(0, qkv, qkv_shape)?;
    let attn_gate = project(1, z, z_shape)?;
    let ssm_alpha = project(2, alpha, alpha_shape)?;
    let ssm_beta = project(3, beta, beta_shape)?;
    let ssm_out = project(4, out, out_shape)?;
    let raw_trainable = frozen.group_trainable(
        &format!("{prefix} raw parameters"),
        &[
            ("conv1d.weight", conv1d.1),
            ("A_log", a_log.1),
            ("dt_bias", dt_bias.1),
            ("norm.weight", norm.1),
        ],
    )?;
    let weights = GdnWeightsWithIds {
        attn_qkv,
        attn_gate,
        ssm_alpha,
        ssm_beta,
        ssm_out,
        conv1d,
        a_log,
        dt_bias,
        norm,
    };
    GdnBlock::with_ids(client, cfg.clone(), weights, raw_trainable)
}

/// A fresh mixer: boostr's reference init through [`GdnBlock::init`], then
/// a LoRA adapter on each projection `lora` targets.
///
/// `vb` is the mixer's builder (`layers.N.linear_attn`). A tensor it holds
/// is taken as is, adapter factors included. A missing factor starts as
/// `lora_a` Kaiming-uniform and `lora_b` zeros, as the attention adapters
/// do, so the adapted mixer starts at the base mixer's function.
///
/// # Errors
///
/// The errors of [`GdnBlock::init`], of the factor draws, and of [`build`].
pub(in crate::model) fn init<R, C>(
    cfg: &GdnConfig,
    vb: &mut VarBuilder<'_, R>,
    dtype: DType,
    client: &C,
    lora: Option<&LoraConfig>,
    prefix: &str,
) -> Result<GdnBlock<R>>
where
    R: boostr::Runtime<DType = DType>,
    C: boostr::RuntimeClient<R>
        + boostr::RandomOps<R>
        + boostr::ScalarOps<R>
        + boostr::BinaryOps<R>
        + boostr::CompareOps<R>
        + boostr::TensorOps<R>,
{
    let drawn = GdnBlock::init(cfg, vb, dtype, client, INITIALIZED_TRAINABLE)?;
    let Some(lora) = lora.filter(|lora| PROJECTIONS.iter().any(|name| lora.targets(name))) else {
        return Ok(drawn);
    };
    let mut weights = read(&drawn, None, prefix, current)?;
    let targets = PROJECTIONS
        .iter()
        .zip(weights.projections.iter_mut())
        .zip(projection_shapes(cfg));
    for ((name, proj), [out_features, in_features]) in targets {
        if !lora.targets(name) {
            continue;
        }
        let a_shape = [lora.rank, in_features];
        let b_shape = [out_features, lora.rank];
        let a_name = format!("{name}.lora_a");
        let b_name = format!("{name}.lora_b");
        let lora_a =
            take_or_init_tensor(vb, &a_name, &a_shape, dtype, Init::PyTorchLinear, client)?;
        let lora_b = take_or_init_tensor(vb, &b_name, &b_shape, dtype, Init::Zeros, client)?;
        let (a_id, b_id) = (lora_a.id(), lora_b.id());
        proj.lora = Some(LoraWithIds {
            a: (lora_a, a_id),
            b: (lora_b, b_id),
            alpha: lora.alpha,
        });
    }
    // INITIALIZED_TRAINABLE: no id is frozen at init. `LitGPT::new` freezes later.
    build(client, cfg, weights, prefix, &FrozenParams::new())
}

/// A variable's live tensor under its own id.
pub(in crate::model) fn current<R: boostr::Runtime>(var: &Var<R>) -> Result<WeightWithId<R>> {
    Ok((var.tensor().clone(), var.id()))
}

/// The live mixer's tensors with their ids, each trainable one read from
/// `params`, for the rebuild after an optimizer step.
///
/// `lora_alpha` is the config's `lora.alpha`. An adapted projection needs it.
///
/// # Errors
///
/// As [`read`], plus a trainable tensor absent from `params`.
pub(in crate::model) fn snapshot<R: boostr::Runtime<DType = DType>>(
    block: &GdnBlock<R>,
    params: &HashMap<TensorId, Tensor<R>>,
    lora_alpha: Option<f32>,
    prefix: &str,
) -> Result<MixerWithIds<R>> {
    read(block, lora_alpha, prefix, |var| {
        LitGPT::<R>::tensor_with_stable_id(var, params)
    })
}

/// Every mixer tensor through `value`, under its own id.
///
/// # Errors
///
/// - A tensor absent from [`GdnBlock::named_parameters`].
/// - An adapted projection while `lora_alpha` is `None`.
/// - The errors of `value`.
pub(in crate::model) fn read<R: boostr::Runtime<DType = DType>>(
    block: &GdnBlock<R>,
    lora_alpha: Option<f32>,
    prefix: &str,
    value: impl Fn(&Var<R>) -> Result<WeightWithId<R>>,
) -> Result<MixerWithIds<R>> {
    let named = block.named_parameters();
    let find = |name: &str| LitGPT::<R>::find_named_var(&named, name).and_then(&value);
    let projection = |name: &str| -> Result<AttnProjWithIds<R>> {
        if LitGPT::<R>::optional_named_var(&named, &format!("{name}.lora_a")).is_none() {
            return Ok(AttnProjWithIds {
                weight: find(&format!("{name}.weight"))?,
                lora: None,
            });
        }
        let alpha = lora_alpha.ok_or_else(|| boostr::error::Error::ModelError {
            reason: format!(
                "'{prefix}.{name}' carries a LoRA adapter but the model config has no lora \
                 block, so the adapter's alpha is unknown. Rebuild the model from the config \
                 that built it."
            ),
        })?;
        Ok(AttnProjWithIds {
            weight: find(&format!("{name}.base.weight"))?,
            lora: Some(LoraWithIds {
                a: find(&format!("{name}.lora_a"))?,
                b: find(&format!("{name}.lora_b"))?,
                alpha,
            }),
        })
    };
    Ok(MixerWithIds {
        projections: [
            projection(PROJECTIONS[0])?,
            projection(PROJECTIONS[1])?,
            projection(PROJECTIONS[2])?,
            projection(PROJECTIONS[3])?,
            projection(PROJECTIONS[4])?,
        ],
        conv1d: find("conv1d.weight")?,
        a_log: find("A_log")?,
        dt_bias: find("dt_bias")?,
        norm: find("norm.weight")?,
    })
}

/// The base-weight id of each adapted projection.
///
/// # Errors
///
/// An adapted projection without its `base.weight`.
pub(in crate::model) fn adapted_base_ids<R: boostr::Runtime<DType = DType>>(
    block: &GdnBlock<R>,
) -> Result<Vec<TensorId>> {
    let named = block.named_parameters();
    let mut ids = Vec::new();
    for name in PROJECTIONS {
        if LitGPT::<R>::optional_named_var(&named, &format!("{name}.lora_a")).is_some() {
            ids.push(LitGPT::<R>::find_named_var(&named, &format!("{name}.base.weight"))?.id());
        }
    }
    Ok(ids)
}

/// The checkpoint name of a [`GdnBlock::named_parameters`] name:
/// `<p>.base.weight` becomes `<p>.weight`, as every oxidizr adapted
/// projection names its base. Every other name is unchanged.
pub(in crate::model) fn checkpoint_name(name: &str) -> String {
    match name.strip_suffix(".base.weight") {
        Some(projection) => format!("{projection}.weight"),
        None => name.to_string(),
    }
}
