use crate::prelude::{DType, Embedding, Init, Linear, Result, RmsNorm, Tensor, VarBuilder};

use crate::config::{LoraConfig, OxidizrModelConfig};
use boostr::model::config::{default_rms_norm_eps, ModelConfig};
use boostr::tensor::TensorId;
// `cast` is a TypeConversionOps method; the trait must be in scope to call it.
use boostr::TypeConversionOps;
use std::collections::HashMap;

/// A weight tensor paired with the stable parameter id it was registered
/// under. Used throughout the `*_weights_with_ids`/`with_ids` rebuild path
/// so a checkpoint's tensors can be re-associated with boostr's `TensorId`
/// keys without re-running initialization.
type WeightWithId<R> = (Tensor<R>, boostr::autograd::TensorId);

/// Trainability for a parameter being initialized for the very first time.
///
/// The `new`/`take_*` helpers mint the [`boostr::autograd::TensorId`] in
/// place, so the id cannot yet appear in a [`FrozenParams`] set — nothing is
/// freezable at initialization. Freezing is expressed on the reconstruction
/// path instead, via [`LitGPT::with_frozen_parameters`], which routes through
/// the `*_with_ids` constructors where `FrozenParams` decides every flag.
const INITIALIZED_TRAINABLE: bool = true;

/// The set of parameters that must be reconstructed frozen.
///
/// This is the SINGLE source of truth for trainability on every
/// reconstruction path. `LitGPT` carries one of these and hands the same
/// instance to every `*_with_ids` constructor, so the flag a parameter is
/// built with on step 1 is the flag it is rebuilt with on step N. Deriving
/// trainability per-call-site from a literal is what let
/// `rebuild_with_parameters` silently un-freeze a frozen weight.
///
/// Keyed by [`boostr::autograd::TensorId`] because that is the identity the
/// optimizer, the gradient store, and `assert_stable_parameter_ids` all use.
/// A parameter's id is stable across rebuilds, so a set captured once stays
/// valid for the life of the model.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrozenParams {
    ids: std::collections::HashSet<boostr::autograd::TensorId>,
}

// The constructors below are the freeze API. `from_ids` has no caller outside
// tests, so the block keeps the allow.
#[allow(dead_code)]
impl FrozenParams {
    /// An empty set: every parameter is trainable. This is what
    /// `LitGPT::new` uses, so default behavior is unchanged.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_ids<I>(ids: I) -> Self
    where
        I: IntoIterator<Item = boostr::autograd::TensorId>,
    {
        Self {
            ids: ids.into_iter().collect(),
        }
    }

    pub fn freeze(&mut self, id: boostr::autograd::TensorId) {
        self.ids.insert(id);
    }

    pub fn is_frozen(&self, id: boostr::autograd::TensorId) -> bool {
        self.ids.contains(&id)
    }

    /// The trainability flag for `id`. Every construction site derives its
    /// `trainable` argument from this one call, never from a literal.
    pub fn trainable(&self, id: boostr::autograd::TensorId) -> bool {
        !self.is_frozen(id)
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Trainability for a group of parameters that share ONE flag.
    ///
    /// boostr's `Mamba1/2/3::with_ids` take a single `trainable` bool for all
    /// their raw (non-`Linear`) parameters, so those cannot be frozen
    /// individually. Freezing part of such a group is rejected rather than
    /// silently rounded up or down.
    fn group_trainable(
        &self,
        group: &str,
        members: &[(&str, boostr::autograd::TensorId)],
    ) -> Result<bool> {
        let frozen: Vec<&str> = members
            .iter()
            .filter(|(_, id)| self.is_frozen(*id))
            .map(|(name, _)| *name)
            .collect();
        if frozen.is_empty() {
            return Ok(true);
        }
        if frozen.len() == members.len() {
            return Ok(false);
        }
        let trainable: Vec<&str> = members
            .iter()
            .filter(|(_, id)| !self.is_frozen(*id))
            .map(|(name, _)| *name)
            .collect();
        Err(boostr::error::Error::ModelError {
            reason: format!(
                "'{group}' shares one trainability flag across all of its raw parameters, \
                 but only part of the group is frozen: frozen [{}], still trainable [{}]. \
                 Add the still-trainable ids to FrozenParams, or remove the frozen ones, \
                 so the whole group is frozen or none of it is.",
                frozen.join(", "),
                trainable.join(", ")
            ),
        })
    }
}

/// Take a tensor from the VarBuilder, or initialize it when absent.
///
/// Delegates to boostr's `VarBuilder::take_or_init_tensor` — the initializer
/// and its RNG belong to boostr so every trainer built on it initializes
/// identically. Re-implementing the `Init` strategies here would silently
/// diverge from the framework's own weight initialization.
fn take_or_init_tensor<R, C>(
    vb: &mut VarBuilder<'_, R>,
    name: &str,
    shape: &[usize],
    dtype: DType,
    init: Init,
    client: &C,
) -> Result<Tensor<R>>
where
    R: boostr::Runtime<DType = DType>,
    C: boostr::RuntimeClient<R>
        + boostr::RandomOps<R>
        + boostr::ScalarOps<R>
        + boostr::BinaryOps<R>
        + boostr::CompareOps<R>
        + boostr::TensorOps<R>,
{
    vb.take_or_init_tensor(name, shape, dtype, init, client)
}

// --- Rotary Positional Embeddings (RoPE) ---

struct RotaryEmbedding<R: boostr::Runtime> {
    cos: boostr::autograd::Var<R>,
    sin: boostr::autograd::Var<R>,
    _dim: usize,
}

impl<R> RotaryEmbedding<R>
where
    R: boostr::Runtime<DType = DType>,
{
    fn new(
        dim: usize,
        max_seq_len: usize,
        theta: f32,
        scaling: Option<&boostr::model::config::RopeScalingConfig>,
        device: &R::Device,
    ) -> Result<Self> {
        let rope =
            boostr::nn::RoPE::<R>::precompute_freqs(max_seq_len, dim, theta, scaling, device)?;
        Ok(Self {
            cos: boostr::autograd::Var::<R>::with_id(
                rope.cos_cache().tensor().clone(),
                rope.cos_cache().id(),
                false,
            ),
            sin: boostr::autograd::Var::<R>::with_id(
                rope.sin_cache().tensor().clone(),
                rope.sin_cache().id(),
                false,
            ),
            _dim: dim,
        })
    }

    fn with_ids(
        cos: Tensor<R>,
        cos_id: boostr::autograd::TensorId,
        sin: Tensor<R>,
        sin_id: boostr::autograd::TensorId,
        dim: usize,
    ) -> Self {
        Self {
            cos: boostr::autograd::Var::<R>::with_id(cos, cos_id, false),
            sin: boostr::autograd::Var::<R>::with_id(sin, sin_id, false),
            _dim: dim,
        }
    }

    fn apply_rotary_emb<C>(
        &self,
        client: &C,
        q: &boostr::autograd::Var<R>,
        k: &boostr::autograd::Var<R>,
    ) -> Result<(boostr::autograd::Var<R>, boostr::autograd::Var<R>)>
    where
        C: boostr::RuntimeClient<R> + boostr::RoPEOps<R>,
    {
        Ok((
            client.apply_rope(q, &self.cos, &self.sin)?,
            client.apply_rope(k, &self.cos, &self.sin)?,
        ))
    }

    fn trainable_parameters(&self) -> Vec<(boostr::autograd::TensorId, &boostr::autograd::Var<R>)> {
        Vec::new()
    }
}

// --- Multi-Head Latent Attention (MLA) with Decoupled RoPE ---
// Delegates DeepSeek-style MLA to boostr::nn::Mla so compressed KV,
// decoupled RoPE, Q/K nope/pe splits, latent RMSNorm, and stable parameter
// identity are owned by the shared boostr implementation.

struct MultiHeadLatentAttention<R: boostr::Runtime> {
    mla: boostr::nn::Mla<R>,
}

impl<R> MultiHeadLatentAttention<R>
where
    R: boostr::Runtime<DType = DType>,
{
    fn new<C>(
        cfg: &ModelConfig,
        mut vb: VarBuilder<'_, R>,
        client: &C,
        dtype: DType,
    ) -> Result<Self>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let mla_config = Self::mla_config(cfg)?;
        let weights = Self::take_weights(&mla_config, &mut vb, client, dtype)?;
        let mla = boostr::nn::Mla::with_ids(&mla_config, weights, vb.device())?;
        Ok(Self { mla })
    }

    #[allow(clippy::too_many_arguments)]
    fn with_ids(
        cfg: &ModelConfig,
        q_a_proj: (Tensor<R>, boostr::autograd::TensorId),
        q_b_proj: (Tensor<R>, boostr::autograd::TensorId),
        q_a_layernorm: (Tensor<R>, boostr::autograd::TensorId),
        kv_a_proj_with_mqa: (Tensor<R>, boostr::autograd::TensorId),
        kv_a_layernorm: (Tensor<R>, boostr::autograd::TensorId),
        kv_b_proj: (Tensor<R>, boostr::autograd::TensorId),
        o_proj: (Tensor<R>, boostr::autograd::TensorId),
        device: &R::Device,
        frozen: &FrozenParams,
    ) -> Result<Self> {
        let mla_config = Self::mla_config(cfg)?;
        let qk_dim = mla_config.qk_head_dim();

        let weights = boostr::nn::MlaWeights {
            q_down: Some(Self::linear_with_ids(
                "q_a_proj.weight",
                q_a_proj.0,
                q_a_proj.1,
                &[mla_config.q_lora_rank, mla_config.hidden_size],
                frozen,
            )?),
            q_up: Self::linear_with_ids(
                "q_b_proj.weight",
                q_b_proj.0,
                q_b_proj.1,
                &[mla_config.num_heads * qk_dim, mla_config.q_lora_rank],
                frozen,
            )?,
            q_norm: Some(Self::rms_norm_with_id(
                "q_a_layernorm.weight",
                q_a_layernorm.0,
                q_a_layernorm.1,
                mla_config.q_lora_rank,
                mla_config.norm_eps,
                frozen,
            )?),
            kv_compress: Self::linear_with_ids(
                "kv_a_proj_with_mqa.weight",
                kv_a_proj_with_mqa.0,
                kv_a_proj_with_mqa.1,
                &[
                    mla_config.kv_lora_rank + mla_config.rope_head_dim,
                    mla_config.hidden_size,
                ],
                frozen,
            )?,
            kv_norm: Some(Self::rms_norm_with_id(
                "kv_a_layernorm.weight",
                kv_a_layernorm.0,
                kv_a_layernorm.1,
                mla_config.kv_lora_rank,
                mla_config.norm_eps,
                frozen,
            )?),
            kv_decompress: Self::linear_with_ids(
                "kv_b_proj.weight",
                kv_b_proj.0,
                kv_b_proj.1,
                &[
                    mla_config.num_heads * (mla_config.head_dim + mla_config.head_dim_v),
                    mla_config.kv_lora_rank,
                ],
                frozen,
            )?,
            o_proj: Self::linear_with_ids(
                "o_proj.weight",
                o_proj.0,
                o_proj.1,
                &[
                    mla_config.hidden_size,
                    mla_config.num_heads * mla_config.head_dim_v,
                ],
                frozen,
            )?,
        };

        let mla = boostr::nn::Mla::with_ids(&mla_config, weights, device)?;
        Ok(Self { mla })
    }

    fn forward<C>(
        &self,
        client: &C,
        x: &boostr::autograd::Var<R>,
    ) -> Result<boostr::autograd::Var<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::QuantMatmulOps<R>
            + boostr::RoPEOps<R>
            + boostr::TypeConversionOps<R>,
        R::Client: boostr::TensorOps<R> + boostr::ScalarOps<R>,
    {
        self.mla.forward(client, x)
    }

    fn trainable_parameters(&self) -> Vec<(boostr::autograd::TensorId, &boostr::autograd::Var<R>)> {
        self.mla.trainable_parameters()
    }

    fn mla_config(cfg: &ModelConfig) -> Result<boostr::nn::MlaConfig> {
        let attn = cfg
            .attention
            .as_ref()
            .ok_or_else(|| boostr::error::Error::ModelError {
                reason: "attention config missing while building MLA config".into(),
            })?;
        let head_dim = Self::head_dim(cfg)?;
        let kv_latent_dim = Self::required_dim("kv_latent_dim", attn.kv_latent_dim)?;
        let q_latent_dim = Self::required_dim("q_latent_dim", attn.q_latent_dim)?;
        let d_rope = Self::required_dim("d_rope", attn.d_rope)?;

        let config = boostr::nn::MlaConfig {
            hidden_size: cfg.hidden_size,
            num_heads: attn.num_heads,
            head_dim,
            head_dim_v: head_dim,
            kv_lora_rank: kv_latent_dim,
            q_lora_rank: q_latent_dim,
            rope_head_dim: d_rope,
            max_seq_len: cfg.max_seq_len,
            rope_theta: attn.rope_theta,
            use_norm: true,
            norm_eps: 1e-6,
        };
        config.validate()?;
        Ok(config)
    }

    fn take_weights<C>(
        config: &boostr::nn::MlaConfig,
        vb: &mut VarBuilder<'_, R>,
        client: &C,
        dtype: DType,
    ) -> Result<boostr::nn::MlaWeights<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let qk_dim = config.qk_head_dim();

        Ok(boostr::nn::MlaWeights {
            q_down: Some(Self::take_linear_no_bias(
                vb.pp("q_a_proj"),
                &[config.q_lora_rank, config.hidden_size],
                dtype,
                client,
            )?),
            q_up: Self::take_linear_no_bias(
                vb.pp("q_b_proj"),
                &[config.num_heads * qk_dim, config.q_lora_rank],
                dtype,
                client,
            )?,
            q_norm: Some(Self::take_rms_norm(
                vb.pp("q_a_layernorm"),
                config.q_lora_rank,
                config.norm_eps,
                dtype,
                client,
            )?),
            kv_compress: Self::take_linear_no_bias(
                vb.pp("kv_a_proj_with_mqa"),
                &[
                    config.kv_lora_rank + config.rope_head_dim,
                    config.hidden_size,
                ],
                dtype,
                client,
            )?,
            kv_norm: Some(Self::take_rms_norm(
                vb.pp("kv_a_layernorm"),
                config.kv_lora_rank,
                config.norm_eps,
                dtype,
                client,
            )?),
            kv_decompress: Self::take_linear_no_bias(
                vb.pp("kv_b_proj"),
                &[
                    config.num_heads * (config.head_dim + config.head_dim_v),
                    config.kv_lora_rank,
                ],
                dtype,
                client,
            )?,
            o_proj: Self::take_linear_no_bias(
                vb.pp("o_proj"),
                &[config.hidden_size, config.num_heads * config.head_dim_v],
                dtype,
                client,
            )?,
        })
    }

    fn take_linear_no_bias<C>(
        mut vb: VarBuilder<'_, R>,
        expected_shape: &[usize],
        dtype: DType,
        client: &C,
    ) -> Result<boostr::nn::MaybeQuantLinear<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let param_name = Self::param_name(&vb, "weight");
        let weight = take_or_init_tensor(
            &mut vb,
            "weight",
            expected_shape,
            dtype,
            Init::PyTorchLinear,
            client,
        )?;
        let weight_id = weight.id();
        Self::ensure_shape(&param_name, &weight, expected_shape)?;
        Ok(boostr::nn::MaybeQuantLinear::Standard(
            boostr::nn::Linear::with_ids(weight, weight_id, None, INITIALIZED_TRAINABLE),
        ))
    }

    fn take_rms_norm<C>(
        mut vb: VarBuilder<'_, R>,
        expected_len: usize,
        eps: f32,
        dtype: DType,
        client: &C,
    ) -> Result<boostr::nn::RmsNorm<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let param_name = Self::param_name(&vb, "weight");
        let weight = take_or_init_tensor(
            &mut vb,
            "weight",
            &[expected_len],
            dtype,
            Init::Ones,
            client,
        )?;
        let weight_id = weight.id();
        Self::ensure_shape(&param_name, &weight, &[expected_len])?;
        Ok(boostr::nn::RmsNorm::with_id(
            weight,
            weight_id,
            eps,
            INITIALIZED_TRAINABLE,
        ))
    }

    fn linear_with_ids(
        name: &str,
        weight: Tensor<R>,
        weight_id: boostr::autograd::TensorId,
        expected_shape: &[usize],
        frozen: &FrozenParams,
    ) -> Result<boostr::nn::MaybeQuantLinear<R>> {
        Self::ensure_shape(name, &weight, expected_shape)?;
        let trainable = frozen.trainable(weight_id);
        Ok(boostr::nn::MaybeQuantLinear::Standard(
            boostr::nn::Linear::with_ids(weight, weight_id, None, trainable),
        ))
    }

    fn rms_norm_with_id(
        name: &str,
        weight: Tensor<R>,
        weight_id: boostr::autograd::TensorId,
        expected_len: usize,
        eps: f32,
        frozen: &FrozenParams,
    ) -> Result<boostr::nn::RmsNorm<R>> {
        Self::ensure_shape(name, &weight, &[expected_len])?;
        let trainable = frozen.trainable(weight_id);
        Ok(boostr::nn::RmsNorm::with_id(
            weight, weight_id, eps, trainable,
        ))
    }

    fn head_dim(cfg: &ModelConfig) -> Result<usize> {
        let attn = cfg
            .attention
            .as_ref()
            .ok_or_else(|| boostr::error::Error::ModelError {
                reason: "attention config missing while computing MLA head_dim".into(),
            })?;
        if attn.num_heads == 0 {
            return Err(boostr::error::Error::ModelError {
                reason: "MLA num_heads must be non-zero".into(),
            });
        }
        if !cfg.hidden_size.is_multiple_of(attn.num_heads) {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "MLA hidden_size ({}) must be divisible by num_heads ({})",
                    cfg.hidden_size, attn.num_heads
                ),
            });
        }
        Ok(cfg.hidden_size / attn.num_heads)
    }

    fn required_dim(name: &str, value: Option<usize>) -> Result<usize> {
        let dim = value.ok_or_else(|| boostr::error::Error::ModelError {
            reason: format!("MLA requires {name}"),
        })?;
        if dim == 0 {
            return Err(boostr::error::Error::ModelError {
                reason: format!("MLA {name} must be non-zero"),
            });
        }
        Ok(dim)
    }

    fn ensure_shape(name: &str, weight: &Tensor<R>, expected_shape: &[usize]) -> Result<()> {
        if weight.shape() != expected_shape {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "shape mismatch for '{name}': expected {:?}, got {:?}",
                    expected_shape,
                    weight.shape()
                ),
            });
        }
        Ok(())
    }

    fn param_name(vb: &VarBuilder<'_, R>, leaf: &str) -> String {
        if vb.prefix().is_empty() {
            leaf.to_string()
        } else {
            format!("{}.{}", vb.prefix(), leaf)
        }
    }
}

// --- Optionally LoRA-adapted projection ---

/// A projection that is either a plain `Linear` or that same `Linear` wrapped
/// in a LoRA adapter. Used by both the attention block and the MLP.
///
/// `LoraLinear::forward` is signature-compatible with `Linear::forward`, so the
/// forward path treats the two identically. Everything else on this type exists
/// so an adapted projection enumerates, names, and rebuilds through the SAME
/// paths a plain one does — a LoRA parameter that skips those paths is a
/// parameter the optimizer, the checkpoint, and `assert_stable_parameter_ids`
/// never see.
enum MaybeLora<R: boostr::Runtime> {
    Plain(boostr::nn::Linear<R>),
    Lora(boostr::nn::LoraLinear<R>),
}

/// One projection's tensors paired with their stable parameter ids, for the
/// `*_with_ids` rebuild path. The adapter pair is present exactly when the
/// projection being rebuilt carries one.
///
/// Shared by the attention projections and the MLP's `gate_proj`/`up_proj`/
/// `down_proj`: both rebuild through the same `projection_with_ids`.
struct AttnProjWithIds<R: boostr::Runtime> {
    weight: WeightWithId<R>,
    lora: Option<LoraWithIds<R>>,
}

/// A LoRA adapter's factors with their stable parameter ids, plus the `alpha`
/// the rebuilt layer must re-derive its scaling from.
struct LoraWithIds<R: boostr::Runtime> {
    a: WeightWithId<R>,
    b: WeightWithId<R>,
    alpha: f32,
}

impl<R> MaybeLora<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// The base projection weight, under either variant. This is the
    /// `<prefix>.weight` every existing caller already expects.
    fn weight(&self) -> &boostr::autograd::Var<R> {
        match self {
            MaybeLora::Plain(linear) => linear.weight(),
            MaybeLora::Lora(lora) => lora.base().weight(),
        }
    }

    /// `(lora_a, lora_b)` when this projection carries an adapter.
    fn adapters(&self) -> Option<(&boostr::autograd::Var<R>, &boostr::autograd::Var<R>)> {
        match self {
            MaybeLora::Plain(_) => None,
            MaybeLora::Lora(lora) => Some((lora.lora_a(), lora.lora_b())),
        }
    }

    fn forward<C>(
        &self,
        client: &C,
        input: &boostr::autograd::Var<R>,
    ) -> Result<boostr::autograd::Var<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::BinaryOps<R>
            + boostr::ScalarOps<R>,
        R::Client: boostr::TensorOps<R> + boostr::BinaryOps<R> + boostr::ScalarOps<R>,
    {
        match self {
            MaybeLora::Plain(linear) => linear.forward(client, input),
            MaybeLora::Lora(lora) => lora.forward(client, input),
        }
    }

    /// Trainable parameters in `named_parameters` order: base weight first (it
    /// is absent once frozen), then `lora_a`, then `lora_b`.
    fn trainable_parameters(&self) -> Vec<(boostr::autograd::TensorId, &boostr::autograd::Var<R>)> {
        match self {
            MaybeLora::Plain(linear) => linear.trainable_parameters(),
            MaybeLora::Lora(lora) => {
                <boostr::nn::LoraLinear<R> as boostr::nn::Module<R>>::trainable_parameters(lora)
            }
        }
    }
}

// --- Causal Self Attention with Grouped Query Attention ---

struct CausalSelfAttention<R: boostr::Runtime> {
    q_proj: MaybeLora<R>,
    k_proj: MaybeLora<R>,
    v_proj: MaybeLora<R>,
    o_proj: MaybeLora<R>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: RotaryEmbedding<R>,
    /// Sliding-window attention span. `0` disables it (boostr's kernel
    /// sentinel for "no window").
    sliding_window: usize,
}

impl<R> CausalSelfAttention<R>
where
    R: boostr::Runtime<DType = DType>,
{
    fn new<C>(
        cfg: &ModelConfig,
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
        let attn_cfg = cfg
            .attention
            .as_ref()
            .ok_or_else(|| boostr::error::Error::ModelError {
                reason: "attention config missing while building legacy causal self-attention"
                    .into(),
            })?;
        let head_dim = Self::head_dim(cfg)?;
        let kv_heads = attn_cfg.num_kv_heads.unwrap_or(attn_cfg.num_heads);
        Self::validate_heads(attn_cfg.num_heads, kv_heads)?;
        let kv_hidden = kv_heads * head_dim;

        let q_proj = Self::projection(
            vb.pp("q_proj"),
            "q_proj",
            &[cfg.hidden_size, cfg.hidden_size],
            dtype,
            lora,
            client,
        )?;
        let k_proj = Self::projection(
            vb.pp("k_proj"),
            "k_proj",
            &[kv_hidden, cfg.hidden_size],
            dtype,
            lora,
            client,
        )?;
        let v_proj = Self::projection(
            vb.pp("v_proj"),
            "v_proj",
            &[kv_hidden, cfg.hidden_size],
            dtype,
            lora,
            client,
        )?;
        let o_proj = Self::projection(
            vb.pp("o_proj"),
            "o_proj",
            &[cfg.hidden_size, cfg.hidden_size],
            dtype,
            lora,
            client,
        )?;
        let rope = RotaryEmbedding::new(
            head_dim,
            cfg.max_seq_len,
            attn_cfg.rope_theta,
            attn_cfg.rope_scaling.as_ref(),
            vb.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: attn_cfg.num_heads,
            num_kv_heads: kv_heads,
            head_dim,
            rope,
            sliding_window: attn_cfg.sliding_window(),
        })
    }

    fn with_ids(
        cfg: &ModelConfig,
        q_proj: AttnProjWithIds<R>,
        k_proj: AttnProjWithIds<R>,
        v_proj: AttnProjWithIds<R>,
        o_proj: AttnProjWithIds<R>,
        rope_cos: (Tensor<R>, boostr::autograd::TensorId),
        rope_sin: (Tensor<R>, boostr::autograd::TensorId),
        frozen: &FrozenParams,
    ) -> Result<Self> {
        let attn_cfg = cfg
            .attention
            .as_ref()
            .ok_or_else(|| boostr::error::Error::ModelError {
                reason: "attention config missing while rebuilding legacy causal self-attention"
                    .into(),
            })?;
        let head_dim = Self::head_dim(cfg)?;
        let kv_heads = attn_cfg.num_kv_heads.unwrap_or(attn_cfg.num_heads);
        Self::validate_heads(attn_cfg.num_heads, kv_heads)?;
        let kv_hidden = kv_heads * head_dim;

        Ok(Self {
            q_proj: Self::projection_with_ids(
                "q_proj",
                q_proj,
                &[cfg.hidden_size, cfg.hidden_size],
                frozen,
            )?,
            k_proj: Self::projection_with_ids(
                "k_proj",
                k_proj,
                &[kv_hidden, cfg.hidden_size],
                frozen,
            )?,
            v_proj: Self::projection_with_ids(
                "v_proj",
                v_proj,
                &[kv_hidden, cfg.hidden_size],
                frozen,
            )?,
            o_proj: Self::projection_with_ids(
                "o_proj",
                o_proj,
                &[cfg.hidden_size, cfg.hidden_size],
                frozen,
            )?,
            num_heads: attn_cfg.num_heads,
            num_kv_heads: kv_heads,
            head_dim,
            rope: RotaryEmbedding::with_ids(
                rope_cos.0, rope_cos.1, rope_sin.0, rope_sin.1, head_dim,
            ),
            sliding_window: attn_cfg.sliding_window(),
        })
    }

    fn forward<C>(
        &self,
        client: &C,
        x: &boostr::autograd::Var<R>,
    ) -> Result<boostr::autograd::Var<R>>
    where
        // `BinaryOps`/`ScalarOps` are `LoraLinear::forward`'s bounds: a
        // LoRA-adapted projection scales and adds its low-rank branch through
        // `var_mul_scalar`/`var_add`. A plain `Linear` needs neither, but the
        // variant is a runtime choice, so the bounds apply either way.
        C: boostr::RuntimeClient<R>
            + boostr::RoPEOps<R>
            + boostr::TensorOps<R>
            + boostr::BinaryOps<R>
            + boostr::ScalarOps<R>,
        R::Client: boostr::FlashAttentionOps<R>
            + boostr::TensorOps<R>
            + boostr::BinaryOps<R>
            + boostr::ScalarOps<R>,
    {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(boostr::error::Error::ModelError {
                reason: format!("attention input must be [batch, seq, hidden], got {shape:?}"),
            });
        }
        let (b_sz, seq_len) = (shape[0], shape[1]);

        // Project to Q, K, V using boostr's Var-based Linear primitive.
        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        // Q: [B, Seq, H, D] -> [B, H, Seq, D]
        let q = boostr::autograd::var_reshape(&q, &[b_sz, seq_len, self.num_heads, self.head_dim])?;
        let q = boostr::autograd::var_permute(&q, &[0, 2, 1, 3])?;
        let q = boostr::nn::var_contiguous(&q)?;

        // K, V: [B, Seq, KV_H, D] -> [B, KV_H, Seq, D]
        let k =
            boostr::autograd::var_reshape(&k, &[b_sz, seq_len, self.num_kv_heads, self.head_dim])?;
        let k = boostr::autograd::var_permute(&k, &[0, 2, 1, 3])?;
        let k = boostr::nn::var_contiguous(&k)?;

        let v =
            boostr::autograd::var_reshape(&v, &[b_sz, seq_len, self.num_kv_heads, self.head_dim])?;
        let v = boostr::autograd::var_permute(&v, &[0, 2, 1, 3])?;
        let v = boostr::nn::var_contiguous(&v)?;

        // Apply boostr RoPE primitive, then boostr FlashAttention primitive.
        let (q, k) = self.rope.apply_rotary_emb(client, &q, &k)?;
        let output = boostr::var_flash_attention(
            &q,
            &k,
            &v,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            true,
            self.sliding_window,
        )?;

        // [B, H, Seq, D] -> [B, Seq, H, D] -> [B, Seq, Hidden]
        let output = boostr::autograd::var_permute(&output, &[0, 2, 1, 3])?;
        let output = boostr::nn::var_contiguous(&output)?;
        let output = boostr::autograd::var_reshape(
            &output,
            &[b_sz, seq_len, self.num_heads * self.head_dim],
        )?;

        self.o_proj.forward(client, &output)
    }

    fn trainable_parameters(&self) -> Vec<(boostr::autograd::TensorId, &boostr::autograd::Var<R>)> {
        let mut params = Vec::new();
        params.extend(self.q_proj.trainable_parameters());
        params.extend(self.k_proj.trainable_parameters());
        params.extend(self.v_proj.trainable_parameters());
        params.extend(self.o_proj.trainable_parameters());
        params.extend(self.rope.trainable_parameters());
        params
    }

    fn head_dim(cfg: &ModelConfig) -> Result<usize> {
        let attn = cfg
            .attention
            .as_ref()
            .ok_or_else(|| boostr::error::Error::ModelError {
                reason: "attention config missing while computing causal self-attention head_dim"
                    .into(),
            })?;
        if attn.num_heads == 0 {
            return Err(boostr::error::Error::ModelError {
                reason: "attention num_heads must be non-zero".into(),
            });
        }
        if !cfg.hidden_size.is_multiple_of(attn.num_heads) {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "hidden_size ({}) must be divisible by num_heads ({})",
                    cfg.hidden_size, attn.num_heads
                ),
            });
        }
        Ok(cfg.hidden_size / attn.num_heads)
    }

    fn validate_heads(num_heads: usize, num_kv_heads: usize) -> Result<()> {
        if num_kv_heads == 0 {
            return Err(boostr::error::Error::ModelError {
                reason: "attention num_key_value_heads must be non-zero".into(),
            });
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "num_heads ({num_heads}) must be divisible by num_key_value_heads ({num_kv_heads})"
                ),
            });
        }
        Ok(())
    }

    /// Build one no-bias projection, wrapping it in a LoRA adapter when `lora`
    /// targets `name`.
    ///
    /// The adapter factors are taken from the same `VarBuilder` as every other
    /// parameter, so they carry the model's `dtype` and are addressable as
    /// `<prefix>.lora_a` / `<prefix>.lora_b`. `LoraLinear::new` would instead
    /// allocate them F32 from its own PRNG, which mismatches a BF16/F16 model
    /// and hides the factors from the checkpoint's name space.
    ///
    /// Both the base and the adapters come back TRAINABLE here. Freezing the
    /// base is `LitGPT::new`'s job, via `FrozenParams` — see
    /// `INITIALIZED_TRAINABLE`.
    fn projection<C>(
        mut vb: VarBuilder<'_, R>,
        name: &str,
        expected_shape: &[usize],
        dtype: DType,
        lora: Option<&LoraConfig>,
        client: &C,
    ) -> Result<MaybeLora<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let prefix = vb.prefix().to_string();
        let param_name = if prefix.is_empty() {
            "weight".to_string()
        } else {
            format!("{prefix}.weight")
        };
        let weight = take_or_init_tensor(
            &mut vb,
            "weight",
            expected_shape,
            dtype,
            Init::PyTorchLinear,
            client,
        )?;
        Self::ensure_shape(&param_name, &weight, expected_shape)?;
        let weight_id = weight.id();
        let base = boostr::nn::Linear::with_ids(weight, weight_id, None, INITIALIZED_TRAINABLE);

        let Some(lora) = lora.filter(|lora| lora.targets(name)) else {
            return Ok(MaybeLora::Plain(base));
        };
        let [out_features, in_features] = expected_shape else {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "'{param_name}' has shape {expected_shape:?}, but a LoRA adapter needs a \
                     2-D [out_features, in_features] projection to size its factors. Remove \
                     '{name}' from lora.target_modules."
                ),
            });
        };
        // A is the standard Kaiming-uniform down-projection; B is zeros, so
        // the adapter contributes nothing at step 0 and the wrapped model
        // starts exactly where the base model left off.
        let lora_a = take_or_init_tensor(
            &mut vb,
            "lora_a",
            &[lora.rank, *in_features],
            dtype,
            Init::PyTorchLinear,
            client,
        )?;
        let lora_b = take_or_init_tensor(
            &mut vb,
            "lora_b",
            &[*out_features, lora.rank],
            dtype,
            Init::Zeros,
            client,
        )?;
        let (lora_a_id, lora_b_id) = (lora_a.id(), lora_b.id());
        Ok(MaybeLora::Lora(boostr::nn::LoraLinear::with_ids(
            base,
            lora_a,
            lora_a_id,
            lora_b,
            lora_b_id,
            lora.alpha,
            INITIALIZED_TRAINABLE,
        )))
    }

    /// Rebuild one projection from optimizer-updated tensors, preserving every
    /// stable id and the adapter's rank/alpha.
    ///
    /// The adapter factors go through `LoraLinear::with_ids`, never
    /// `from_weights`: `from_weights` derives each id from the tensor it is
    /// handed, and `tensor_with_stable_id` hands over a CLONE, whose id is
    /// fresh. That would give every adapter a new `TensorId` each step,
    /// detaching the optimizer state and tripping
    /// `assert_stable_parameter_ids`.
    fn projection_with_ids(
        name: &str,
        proj: AttnProjWithIds<R>,
        expected_shape: &[usize],
        frozen: &FrozenParams,
    ) -> Result<MaybeLora<R>> {
        let AttnProjWithIds { weight, lora } = proj;
        let weight_name = format!("{name}.weight");
        Self::ensure_shape(&weight_name, &weight.0, expected_shape)?;
        let base =
            boostr::nn::Linear::with_ids(weight.0, weight.1, None, frozen.trainable(weight.1));

        let Some(lora) = lora else {
            return Ok(MaybeLora::Plain(base));
        };
        let [out_features, in_features] = expected_shape else {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "'{weight_name}' has shape {expected_shape:?}, but its LoRA adapter needs \
                     a 2-D [out_features, in_features] projection to check its factors against."
                ),
            });
        };
        let a_name = format!("{name}.lora_a");
        let b_name = format!("{name}.lora_b");
        let rank = match lora.a.0.shape() {
            [rank, a_in] if a_in == in_features => *rank,
            other => {
                return Err(boostr::error::Error::ModelError {
                    reason: format!(
                        "shape mismatch for '{a_name}': expected [rank, {in_features}], got \
                         {other:?}"
                    ),
                })
            }
        };
        Self::ensure_shape(&b_name, &lora.b.0, &[*out_features, rank])?;
        // boostr's `LoraLinear` takes ONE trainability flag for the factor
        // pair, so a half-frozen adapter is rejected rather than silently
        // rounded to trainable or frozen.
        let trainable = frozen.group_trainable(
            name,
            &[(a_name.as_str(), lora.a.1), (b_name.as_str(), lora.b.1)],
        )?;
        Ok(MaybeLora::Lora(boostr::nn::LoraLinear::with_ids(
            base, lora.a.0, lora.a.1, lora.b.0, lora.b.1, lora.alpha, trainable,
        )))
    }

    fn ensure_shape(name: &str, weight: &Tensor<R>, expected_shape: &[usize]) -> Result<()> {
        if weight.shape() != expected_shape {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "shape mismatch for '{name}': expected {:?}, got {:?}",
                    expected_shape,
                    weight.shape()
                ),
            });
        }
        Ok(())
    }
}

// --- MLP with SwiGLU Activation ---

struct Mlp<R: boostr::Runtime> {
    gate_proj: MaybeLora<R>,
    up_proj: MaybeLora<R>,
    down_proj: MaybeLora<R>,
}

impl<R> Mlp<R>
where
    R: boostr::Runtime<DType = DType>,
{
    fn new<C>(
        cfg: &ModelConfig,
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
        let intermediate_size = cfg.intermediate_size.unwrap_or(cfg.hidden_size * 4);

        // `CausalSelfAttention::projection` is the single builder for an
        // optionally-adapted no-bias projection: it takes the factors from the
        // same `VarBuilder` (so they carry the model dtype and are named
        // `<prefix>.lora_a` / `<prefix>.lora_b`) and returns both the base and
        // the adapters trainable. Freezing the wrapped base stays
        // `LitGPT::new`'s job, via `FrozenParams`.
        Ok(Self {
            gate_proj: CausalSelfAttention::<R>::projection(
                vb.pp("gate_proj"),
                "gate_proj",
                &[intermediate_size, cfg.hidden_size],
                dtype,
                lora,
                client,
            )?,
            up_proj: CausalSelfAttention::<R>::projection(
                vb.pp("up_proj"),
                "up_proj",
                &[intermediate_size, cfg.hidden_size],
                dtype,
                lora,
                client,
            )?,
            down_proj: CausalSelfAttention::<R>::projection(
                vb.pp("down_proj"),
                "down_proj",
                &[cfg.hidden_size, intermediate_size],
                dtype,
                lora,
                client,
            )?,
        })
    }

    fn with_ids(
        cfg: &ModelConfig,
        gate_proj: AttnProjWithIds<R>,
        up_proj: AttnProjWithIds<R>,
        down_proj: AttnProjWithIds<R>,
        frozen: &FrozenParams,
    ) -> Result<Self> {
        let intermediate_size = cfg.intermediate_size.unwrap_or(cfg.hidden_size * 4);

        Ok(Self {
            gate_proj: CausalSelfAttention::<R>::projection_with_ids(
                "gate_proj",
                gate_proj,
                &[intermediate_size, cfg.hidden_size],
                frozen,
            )?,
            up_proj: CausalSelfAttention::<R>::projection_with_ids(
                "up_proj",
                up_proj,
                &[intermediate_size, cfg.hidden_size],
                frozen,
            )?,
            down_proj: CausalSelfAttention::<R>::projection_with_ids(
                "down_proj",
                down_proj,
                &[cfg.hidden_size, intermediate_size],
                frozen,
            )?,
        })
    }

    /// The three projections in `named_parameters` order.
    fn projections(&self) -> [&MaybeLora<R>; 3] {
        [&self.gate_proj, &self.up_proj, &self.down_proj]
    }

    fn forward<C>(
        &self,
        client: &C,
        x: &boostr::autograd::Var<R>,
    ) -> Result<boostr::autograd::Var<R>>
    where
        // `BinaryOps` is `LoraLinear::forward`'s bound: an adapted projection
        // scales and adds its low-rank branch. A plain `Linear` needs none of
        // it, but the variant is a runtime choice, so the bound applies either
        // way.
        C: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::TypeConversionOps<R>,
        R::Client: boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::TypeConversionOps<R>,
    {
        // SwiGLU: silu(gate) * up, delegated to boostr/numr's fused Var op.
        let gate_linear = self.gate_proj.forward(client, x)?;
        let up = self.up_proj.forward(client, x)?;
        let fused = boostr::autograd::var_swiglu(&gate_linear, &up, client)?;
        let out = self.down_proj.forward(client, &fused)?;

        // DEBUG: Check weight norms
        if std::env::var("DEBUG_MLP").is_ok() {
            let gate_w = Self::tensor_to_f32_vec(client, self.gate_proj.weight().tensor())?;
            let w_std = {
                let mean = gate_w.iter().sum::<f32>() / gate_w.len() as f32;
                (gate_w.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / gate_w.len() as f32)
                    .sqrt()
            };
            let w_norm: f32 = gate_w.iter().map(|v| v * v).sum::<f32>().sqrt();
            eprintln!(
                "[MLP weights] gate_proj.weight: shape={:?}, std={:.6e}, norm={:.4}",
                self.gate_proj.weight().tensor().shape(),
                w_std,
                w_norm
            );
        }

        // DEBUG: Check MLP forward values
        if std::env::var("DEBUG_MLP").is_ok() {
            let x_vec = Self::tensor_to_f32_vec(client, x.tensor())?;
            let gate_vec = Self::tensor_to_f32_vec(client, gate_linear.tensor())?;
            let up_vec = Self::tensor_to_f32_vec(client, up.tensor())?;
            let fused_vec = Self::tensor_to_f32_vec(client, fused.tensor())?;
            let out_vec = Self::tensor_to_f32_vec(client, out.tensor())?;
            let x_std = {
                let mean = x_vec.iter().sum::<f32>() / x_vec.len() as f32;
                (x_vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x_vec.len() as f32).sqrt()
            };
            let gate_std = {
                let mean = gate_vec.iter().sum::<f32>() / gate_vec.len() as f32;
                (gate_vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / gate_vec.len() as f32)
                    .sqrt()
            };
            let up_std = {
                let mean = up_vec.iter().sum::<f32>() / up_vec.len() as f32;
                (up_vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / up_vec.len() as f32)
                    .sqrt()
            };
            let fused_std = {
                let mean = fused_vec.iter().sum::<f32>() / fused_vec.len() as f32;
                (fused_vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / fused_vec.len() as f32)
                    .sqrt()
            };
            let out_std = {
                let mean = out_vec.iter().sum::<f32>() / out_vec.len() as f32;
                (out_vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / out_vec.len() as f32)
                    .sqrt()
            };
            eprintln!(
                "[MLP forward] x_std={:.6e}, gate_std={:.6e}, up_std={:.6e}, fused_std={:.6e}, out_std={:.6e}",
                x_std, gate_std, up_std, fused_std, out_std
            );
            eprintln!(
                "[MLP forward] x[:3]={:?}, gate[:3]={:?}, up[:3]={:?}, fused[:3]={:?}, out[:3]={:?}",
                &x_vec[..3.min(x_vec.len())],
                &gate_vec[..3.min(gate_vec.len())],
                &up_vec[..3.min(up_vec.len())],
                &fused_vec[..3.min(fused_vec.len())],
                &out_vec[..3.min(out_vec.len())]
            );
        }

        Ok(out)
    }

    fn trainable_parameters(&self) -> Vec<(boostr::autograd::TensorId, &boostr::autograd::Var<R>)> {
        let mut params = Vec::new();
        params.extend(self.gate_proj.trainable_parameters());
        params.extend(self.up_proj.trainable_parameters());
        params.extend(self.down_proj.trainable_parameters());
        params
    }

    fn tensor_to_f32_vec<C>(client: &C, tensor: &Tensor<R>) -> Result<Vec<f32>>
    where
        C: boostr::RuntimeClient<R> + boostr::TensorOps<R> + boostr::TypeConversionOps<R>,
    {
        let tensor = if tensor.dtype() == DType::F32 {
            tensor.contiguous()?
        } else {
            client.cast(tensor, DType::F32)?
        };
        Ok(tensor.contiguous()?.to_vec::<f32>())
    }
}

// --- Fine-Grained Mixture of Experts (MoE) ---
// Delegates oxidizr's fine-grained MoE to boostr::nn::moe so routing,
// SwiGLU experts, auxiliary loss, z-loss, and parameter identity all use the
// shared Var-based implementation.

struct Expert<R: boostr::Runtime> {
    expert: boostr::nn::moe::expert::Expert<R>,
}

impl<R> Expert<R>
where
    R: boostr::Runtime<DType = DType>,
{
    fn new<C>(
        hidden: usize,
        intermediate: usize,
        mut vb: VarBuilder<'_, R>,
        client: &C,
        dtype: DType,
    ) -> Result<Self>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let gate_proj =
            Self::linear_no_bias(vb.pp("gate_proj"), &[intermediate, hidden], dtype, client)?;
        let up_proj =
            Self::linear_no_bias(vb.pp("up_proj"), &[intermediate, hidden], dtype, client)?;
        let down_proj =
            Self::linear_no_bias(vb.pp("down_proj"), &[hidden, intermediate], dtype, client)?;

        Ok(Self {
            expert: boostr::nn::moe::expert::Expert::new(gate_proj, up_proj, down_proj),
        })
    }

    fn with_ids(
        hidden: usize,
        intermediate: usize,
        gate_proj: (Tensor<R>, boostr::autograd::TensorId),
        up_proj: (Tensor<R>, boostr::autograd::TensorId),
        down_proj: (Tensor<R>, boostr::autograd::TensorId),
        frozen: &FrozenParams,
    ) -> Result<Self> {
        let gate_proj = Self::linear_with_ids(
            "gate_proj.weight",
            gate_proj.0,
            gate_proj.1,
            &[intermediate, hidden],
            frozen,
        )?;
        let up_proj = Self::linear_with_ids(
            "up_proj.weight",
            up_proj.0,
            up_proj.1,
            &[intermediate, hidden],
            frozen,
        )?;
        let down_proj = Self::linear_with_ids(
            "down_proj.weight",
            down_proj.0,
            down_proj.1,
            &[hidden, intermediate],
            frozen,
        )?;

        Ok(Self {
            expert: boostr::nn::moe::expert::Expert::new(gate_proj, up_proj, down_proj),
        })
    }

    fn into_inner(self) -> boostr::nn::moe::expert::Expert<R> {
        self.expert
    }

    fn extend_trainable_parameters<'a>(
        params: &mut Vec<(boostr::autograd::TensorId, &'a boostr::autograd::Var<R>)>,
        expert: &'a boostr::nn::moe::expert::Expert<R>,
    ) {
        params.extend(expert.gate_proj().trainable_parameters());
        params.extend(expert.up_proj().trainable_parameters());
        params.extend(expert.down_proj().trainable_parameters());
    }

    fn linear_no_bias<C>(
        mut vb: VarBuilder<'_, R>,
        expected_shape: &[usize],
        dtype: DType,
        client: &C,
    ) -> Result<boostr::nn::Linear<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let param_name = Self::param_name(&vb, "weight");
        let weight = take_or_init_tensor(
            &mut vb,
            "weight",
            expected_shape,
            dtype,
            Init::PyTorchLinear,
            client,
        )?;
        let weight_id = weight.id();
        Self::ensure_shape(&param_name, &weight, expected_shape)?;
        Ok(boostr::nn::Linear::with_ids(
            weight,
            weight_id,
            None,
            INITIALIZED_TRAINABLE,
        ))
    }

    fn linear_with_ids(
        name: &str,
        weight: Tensor<R>,
        weight_id: boostr::autograd::TensorId,
        expected_shape: &[usize],
        frozen: &FrozenParams,
    ) -> Result<boostr::nn::Linear<R>> {
        Self::ensure_shape(name, &weight, expected_shape)?;
        let trainable = frozen.trainable(weight_id);
        Ok(boostr::nn::Linear::with_ids(
            weight, weight_id, None, trainable,
        ))
    }

    fn ensure_shape(name: &str, weight: &Tensor<R>, expected_shape: &[usize]) -> Result<()> {
        if weight.shape() != expected_shape {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "shape mismatch for '{name}': expected {:?}, got {:?}",
                    expected_shape,
                    weight.shape()
                ),
            });
        }
        Ok(())
    }

    fn param_name(vb: &VarBuilder<'_, R>, leaf: &str) -> String {
        if vb.prefix().is_empty() {
            leaf.to_string()
        } else {
            format!("{}.{}", vb.prefix(), leaf)
        }
    }
}

struct Router<R: boostr::Runtime> {
    router: boostr::nn::moe::router::MoeRouter<R>,
}

impl<R> Router<R>
where
    R: boostr::Runtime<DType = DType>,
{
    fn new<C>(
        hidden: usize,
        config: boostr::nn::moe::router::MoeRouterConfig,
        mut vb: VarBuilder<'_, R>,
        client: &C,
        dtype: DType,
    ) -> Result<Self>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let gate = Expert::<R>::linear_no_bias(
            vb.pp("gate"),
            &[config.num_experts, hidden],
            dtype,
            client,
        )?;
        Ok(Self {
            router: boostr::nn::moe::router::MoeRouter::new(gate, config),
        })
    }

    fn with_ids(
        hidden: usize,
        config: boostr::nn::moe::router::MoeRouterConfig,
        gate: (Tensor<R>, boostr::autograd::TensorId),
        frozen: &FrozenParams,
    ) -> Result<Self> {
        let gate = Expert::<R>::linear_with_ids(
            "gate.weight",
            gate.0,
            gate.1,
            &[config.num_experts, hidden],
            frozen,
        )?;
        Ok(Self {
            router: boostr::nn::moe::router::MoeRouter::new(gate, config),
        })
    }

    fn into_inner(self) -> boostr::nn::moe::router::MoeRouter<R> {
        self.router
    }
}

struct MoeLayer<R: boostr::Runtime> {
    moe: boostr::nn::moe::layer::MoeLayer<R>,
}

impl<R> MoeLayer<R>
where
    R: boostr::Runtime<DType = DType>,
{
    fn new<C>(
        cfg: &ModelConfig,
        mut vb: VarBuilder<'_, R>,
        client: &C,
        dtype: DType,
    ) -> Result<Self>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let (num_experts, experts_per_tok, intermediate) = Self::required_moe_dims(cfg)?;

        let mut experts = Vec::with_capacity(num_experts);
        for expert_idx in 0..num_experts {
            let expert = Expert::new(
                cfg.hidden_size,
                intermediate,
                vb.pp(&format!("experts.{expert_idx}")),
                client,
                dtype,
            )?;
            experts.push(expert.into_inner());
        }

        let shared_expert = Expert::new(
            cfg.hidden_size,
            intermediate,
            vb.pp("shared_expert"),
            client,
            dtype,
        )?
        .into_inner();
        let router = Router::new(
            cfg.hidden_size,
            Self::router_config(num_experts, experts_per_tok),
            vb.pp("router"),
            client,
            dtype,
        )?
        .into_inner();

        let moe = boostr::nn::moe::layer::MoeLayer::new(router, experts, Some(shared_expert))
            .with_shared_expert_scale(Self::shared_expert_scale(experts_per_tok))?;

        Ok(Self { moe })
    }

    #[allow(clippy::type_complexity)]
    fn with_ids(
        cfg: &ModelConfig,
        experts: Vec<(
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
        )>,
        shared_expert: (
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
        ),
        router_gate: (Tensor<R>, boostr::autograd::TensorId),
        frozen: &FrozenParams,
    ) -> Result<Self> {
        let (num_experts, experts_per_tok, intermediate) = Self::required_moe_dims(cfg)?;
        if experts.len() != num_experts {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "MoE expected {num_experts} experts from cfg.moe.num_experts, got {}",
                    experts.len()
                ),
            });
        }

        let mut boostr_experts = Vec::with_capacity(num_experts);
        for (gate_proj, up_proj, down_proj) in experts {
            let expert = Expert::with_ids(
                cfg.hidden_size,
                intermediate,
                gate_proj,
                up_proj,
                down_proj,
                frozen,
            )?;
            boostr_experts.push(expert.into_inner());
        }

        let shared_expert = Expert::with_ids(
            cfg.hidden_size,
            intermediate,
            shared_expert.0,
            shared_expert.1,
            shared_expert.2,
            frozen,
        )?
        .into_inner();
        let router = Router::with_ids(
            cfg.hidden_size,
            Self::router_config(num_experts, experts_per_tok),
            router_gate,
            frozen,
        )?
        .into_inner();

        let moe =
            boostr::nn::moe::layer::MoeLayer::new(router, boostr_experts, Some(shared_expert))
                .with_shared_expert_scale(Self::shared_expert_scale(experts_per_tok))?;

        Ok(Self { moe })
    }

    /// Forward pass for MoE layer.
    ///
    /// Returns: (output, load_balance_loss, z_loss)
    /// - output: The mixed expert output [B, Seq, Hidden]
    /// - load_balance_loss: boostr::nn::moe::MoeOutput::aux_loss
    /// - z_loss: boostr::nn::moe::MoeOutput::z_loss
    fn forward<C>(
        &self,
        client: &C,
        x: &boostr::autograd::Var<R>,
    ) -> Result<(
        boostr::autograd::Var<R>,
        boostr::autograd::Var<R>,
        boostr::autograd::Var<R>,
    )>
    where
        C: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::ReduceOps<R>
            + boostr::ShapeOps<R>
            + boostr::ActivationOps<R>
            + boostr::SortingOps<R>
            + boostr::IndexingOps<R>
            + boostr::CompareOps<R>
            + boostr::BinaryOps<R>
            + boostr::UnaryOps<R>,
        R::Client: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::ActivationOps<R>
            + boostr::ReduceOps<R>
            + boostr::ShapeOps<R>
            + boostr::BinaryOps<R>
            + boostr::UnaryOps<R>,
    {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(boostr::error::Error::ModelError {
                reason: format!("MoE expects [B, Seq, Hidden], got {:?}", shape),
            });
        }
        let b_sz = shape[0];
        let seq_len = shape[1];
        let hidden = shape[2];
        let num_tokens = b_sz * seq_len;

        let x_flat = boostr::autograd::var_reshape(x, &[num_tokens, hidden])
            .map_err(boostr::error::Error::Numr)?;
        let moe_output = self.moe.forward(client, &x_flat)?;
        let output = boostr::autograd::var_reshape(&moe_output.output, &[b_sz, seq_len, hidden])
            .map_err(boostr::error::Error::Numr)?;

        Ok((output, moe_output.aux_loss, moe_output.z_loss))
    }

    fn trainable_parameters(&self) -> Vec<(boostr::autograd::TensorId, &boostr::autograd::Var<R>)> {
        let mut params = Vec::new();
        params.extend(self.moe.router().gate().trainable_parameters());
        for expert in self.moe.experts() {
            Expert::<R>::extend_trainable_parameters(&mut params, expert);
        }
        if let Some(shared_expert) = self.moe.shared_expert() {
            Expert::<R>::extend_trainable_parameters(&mut params, shared_expert);
        }
        params
    }

    fn router_config(
        num_experts: usize,
        experts_per_tok: usize,
    ) -> boostr::nn::moe::router::MoeRouterConfig {
        boostr::nn::moe::router::MoeRouterConfig {
            router_temperature: 2.0,
            load_balance_loss_mode:
                boostr::nn::moe::router::MoeLoadBalanceLossMode::SwitchPlusSquaredProb,
            ..boostr::nn::moe::router::MoeRouterConfig::new(num_experts, experts_per_tok)
        }
    }

    fn shared_expert_scale(experts_per_tok: usize) -> f32 {
        1.0 / (experts_per_tok as f32 + 1.0)
    }

    fn required_moe_dims(cfg: &ModelConfig) -> Result<(usize, usize, usize)> {
        let moe = cfg
            .moe
            .as_ref()
            .ok_or_else(|| boostr::error::Error::ModelError {
                reason: "moe config missing while building an MoE FfnLayer".into(),
            })?;
        let intermediate = Self::required_positive("moe.intermediate_size", moe.intermediate_size)?;
        let num_experts = moe.num_experts;
        let experts_per_tok = moe.experts_per_tok;
        if experts_per_tok > num_experts {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "MoE experts_per_tok ({experts_per_tok}) must not exceed num_experts ({num_experts})"
                ),
            });
        }
        Ok((num_experts, experts_per_tok, intermediate))
    }

    fn required_positive(name: &str, value: Option<usize>) -> Result<usize> {
        let value = value.ok_or_else(|| boostr::error::Error::ModelError {
            reason: format!("MoE requires cfg.{name}"),
        })?;
        if value == 0 {
            return Err(boostr::error::Error::ModelError {
                reason: format!("MoE cfg.{name} must be non-zero"),
            });
        }
        Ok(value)
    }
}

// --- Feed-Forward Network Layer (MLP or MoE) ---
// Supports both simple MLP and Mixture of Experts based on config

// Both payloads are large (hundreds of bytes of projections), so both are boxed:
// an unboxed variant would make every Block pay that variant's full size.
enum FfnLayer<R: boostr::Runtime> {
    Mlp(Box<Mlp<R>>),
    Moe(Box<MoeLayer<R>>),
}

impl<R> FfnLayer<R>
where
    R: boostr::Runtime<DType = DType>,
{
    fn new<C>(
        cfg: &ModelConfig,
        vb: VarBuilder<'_, R>,
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
        if cfg.moe.is_some() {
            // MoE mode. boostr's `nn::moe::expert::Expert` stores concrete
            // `Linear` fields, so an expert projection cannot hold an adapter.
            // `validate_lora` already rejects this pairing at config load; the
            // check is repeated here so no other caller can build a model that
            // claims adapters it does not have.
            Self::reject_moe_lora(lora)?;
            Ok(FfnLayer::Moe(Box::new(MoeLayer::new(
                cfg, vb, client, dtype,
            )?)))
        } else {
            // Simple MLP mode (no MoE)
            Ok(FfnLayer::Mlp(Box::new(Mlp::new(
                cfg, vb, client, dtype, lora,
            )?)))
        }
    }

    /// Refuse a `lora` block that names an MLP projection while this layer is
    /// a MoE layer.
    fn reject_moe_lora(lora: Option<&LoraConfig>) -> Result<()> {
        let Some(lora) = lora else {
            return Ok(());
        };
        let Some(target) = crate::config::MLP_LORA_TARGETS
            .iter()
            .find(|&&target| lora.targets(target))
        else {
            return Ok(());
        };
        Err(boostr::error::Error::ModelError {
            reason: format!(
                "lora.target_modules names '{target}' but this layer is a MoE layer, whose \
                 experts are built from boostr's nn::moe::expert::Expert — it stores concrete \
                 Linear fields and cannot hold a LoRA adapter. Remove '{target}' from \
                 lora.target_modules, or remove num_experts from the model config."
            ),
        })
    }

    #[allow(clippy::type_complexity)]
    fn with_ids(
        cfg: &ModelConfig,
        mlp: Option<(AttnProjWithIds<R>, AttnProjWithIds<R>, AttnProjWithIds<R>)>,
        moe: Option<(
            Vec<(
                (Tensor<R>, boostr::autograd::TensorId),
                (Tensor<R>, boostr::autograd::TensorId),
                (Tensor<R>, boostr::autograd::TensorId),
            )>,
            (
                (Tensor<R>, boostr::autograd::TensorId),
                (Tensor<R>, boostr::autograd::TensorId),
                (Tensor<R>, boostr::autograd::TensorId),
            ),
            (Tensor<R>, boostr::autograd::TensorId),
        )>,
        frozen: &FrozenParams,
    ) -> Result<Self> {
        if cfg.moe.is_some() {
            let (experts, shared_expert, router_gate) =
                moe.ok_or_else(|| boostr::error::Error::ModelError {
                    reason: "FfnLayer::with_ids requires MoE weights when cfg.moe is set".into(),
                })?;
            Ok(FfnLayer::Moe(Box::new(MoeLayer::with_ids(
                cfg,
                experts,
                shared_expert,
                router_gate,
                frozen,
            )?)))
        } else {
            let (gate_proj, up_proj, down_proj) =
                mlp.ok_or_else(|| boostr::error::Error::ModelError {
                    reason: "FfnLayer::with_ids requires MLP weights when cfg.moe is unset".into(),
                })?;
            Ok(FfnLayer::Mlp(Box::new(Mlp::with_ids(
                cfg, gate_proj, up_proj, down_proj, frozen,
            )?)))
        }
    }

    /// Forward pass.
    /// Returns: (output, load_balance_loss, z_loss)
    /// For MLP, losses are constant zero Vars.
    fn forward<C>(
        &self,
        client: &C,
        x: &boostr::autograd::Var<R>,
    ) -> Result<(
        boostr::autograd::Var<R>,
        boostr::autograd::Var<R>,
        boostr::autograd::Var<R>,
    )>
    where
        C: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::ReduceOps<R>
            + boostr::ShapeOps<R>
            + boostr::ActivationOps<R>
            + boostr::SortingOps<R>
            + boostr::IndexingOps<R>
            + boostr::CompareOps<R>
            + boostr::BinaryOps<R>
            + boostr::UnaryOps<R>
            + boostr::TypeConversionOps<R>,
        R::Client: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::ActivationOps<R>
            + boostr::ReduceOps<R>
            + boostr::ShapeOps<R>
            + boostr::BinaryOps<R>
            + boostr::UnaryOps<R>
            + boostr::TypeConversionOps<R>,
    {
        match self {
            FfnLayer::Mlp(mlp) => {
                let out = mlp.forward(client, x)?;
                let load_balance_loss = boostr::autograd::Var::new(
                    Tensor::<R>::zeros(&[1], DType::F32, x.tensor().device()),
                    false,
                );
                let z_loss = boostr::autograd::Var::new(
                    Tensor::<R>::zeros(&[1], DType::F32, x.tensor().device()),
                    false,
                );
                Ok((out, load_balance_loss, z_loss))
            }
            FfnLayer::Moe(moe) => moe.forward(client, x),
        }
    }

    fn trainable_parameters(&self) -> Vec<(boostr::autograd::TensorId, &boostr::autograd::Var<R>)> {
        match self {
            FfnLayer::Mlp(mlp) => mlp.trainable_parameters(),
            FfnLayer::Moe(moe) => moe.trainable_parameters(),
        }
    }
}

// --- Hybrid Block (supports both Mamba and Attention) ---

// Output from a block, including optional auxiliary losses
pub struct BlockOutput<R: boostr::Runtime> {
    pub hidden: boostr::autograd::Var<R>,
    pub aux_loss: Option<boostr::autograd::Var<R>>, // MoE load balance loss
    pub z_loss: Option<boostr::autograd::Var<R>>,   // Router z-loss (prevents softmax saturation)
}

/// Epsilon for the Mamba block-input pre-norms.
///
/// Deliberately 1e-6, not the 1e-5 used by the attention blocks' norms — the
/// deleted `Mamba2Block`/`Mamba3Block` wrappers both used 1e-6 and changing it
/// would silently alter training.
const MAMBA_PRENORM_EPS: f32 = 1e-6;

enum BlockType<R: boostr::Runtime> {
    // The SSM blocks are far larger than the attention variants; boxing them
    // keeps every Block from paying the largest variant's footprint.
    Mamba {
        block: Box<boostr::model::Mamba1<R>>,
        norm: RmsNorm<R>,
    },
    Mamba2 {
        block: Box<boostr::model::Mamba2<R>>,
        /// Pre-norm on the block input.
        ///
        /// boostr's `Mamba2` is the mixer alone — its internal norm operates on
        /// the gated SSM output at `intermediate_size`. The block-level input
        /// pre-norm at `hidden_size` lives here, matching the structure the
        /// Mamba1 arm already uses.
        norm: RmsNorm<R>,
    },
    Mamba3 {
        block: Box<boostr::model::Mamba3<R>>,
        /// Pre-norm on the block input. Mamba3 uses LayerNorm, not RmsNorm.
        norm: boostr::nn::LayerNorm<R>,
    },
    Attention {
        attn: Box<MultiHeadLatentAttention<R>>,
        ffn: FfnLayer<R>, // MLP or MoE based on config
        input_layernorm: RmsNorm<R>,
        post_attention_layernorm: RmsNorm<R>,
    },
    LegacyAttention {
        attn: Box<CausalSelfAttention<R>>,
        mlp: Box<Mlp<R>>,
        input_layernorm: RmsNorm<R>,
        post_attention_layernorm: RmsNorm<R>,
    },
}

struct Block<R: boostr::Runtime> {
    block_type: BlockType<R>,
}

impl<R> Block<R>
where
    R: boostr::Runtime<DType = DType>,
{
    // Create appropriate block type based on configuration
    fn new<C>(
        cfg: &ModelConfig,
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
        // The config's value, not a literal: boostr's `ModelConfig` carries
        // `rms_norm_eps`, and hardcoding it here would silently ignore a
        // config that asks for anything else. (Mamba pre-norms deliberately
        // keep MAMBA_PRENORM_EPS — see its doc comment.)
        let eps = cfg.rms_norm_eps as f32;

        // Determine if this should be a Mamba layer. `is_ssm_layer` handles
        // all three config shapes (pure attention, pure SSM, hybrid) — a raw
        // `hybrid_layers` read would misreport every layer of a pure-SSM
        // config as attention, since pure-SSM configs carry no
        // `hybrid_layers` at all.
        let is_mamba = cfg.is_ssm_layer(layer_idx);

        // Determine if using Mamba3 (Trapezoidal + Complex RoPE + MIMO)
        let use_mamba3 = cfg
            .ssm
            .as_ref()
            .map(|ssm| ssm.variant == "mamba3")
            .unwrap_or(false);

        // Determine if using Mamba2 (State Space Duality)
        let use_mamba2 = cfg.ssm.as_ref().is_some_and(|s| s.variant != "mamba1");

        // Determine if using MLA (Multi-Head Latent Attention) or legacy GQA
        let use_mla = cfg
            .attention
            .as_ref()
            .and_then(|a| a.kv_latent_dim)
            .is_some();

        let block_type = if is_mamba && use_mamba3 && use_mamba2 {
            // Mamba3 layer (extends Mamba2 with trapezoidal discretization, complex RoPE, MIMO)
            let mamba3_config = Self::mamba3_config(cfg)?;
            let norm = Self::take_layer_norm(
                vb.pp("norm"),
                cfg.hidden_size,
                MAMBA_PRENORM_EPS,
                dtype,
                client,
            )?;
            let mut mamba3_vb = vb.pp("mamba3");
            BlockType::Mamba3 {
                block: Box::new(boostr::model::Mamba3::init(
                    &mamba3_config,
                    &mut mamba3_vb,
                    dtype,
                    client,
                    INITIALIZED_TRAINABLE,
                )?),
                norm,
            }
        } else if is_mamba && use_mamba2 {
            // Mamba2 layer (SSD - State Space Duality)
            let mamba2_config = Self::mamba2_config(cfg)?;
            let norm = Self::take_rms_norm(
                vb.pp("norm"),
                cfg.hidden_size,
                MAMBA_PRENORM_EPS,
                dtype,
                client,
            )?;
            let mut mamba2_vb = vb.pp("mamba2");
            BlockType::Mamba2 {
                block: Box::new(boostr::model::Mamba2::init(
                    &mamba2_config,
                    &mut mamba2_vb,
                    dtype,
                    client,
                    INITIALIZED_TRAINABLE,
                )?),
                norm,
            }
        } else if is_mamba {
            // Mamba1 layer (original sequential)
            let mamba1_config = Self::mamba1_config(cfg)?;
            let mut mamba1_vb = vb.pp("mamba");
            BlockType::Mamba {
                block: Box::new(boostr::model::Mamba1::init(
                    &mamba1_config,
                    &mut mamba1_vb,
                    dtype,
                    client,
                    INITIALIZED_TRAINABLE,
                )?),
                norm: Self::take_rms_norm(vb.pp("norm"), cfg.hidden_size, eps, dtype, client)?,
            }
        } else if use_mla {
            // MLA attention layer (with MLP or MoE based on config)
            let ffn_prefix = if cfg.moe.is_some() { "moe" } else { "mlp" };
            BlockType::Attention {
                attn: Box::new(MultiHeadLatentAttention::new(
                    cfg,
                    vb.pp("self_attn"),
                    client,
                    dtype,
                )?),
                ffn: FfnLayer::new(cfg, vb.pp(ffn_prefix), client, dtype, lora)?,
                input_layernorm: Self::take_rms_norm(
                    vb.pp("input_layernorm"),
                    cfg.hidden_size,
                    eps,
                    dtype,
                    client,
                )?,
                post_attention_layernorm: Self::take_rms_norm(
                    vb.pp("post_attention_layernorm"),
                    cfg.hidden_size,
                    eps,
                    dtype,
                    client,
                )?,
            }
        } else {
            // Legacy GQA + MLP attention layer (original nano)
            BlockType::LegacyAttention {
                attn: Box::new(CausalSelfAttention::new(
                    cfg,
                    vb.pp("self_attn"),
                    client,
                    dtype,
                    lora,
                )?),
                mlp: Box::new(Mlp::new(cfg, vb.pp("mlp"), client, dtype, lora)?),
                input_layernorm: Self::take_rms_norm(
                    vb.pp("input_layernorm"),
                    cfg.hidden_size,
                    eps,
                    dtype,
                    client,
                )?,
                post_attention_layernorm: Self::take_rms_norm(
                    vb.pp("post_attention_layernorm"),
                    cfg.hidden_size,
                    eps,
                    dtype,
                    client,
                )?,
            }
        };

        Ok(Self { block_type })
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn with_ids(
        cfg: &ModelConfig,
        layer_idx: usize,
        mamba1: Option<(
            boostr::model::mamba::Mamba1WeightsWithIds<R>,
            (Tensor<R>, boostr::autograd::TensorId),
        )>,
        // Each carries the block-input pre-norm alongside the mixer weights:
        // RmsNorm weight for Mamba2, LayerNorm weight+bias for Mamba3.
        mamba2: Option<(
            boostr::model::mamba::Mamba2WeightsWithIds<R>,
            (Tensor<R>, boostr::autograd::TensorId),
        )>,
        mamba3: Option<(
            boostr::model::mamba::Mamba3WeightsWithIds<R>,
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
        )>,
        attention: Option<(
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            Option<(AttnProjWithIds<R>, AttnProjWithIds<R>, AttnProjWithIds<R>)>,
            Option<(
                Vec<(
                    (Tensor<R>, boostr::autograd::TensorId),
                    (Tensor<R>, boostr::autograd::TensorId),
                    (Tensor<R>, boostr::autograd::TensorId),
                )>,
                (
                    (Tensor<R>, boostr::autograd::TensorId),
                    (Tensor<R>, boostr::autograd::TensorId),
                    (Tensor<R>, boostr::autograd::TensorId),
                ),
                (Tensor<R>, boostr::autograd::TensorId),
            )>,
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
        )>,
        legacy_attention: Option<(
            AttnProjWithIds<R>,
            AttnProjWithIds<R>,
            AttnProjWithIds<R>,
            AttnProjWithIds<R>,
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            AttnProjWithIds<R>,
            AttnProjWithIds<R>,
            AttnProjWithIds<R>,
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
        )>,
        device: &R::Device,
        frozen: &FrozenParams,
    ) -> Result<Self> {
        // The config's value, not a literal: boostr's `ModelConfig` carries
        // `rms_norm_eps`, and hardcoding it here would silently ignore a
        // config that asks for anything else. (Mamba pre-norms deliberately
        // keep MAMBA_PRENORM_EPS — see its doc comment.)
        let eps = cfg.rms_norm_eps as f32;

        // Determine if this should be a Mamba layer. `is_ssm_layer` handles
        // all three config shapes (pure attention, pure SSM, hybrid) — a raw
        // `hybrid_layers` read would misreport every layer of a pure-SSM
        // config as attention, since pure-SSM configs carry no
        // `hybrid_layers` at all.
        let is_mamba = cfg.is_ssm_layer(layer_idx);

        // Determine if using Mamba3 (Trapezoidal + Complex RoPE + MIMO)
        let use_mamba3 = cfg
            .ssm
            .as_ref()
            .map(|ssm| ssm.variant == "mamba3")
            .unwrap_or(false);

        // Determine if using Mamba2 (State Space Duality)
        let use_mamba2 = cfg.ssm.as_ref().is_some_and(|s| s.variant != "mamba1");

        // Determine if using MLA (Multi-Head Latent Attention) or legacy GQA
        let use_mla = cfg
            .attention
            .as_ref()
            .and_then(|a| a.kv_latent_dim)
            .is_some();

        let block_type = if is_mamba && use_mamba3 && use_mamba2 {
            let (weights, norm_weight, norm_bias) =
                mamba3.ok_or_else(|| boostr::error::Error::ModelError {
                    reason:
                        "Block::with_ids requires Mamba3 weights and pre-norm for a Mamba3 layer"
                            .into(),
                })?;
            // boostr's Mamba3 applies one `trainable` flag to every raw
            // parameter it wraps, so the whole group must agree.
            let mut raw = vec![
                ("b_bias", weights.b_bias.1),
                ("c_bias", weights.c_bias.1),
                ("a_log", weights.a_log.1),
            ];
            if let Some((_, id)) = weights.dt_bias.as_ref() {
                raw.push(("dt_bias", *id));
            }
            if let Some((_, id)) = weights.d_param.as_ref() {
                raw.push(("d_param", *id));
            }
            let raw_trainable = frozen
                .group_trainable(&format!("layers.{layer_idx}.mamba3 raw parameters"), &raw)?;
            BlockType::Mamba3 {
                block: Box::new(boostr::model::Mamba3::with_ids(
                    Self::mamba3_config(cfg)?,
                    weights,
                    raw_trainable,
                )),
                norm: Self::layer_norm_with_ids(
                    "norm",
                    norm_weight,
                    norm_bias,
                    cfg.hidden_size,
                    MAMBA_PRENORM_EPS,
                    frozen,
                )?,
            }
        } else if is_mamba && use_mamba2 {
            let (weights, norm) = mamba2.ok_or_else(|| boostr::error::Error::ModelError {
                reason: "Block::with_ids requires Mamba2 weights and pre-norm for a Mamba2 layer"
                    .into(),
            })?;
            // boostr's Mamba2 applies one `trainable` flag to every raw
            // parameter it wraps, so the whole group must agree.
            let mut raw = vec![("a_log", weights.a_log.1)];
            if let Some((_, id)) = weights.dt_bias.as_ref() {
                raw.push(("dt_bias", *id));
            }
            if let Some((_, id)) = weights.d_param.as_ref() {
                raw.push(("d_param", *id));
            }
            let raw_trainable = frozen
                .group_trainable(&format!("layers.{layer_idx}.mamba2 raw parameters"), &raw)?;
            BlockType::Mamba2 {
                block: Box::new(boostr::model::Mamba2::with_ids(
                    Self::mamba2_config(cfg)?,
                    weights,
                    raw_trainable,
                )),
                norm: Self::rms_norm_with_id(
                    "norm.weight",
                    norm.0,
                    norm.1,
                    cfg.hidden_size,
                    MAMBA_PRENORM_EPS,
                    frozen,
                )?,
            }
        } else if is_mamba {
            let (weights, norm) = mamba1.ok_or_else(|| boostr::error::Error::ModelError {
                reason: "Block::with_ids requires Mamba1 weights and norm for a Mamba1 layer"
                    .into(),
            })?;
            // boostr's Mamba1 applies one `trainable` flag to every raw
            // parameter it wraps, so the whole group must agree.
            let mut raw = vec![("a_log", weights.a_log.1)];
            if let Some((_, id)) = weights.d_param.as_ref() {
                raw.push(("d_param", *id));
            }
            let raw_trainable = frozen
                .group_trainable(&format!("layers.{layer_idx}.mamba raw parameters"), &raw)?;
            BlockType::Mamba {
                block: Box::new(boostr::model::Mamba1::with_ids(
                    Self::mamba1_config(cfg)?,
                    weights,
                    raw_trainable,
                )),
                norm: Self::rms_norm_with_id(
                    "norm.weight",
                    norm.0,
                    norm.1,
                    cfg.hidden_size,
                    eps,
                    frozen,
                )?,
            }
        } else if use_mla {
            let (
                q_a_proj,
                q_b_proj,
                q_a_layernorm,
                kv_a_proj_with_mqa,
                kv_a_layernorm,
                kv_b_proj,
                o_proj,
                mlp,
                moe,
                input_layernorm,
                post_attention_layernorm,
            ) = attention.ok_or_else(|| boostr::error::Error::ModelError {
                reason: "Block::with_ids requires MLA attention weights for an attention layer"
                    .into(),
            })?;

            BlockType::Attention {
                attn: Box::new(MultiHeadLatentAttention::with_ids(
                    cfg,
                    q_a_proj,
                    q_b_proj,
                    q_a_layernorm,
                    kv_a_proj_with_mqa,
                    kv_a_layernorm,
                    kv_b_proj,
                    o_proj,
                    device,
                    frozen,
                )?),
                ffn: FfnLayer::with_ids(cfg, mlp, moe, frozen)?,
                input_layernorm: Self::rms_norm_with_id(
                    "input_layernorm.weight",
                    input_layernorm.0,
                    input_layernorm.1,
                    cfg.hidden_size,
                    eps,
                    frozen,
                )?,
                post_attention_layernorm: Self::rms_norm_with_id(
                    "post_attention_layernorm.weight",
                    post_attention_layernorm.0,
                    post_attention_layernorm.1,
                    cfg.hidden_size,
                    eps,
                    frozen,
                )?,
            }
        } else {
            let (
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                rope_cos,
                rope_sin,
                gate_proj,
                up_proj,
                down_proj,
                input_layernorm,
                post_attention_layernorm,
            ) = legacy_attention.ok_or_else(|| boostr::error::Error::ModelError {
                reason: "Block::with_ids requires legacy attention weights for a legacy layer"
                    .into(),
            })?;

            BlockType::LegacyAttention {
                attn: Box::new(CausalSelfAttention::with_ids(
                    cfg, q_proj, k_proj, v_proj, o_proj, rope_cos, rope_sin, frozen,
                )?),
                mlp: Box::new(Mlp::with_ids(cfg, gate_proj, up_proj, down_proj, frozen)?),
                input_layernorm: Self::rms_norm_with_id(
                    "input_layernorm.weight",
                    input_layernorm.0,
                    input_layernorm.1,
                    cfg.hidden_size,
                    eps,
                    frozen,
                )?,
                post_attention_layernorm: Self::rms_norm_with_id(
                    "post_attention_layernorm.weight",
                    post_attention_layernorm.0,
                    post_attention_layernorm.1,
                    cfg.hidden_size,
                    eps,
                    frozen,
                )?,
            }
        };

        Ok(Self { block_type })
    }

    fn forward<C>(&self, client: &C, x: &boostr::autograd::Var<R>) -> Result<BlockOutput<R>>
    where
        // `BinaryOps` is carried for `CausalSelfAttention::forward`, whose
        // projections may be LoRA-adapted.
        C: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::ConvOps<R>
            + boostr::QuantMatmulOps<R>
            + boostr::RoPEOps<R>
            + boostr::TypeConversionOps<R>,
        R::Client: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::ConvOps<R>
            + boostr::FlashAttentionOps<R>
            + boostr::TypeConversionOps<R>,
    {
        match &self.block_type {
            BlockType::Mamba { block, norm } => {
                // Mamba1: single layer with normalization and residual
                let residual = x;
                let x = norm.forward(client, x)?;
                let x = block.forward(client, &x)?;
                let x = boostr::autograd::var_add(&x, residual, client)
                    .map_err(boostr::error::Error::Numr)?;

                Ok(BlockOutput {
                    hidden: x,
                    aux_loss: None,
                    z_loss: None,
                })
            }
            BlockType::Mamba2 { block, norm } => {
                // Mamba2 delegates SSM math to boostr and keeps the block
                // pre-norm and residual here: pre-norm -> mixer -> + residual.
                let residual = x;
                let x = norm.forward(client, x)?;
                let x = block.forward(client, &x)?;
                let x = boostr::autograd::var_add(&x, residual, client)
                    .map_err(boostr::error::Error::Numr)?;

                Self::debug_activation("Mamba2 output", &x)?;

                Ok(BlockOutput {
                    hidden: x,
                    aux_loss: None,
                    z_loss: None,
                })
            }
            BlockType::Mamba3 { block, norm } => {
                // Mamba3 delegates SSM math to boostr and keeps the block
                // pre-norm and residual here: pre-norm -> mixer -> + residual.
                let residual = x;
                let x = norm.forward(client, x)?;
                let x = block.forward(client, &x)?;
                let x = boostr::autograd::var_add(&x, residual, client)
                    .map_err(boostr::error::Error::Numr)?;

                Ok(BlockOutput {
                    hidden: x,
                    aux_loss: None,
                    z_loss: None,
                })
            }
            BlockType::Attention {
                attn,
                ffn,
                input_layernorm,
                post_attention_layernorm,
            } => {
                // MLA + MoE architecture
                // Attention sub-block
                let residual = x;
                let x = input_layernorm.forward(client, x)?;
                LitGPT::check_nan(x.tensor(), "attn_input_norm")?;

                let x = attn.forward(client, &x)?;
                LitGPT::check_nan(x.tensor(), "attn_output")?;

                let x = boostr::autograd::var_add(&x, residual, client)
                    .map_err(boostr::error::Error::Numr)?;
                LitGPT::check_nan(x.tensor(), "attn_residual")?;

                // MoE sub-block
                let residual = &x;
                let x = post_attention_layernorm.forward(client, &x)?;
                LitGPT::check_nan(x.tensor(), "moe_input_norm")?;

                // MoE forward returns (output, load_balance_loss, z_loss)
                let (x, moe_loss, z_loss) = ffn.forward(client, &x)?;
                LitGPT::check_nan(x.tensor(), "moe_output")?;

                let x = boostr::autograd::var_add(&x, residual, client)
                    .map_err(boostr::error::Error::Numr)?;
                LitGPT::check_nan(x.tensor(), "moe_residual")?;

                Self::debug_activation("MLA+MoE output", &x)?;

                Ok(BlockOutput {
                    hidden: x,
                    aux_loss: Some(moe_loss),
                    z_loss: Some(z_loss),
                })
            }
            BlockType::LegacyAttention {
                attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            } => {
                // Legacy GQA + MLP (original nano)
                let residual = x;
                let x = input_layernorm.forward(client, x)?;
                let x = attn.forward(client, &x)?;
                let x = boostr::autograd::var_add(&x, residual, client)
                    .map_err(boostr::error::Error::Numr)?;

                let residual = &x;
                let x = post_attention_layernorm.forward(client, &x)?;
                let x = mlp.forward(client, &x)?;
                let x = boostr::autograd::var_add(&x, residual, client)
                    .map_err(boostr::error::Error::Numr)?;

                Ok(BlockOutput {
                    hidden: x,
                    aux_loss: None,
                    z_loss: None,
                })
            }
        }
    }

    fn trainable_parameters(&self) -> Vec<(boostr::autograd::TensorId, &boostr::autograd::Var<R>)> {
        let mut params = Vec::new();
        match &self.block_type {
            BlockType::Mamba { block, norm } => {
                params.extend(norm.trainable_parameters());
                params.extend(block.trainable_parameters());
            }
            BlockType::Mamba2 { block, norm } => {
                params.extend(norm.trainable_parameters());
                params.extend(block.trainable_parameters());
            }
            BlockType::Mamba3 { block, norm } => {
                params.extend(norm.trainable_parameters());
                params.extend(block.trainable_parameters());
            }
            BlockType::Attention {
                attn,
                ffn,
                input_layernorm,
                post_attention_layernorm,
            } => {
                params.extend(input_layernorm.trainable_parameters());
                params.extend(attn.trainable_parameters());
                params.extend(post_attention_layernorm.trainable_parameters());
                params.extend(ffn.trainable_parameters());
            }
            BlockType::LegacyAttention {
                attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            } => {
                params.extend(input_layernorm.trainable_parameters());
                params.extend(attn.trainable_parameters());
                params.extend(post_attention_layernorm.trainable_parameters());
                params.extend(mlp.trainable_parameters());
            }
        }
        params
    }

    /// Build a `Mamba1Config` for a bare-Mamba1 layer.
    ///
    /// Unreachable via `OxidizrModelConfig::to_boostr`: `ssm` on boostr's
    /// `ModelConfig` carries mandatory Mamba2-shaped dims (`num_heads`,
    /// `head_dim`, `state_size`, `chunk_size`) whenever it is present at
    /// all — there is no "SSM present but incompletely configured" state
    /// like flat `OxidizrModelConfig` had, where a missing
    /// `mamba2_num_heads` meant "this is Mamba1". So `is_mamba` (now
    /// `cfg.is_ssm_layer`) already implies `cfg.ssm.is_some()`, which means
    /// `use_mamba2` is always true whenever this branch would be taken —
    /// the `is_mamba && !use_mamba2` case that calls this function can
    /// never be populated by a converted config. Kept for call-site
    /// completeness; errors loudly instead of fabricating a sourceless
    /// config.
    fn mamba1_config(cfg: &ModelConfig) -> Result<boostr::model::Mamba1Config> {
        let ssm = cfg
            .ssm
            .as_ref()
            .ok_or_else(|| boostr::error::Error::ModelError {
                reason: "mamba1_config: ssm config missing while building an SSM layer".into(),
            })?;
        // Mamba1 carries no head decomposition, so `num_heads`/`head_dim` are
        // not read here — the inner width follows from `hidden_size * expand`.
        Ok(boostr::model::Mamba1Config::new(cfg.hidden_size)
            .with_d_state(ssm.state_size)
            .with_expand(ssm.expand)
            .with_d_conv(ssm.conv_kernel))
    }

    fn mamba2_config(cfg: &ModelConfig) -> Result<boostr::model::Mamba2Config> {
        let ssm = cfg
            .ssm
            .as_ref()
            .ok_or_else(|| boostr::error::Error::ModelError {
                reason: "ssm config missing while building a Mamba2 layer".into(),
            })?;
        let config = boostr::model::Mamba2Config {
            d_model: cfg.hidden_size,
            d_state: ssm.state_size,
            nheads: ssm.num_heads,
            headdim: ssm.head_dim,
            expand: ssm.expand,
            ngroups: ssm.n_groups,
            d_conv: ssm.conv_kernel,
            chunk_size: ssm.chunk_size,
            use_dt_bias: true,
            use_d: true,
            dt_softplus: true,
        };
        config.validate()?;
        Ok(config)
    }

    fn mamba3_config(cfg: &ModelConfig) -> Result<boostr::model::Mamba3Config> {
        let ssm = cfg
            .ssm
            .as_ref()
            .ok_or_else(|| boostr::error::Error::ModelError {
                reason: "ssm config missing while building a Mamba3 layer".into(),
            })?;
        let config = boostr::model::Mamba3Config {
            d_model: cfg.hidden_size,
            d_state: ssm.state_size,
            nheads: ssm.num_heads,
            headdim: ssm.head_dim,
            expand: ssm.expand,
            ngroups: ssm.n_groups,
            d_conv: ssm.conv_kernel,
            chunk_size: ssm.chunk_size,
            dt_softplus: true,
            use_d: true,
            use_dt_bias: true,
            use_complex_rope: ssm.complex_rope.unwrap_or(true),
            mimo_rank: ssm.mimo_rank.unwrap_or(0),
            use_conv: ssm.use_conv.unwrap_or(false),
            time_step_min: 0.001,
            time_step_max: 4.0,
        };
        config.validate()?;
        Ok(config)
    }

    fn take_rms_norm<C>(
        mut vb: VarBuilder<'_, R>,
        expected_len: usize,
        eps: f32,
        dtype: DType,
        client: &C,
    ) -> Result<RmsNorm<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let param_name = Self::param_name(&vb, "weight");
        let weight = take_or_init_tensor(
            &mut vb,
            "weight",
            &[expected_len],
            dtype,
            Init::Ones,
            client,
        )?;
        let weight_id = weight.id();
        Self::ensure_shape(&param_name, &weight, &[expected_len])?;
        Ok(RmsNorm::with_id(
            weight,
            weight_id,
            eps,
            INITIALIZED_TRAINABLE,
        ))
    }

    fn layer_norm_with_ids(
        name: &str,
        weight: (Tensor<R>, boostr::autograd::TensorId),
        bias: (Tensor<R>, boostr::autograd::TensorId),
        expected_len: usize,
        eps: f32,
        frozen: &FrozenParams,
    ) -> Result<boostr::nn::LayerNorm<R>> {
        let weight_name = format!("{name}.weight");
        let bias_name = format!("{name}.bias");
        Self::ensure_shape(&weight_name, &weight.0, &[expected_len])?;
        Self::ensure_shape(&bias_name, &bias.0, &[expected_len])?;
        // boostr's LayerNorm takes one flag for weight and bias together, so
        // both must be frozen or neither.
        let trainable = frozen.group_trainable(
            &format!("{name} (LayerNorm)"),
            &[
                (weight_name.as_str(), weight.1),
                (bias_name.as_str(), bias.1),
            ],
        )?;
        Ok(boostr::nn::LayerNorm::with_ids(
            weight.0, weight.1, bias.0, bias.1, eps, trainable,
        ))
    }

    fn take_layer_norm<C>(
        mut vb: VarBuilder<'_, R>,
        expected_len: usize,
        eps: f32,
        dtype: DType,
        client: &C,
    ) -> Result<boostr::nn::LayerNorm<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let weight_name = Self::param_name(&vb, "weight");
        let weight = take_or_init_tensor(
            &mut vb,
            "weight",
            &[expected_len],
            dtype,
            Init::Ones,
            client,
        )?;
        let weight_id = weight.id();
        Self::ensure_shape(&weight_name, &weight, &[expected_len])?;

        let bias_name = Self::param_name(&vb, "bias");
        let bias =
            take_or_init_tensor(&mut vb, "bias", &[expected_len], dtype, Init::Zeros, client)?;
        let bias_id = bias.id();
        Self::ensure_shape(&bias_name, &bias, &[expected_len])?;

        Ok(boostr::nn::LayerNorm::with_ids(
            weight,
            weight_id,
            bias,
            bias_id,
            eps,
            INITIALIZED_TRAINABLE,
        ))
    }

    fn rms_norm_with_id(
        name: &str,
        weight: Tensor<R>,
        weight_id: boostr::autograd::TensorId,
        expected_len: usize,
        eps: f32,
        frozen: &FrozenParams,
    ) -> Result<RmsNorm<R>> {
        Self::ensure_shape(name, &weight, &[expected_len])?;
        let trainable = frozen.trainable(weight_id);
        Ok(RmsNorm::with_id(weight, weight_id, eps, trainable))
    }

    fn ensure_shape(name: &str, weight: &Tensor<R>, expected_shape: &[usize]) -> Result<()> {
        if weight.shape() != expected_shape {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "shape mismatch for '{name}': expected {:?}, got {:?}",
                    expected_shape,
                    weight.shape()
                ),
            });
        }
        Ok(())
    }

    fn param_name(vb: &VarBuilder<'_, R>, leaf: &str) -> String {
        if vb.prefix().is_empty() {
            leaf.to_string()
        } else {
            format!("{}.{}", vb.prefix(), leaf)
        }
    }

    fn debug_activation(label: &str, x: &boostr::autograd::Var<R>) -> Result<()> {
        if std::env::var("DEBUG_ACTIVATION").is_ok() {
            let vals = x.tensor().contiguous()?.to_vec::<f32>();
            if !vals.is_empty() {
                let rms = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32).sqrt();
                let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                eprintln!("[{label}] rms={:.6}, max_abs={:.6}", rms, max_abs);
            }
        }
        Ok(())
    }
}

// --- Main LitGPT Model ---

pub struct ModelOutput<R: boostr::Runtime> {
    pub logits: boostr::autograd::Var<R>,
    pub aux_losses: Vec<boostr::autograd::Var<R>>, // Auxiliary losses (MoE load balancing)
    pub z_losses: Vec<boostr::autograd::Var<R>>,   // Router z-losses (prevents MoE logit explosion)
}

pub struct LitGPT<R: boostr::Runtime> {
    cfg: OxidizrModelConfig,
    embed_tokens: Embedding<R>,
    layers: Vec<Block<R>>,
    norm: RmsNorm<R>,
    lm_head: Linear<R>,
    /// Model dtype (F32/F16/BF16 as configured)
    /// RmsNorm and SSM computations use F32 internally for numerical stability
    #[allow(dead_code)]
    compute_dtype: DType,
    /// Parameters that must stay frozen. Carried on the model and handed to
    /// every `*_with_ids` constructor, so `rebuild_with_parameters` — which the
    /// trainer calls every step — reconstructs each parameter with the same
    /// trainability it was built with.
    frozen: FrozenParams,
}

/// Conv1d hyperparameters needed to rebuild a `Conv1d` from named tensors.
/// Grouped so `conv1d_from_named` doesn't need to take them as four separate
/// arguments.
struct Conv1dHyperParams {
    stride: usize,
    padding: boostr::PaddingMode,
    dilation: usize,
    groups: usize,
}

impl<R> LitGPT<R>
where
    R: boostr::Runtime<DType = DType>,
{
    pub fn new<C>(cfg: &OxidizrModelConfig, vb: &mut VarBuilder<'_, R>, client: &C) -> Result<Self>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let bcfg = cfg.to_boostr()?;
        let compute_dtype = cfg.dtype.to_dtype();
        let embed = Self::take_embedding(
            vb.pp("embed_tokens"),
            cfg.vocab_size,
            cfg.hidden_size,
            compute_dtype,
            bcfg.tie_word_embeddings,
            client,
        )?;

        // DEBUG: Log embedding init
        if std::env::var("DEBUG_INIT").is_ok() {
            let embed_tensor = embed.weight().tensor().contiguous()?;
            let v = embed_tensor.to_vec::<f32>();
            let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
            let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "[EMBED INIT] embed_tokens: rms={:.6e}, max={:.6e}, num_tokens={}",
                rms, max_abs, cfg.vocab_size
            );
        }

        let mut layers = Vec::new();
        for i in 0..cfg.num_layers {
            layers.push(Block::new(
                &bcfg,
                i,
                vb.pp(&format!("layers.{}", i)),
                client,
                compute_dtype,
                cfg.lora.as_ref(),
            )?);
        }

        let norm = Self::take_rms_norm(
            vb.pp("norm"),
            cfg.hidden_size,
            bcfg.rms_norm_eps as f32,
            compute_dtype,
            client,
        )?;
        // A tied checkpoint (e.g. Qwen3-1.7B) has no separate `lm_head.weight`
        // tensor, so loading one here would hard-error. `with_ids` re-derives
        // the tuple from `embed_tokens` when `cfg.tie_word_embeddings` is set,
        // so this one is a placeholder that gets discarded there.
        let lm_head_tuple = if cfg.tie_word_embeddings {
            (embed.weight().tensor().clone(), embed.weight().id())
        } else {
            let lm_head = Self::take_linear_no_bias(
                vb.pp("lm_head"),
                &[cfg.vocab_size, cfg.hidden_size],
                compute_dtype,
                client,
            )?;
            (lm_head.weight().tensor().clone(), lm_head.weight().id())
        };

        let model = Self::with_ids(
            cfg,
            (embed.weight().tensor().clone(), embed.weight().id()),
            layers,
            (norm.weight().tensor().clone(), norm.weight().id()),
            lm_head_tuple,
            // Every id above was minted by `take_*` in this call, so none of
            // them can be frozen yet. See `INITIALIZED_TRAINABLE`.
            FrozenParams::new(),
        )?;

        // LoRA trains the adapter, not the projection it adapts. This is the
        // first point at which the wrapped base ids EXIST, and routing the
        // freeze through `FrozenParams` is what makes it survive every later
        // `rebuild_with_parameters` — the trainer calls that once per step.
        let frozen = model.lora_frozen_base_ids();
        if frozen.is_empty() {
            return Ok(model);
        }
        model.with_frozen_parameters(frozen)
    }

    /// The base-weight ids of every LoRA-adapted projection, attention and MLP
    /// alike.
    ///
    /// Empty when no projection carries an adapter, which is what keeps a
    /// non-LoRA model's parameter set byte-identical to before LoRA existed.
    fn lora_frozen_base_ids(&self) -> FrozenParams {
        let mut frozen = FrozenParams::new();
        let mut freeze_adapted = |proj: &MaybeLora<R>| {
            if proj.adapters().is_some() {
                frozen.freeze(proj.weight().id());
            }
        };
        for layer in &self.layers {
            match &layer.block_type {
                BlockType::LegacyAttention { attn, mlp, .. } => {
                    for proj in [&attn.q_proj, &attn.k_proj, &attn.v_proj, &attn.o_proj] {
                        freeze_adapted(proj);
                    }
                    for proj in mlp.projections() {
                        freeze_adapted(proj);
                    }
                }
                // The MLA path's MLP shares `Mlp`, so an adapter reaching it
                // must freeze its base here too rather than train both halves.
                BlockType::Attention {
                    ffn: FfnLayer::Mlp(mlp),
                    ..
                } => {
                    for proj in mlp.projections() {
                        freeze_adapted(proj);
                    }
                }
                _ => {}
            }
        }
        frozen
    }

    fn with_ids(
        cfg: &OxidizrModelConfig,
        embed_tokens: (Tensor<R>, boostr::autograd::TensorId),
        layers: Vec<Block<R>>,
        norm: (Tensor<R>, boostr::autograd::TensorId),
        lm_head: (Tensor<R>, boostr::autograd::TensorId),
        frozen: FrozenParams,
    ) -> Result<Self> {
        // Tied checkpoints (e.g. Qwen3-1.7B) have no independent
        // `lm_head.weight`: re-derive the tuple from `embed_tokens` here,
        // on every call, not just at init. `rebuild_with_parameters` also
        // routes through this, so a rebuild re-aliases lm_head from the
        // optimizer-updated embedding rather than reading a stale separate
        // entry — both share the same `TensorId` across steps, which is what
        // keeps `assert_stable_parameter_ids` happy and the gradient tied
        // (`dL/dW = dL/dW_embed + dL/dW_lm_head` accumulates into one bucket,
        // and the optimizer's `TensorId`-keyed state sees one parameter).
        let lm_head = if cfg.tie_word_embeddings {
            (embed_tokens.0.clone(), embed_tokens.1)
        } else {
            lm_head
        };

        Self::ensure_shape(
            "embed_tokens.weight",
            &embed_tokens.0,
            &[cfg.vocab_size, cfg.hidden_size],
        )?;
        Self::ensure_shape("norm.weight", &norm.0, &[cfg.hidden_size])?;
        Self::ensure_shape(
            "lm_head.weight",
            &lm_head.0,
            &[cfg.vocab_size, cfg.hidden_size],
        )?;
        if layers.len() != cfg.num_layers {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "LitGPT::with_ids expected {} layers from cfg.num_layers, got {}",
                    cfg.num_layers,
                    layers.len()
                ),
            });
        }

        let embed_trainable = frozen.trainable(embed_tokens.1);
        let norm_trainable = frozen.trainable(norm.1);
        let lm_head_trainable = frozen.trainable(lm_head.1);

        Ok(Self {
            cfg: cfg.clone(),
            embed_tokens: Embedding::with_id(embed_tokens.0, embed_tokens.1, embed_trainable),
            layers,
            // boostr's default rather than a bare literal. oxidizr's YAML has no
            // `rms_norm_eps` field yet, so `to_boostr()` always yields the default
            // here — calling it keeps this in step with boostr if that default
            // ever changes, and marks the one place to read the config instead
            // once the field is surfaced.
            norm: RmsNorm::with_id(
                norm.0,
                norm.1,
                default_rms_norm_eps() as f32,
                norm_trainable,
            ),
            lm_head: Linear::with_ids(lm_head.0, lm_head.1, None, lm_head_trainable),
            compute_dtype: cfg.dtype.to_dtype(),
            frozen,
        })
    }

    pub fn rebuild_with_parameters(
        &self,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
    ) -> Result<Self> {
        // `self.frozen`, never a fresh set: this is the call the training loop
        // makes every step, and re-deriving trainability here is exactly how a
        // frozen parameter would silently come back trainable.
        self.rebuild(params, self.frozen.clone())
    }

    /// The parameters this model keeps frozen.
    #[allow(dead_code)]
    pub fn frozen_parameters(&self) -> &FrozenParams {
        &self.frozen
    }

    /// Rebuild the model with a new frozen set, carrying every current tensor
    /// value through unchanged.
    ///
    /// This is the only way to freeze a parameter: boostr's layers expose no
    /// mutable access to their `Var`s, so trainability is fixed at
    /// construction. The returned model keeps `frozen`, so every subsequent
    /// `rebuild_with_parameters` honours it too.
    #[allow(dead_code)]
    pub fn with_frozen_parameters(&self, frozen: FrozenParams) -> Result<Self> {
        // Every currently trainable parameter must appear in the map:
        // `tensor_with_stable_id` treats a missing trainable parameter as the
        // optimizer having lost it, and that check stays in force here.
        let params: std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>> = self
            .trainable_parameters()
            .into_iter()
            .map(|(id, var)| (id, var.tensor().clone()))
            .collect();
        self.rebuild(&params, frozen)
    }

    fn rebuild(
        &self,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
        frozen: FrozenParams,
    ) -> Result<Self> {
        let device = self.embed_tokens.weight().tensor().device();
        let mut layers = Vec::with_capacity(self.layers.len());
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            layers.push(Self::rebuild_block_with_parameters(
                &self.cfg, layer_idx, layer, params, device, &frozen,
            )?);
        }

        Self::with_ids(
            &self.cfg,
            Self::tensor_with_stable_id(self.embed_tokens.weight(), params)?,
            layers,
            Self::tensor_with_stable_id(self.norm.weight(), params)?,
            Self::tensor_with_stable_id(self.lm_head.weight(), params)?,
            frozen,
        )
    }

    pub fn forward<C>(&self, client: &C, input_ids: &Tensor<R>) -> Result<ModelOutput<R>>
    where
        // `BinaryOps` is carried for LoRA-adapted attention projections; see
        // `CausalSelfAttention::forward`.
        C: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::ConvOps<R>
            + boostr::QuantMatmulOps<R>
            + boostr::RoPEOps<R>
            + boostr::IndexingOps<R>
            + boostr::NormalizationOps<R>
            + boostr::TypeConversionOps<R>,
        R::Client: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::ConvOps<R>
            + boostr::FlashAttentionOps<R>
            + boostr::IndexingOps<R>
            + boostr::TypeConversionOps<R>,
    {
        // Token IDs are integer indices, not differentiable inputs. The graph
        // starts at boostr's Embedding output and stays Var-based thereafter.

        // DEBUG: Check input_ids for duplicate tokens
        if std::env::var("DEBUG_GRAD").is_ok() {
            let ids_tensor = input_ids.contiguous()?;
            let ids = ids_tensor.to_vec::<u32>();
            let seq_len = input_ids.shape().last().copied().unwrap_or(0);
            eprintln!(
                "[Model forward] input_ids shape={:?}, first 10 tokens: {:?}",
                input_ids.shape(),
                &ids[..std::cmp::min(10, ids.len())]
            );
            // Check if all tokens are the same
            if seq_len > 0 && ids.windows(2).all(|w| w[0] == w[1]) {
                eprintln!(
                    "[Model forward] WARNING: All tokens are the same! token={}",
                    ids[0]
                );
            }
        }

        let mut x = self.embed_tokens.forward(client, input_ids)?;
        x = boostr::nn::var_contiguous(&x)?;

        // DEBUG: Check embedding output
        if std::env::var("DEBUG_GRAD").is_ok() {
            let x_tensor = x.tensor().contiguous()?;
            let x_vec = x_tensor.to_vec::<f32>();
            let hidden_size = x.shape().last().copied().unwrap_or(0);
            eprintln!(
                "[Model forward] embed_out shape={:?}, hidden_size={}",
                x.shape(),
                hidden_size
            );
            if hidden_size > 0 {
                let pos0 = &x_vec[0..std::cmp::min(3, hidden_size)];
                let pos2_start = 2 * hidden_size;
                let pos2 = if pos2_start + 3 <= x_vec.len() {
                    &x_vec[pos2_start..pos2_start + 3]
                } else {
                    &[]
                };
                eprintln!("[Model forward] embed_out pos0[:3] = {:?}", pos0);
                eprintln!("[Model forward] embed_out pos2[:3] = {:?}", pos2);
                if pos2.len() == 3 {
                    let diff: f32 = x_vec[0..hidden_size]
                        .iter()
                        .zip(&x_vec[pos2_start..pos2_start + hidden_size])
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    eprintln!(
                        "[Model forward] embed_out diff (pos0 vs pos2) = {:.6e}",
                        diff
                    );
                }
            }
        }

        // Debug: check for NaN after embedding
        Self::check_nan(x.tensor(), "embed_tokens")?;

        let mut aux_losses = Vec::new();
        let mut z_losses = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let output = layer.forward(client, &x)?;
            x = boostr::nn::var_contiguous(&output.hidden)?; // Force contiguous after each layer

            // Debug: check for NaN after each layer
            Self::check_nan(x.tensor(), &format!("layer_{}", i))?;

            // Collect MoE auxiliary losses
            if let Some(loss) = output.aux_loss {
                aux_losses.push(loss);
            }

            // Collect router z-losses (prevents softmax saturation in MoE)
            if let Some(z) = output.z_loss {
                z_losses.push(z);
            }
        }

        let x = self.norm.forward(client, &x)?;
        Self::check_nan(x.tensor(), "final_norm")?;

        let logits = self.lm_head.forward(client, &x)?;
        Self::check_nan(logits.tensor(), "lm_head")?;

        Ok(ModelOutput {
            logits,
            aux_losses,
            z_losses,
        })
    }

    pub fn trainable_parameters(
        &self,
    ) -> Vec<(boostr::autograd::TensorId, &boostr::autograd::Var<R>)> {
        let mut params = Vec::new();
        params.extend(self.embed_tokens.trainable_parameters());
        for layer in &self.layers {
            params.extend(layer.trainable_parameters());
        }
        params.extend(self.norm.trainable_parameters());
        // Tied: lm_head shares embed_tokens's TensorId, so listing it again
        // here would double-count one parameter under two entries.
        if !self.cfg.tie_word_embeddings {
            params.extend(self.lm_head.trainable_parameters());
        }
        params
    }

    /// Every parameter a checkpoint must carry to reconstruct this model,
    /// frozen ones included.
    ///
    /// This is the completeness contract for both save and load. It cannot be
    /// narrowed to the trainable set: `VarBuilder::take_or_init_tensor`
    /// random-initializes any name the varmap lacks, so a checkpoint missing a
    /// frozen base weight would "load" into numerical garbage.
    pub fn named_parameters(
        &self,
    ) -> Vec<(
        String,
        boostr::autograd::TensorId,
        &boostr::autograd::Var<R>,
    )> {
        let mut params = Vec::new();
        Self::push_named_param(
            &mut params,
            "embed_tokens.weight".to_string(),
            self.embed_tokens.weight(),
        );
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            Self::extend_block_named_trainable(&mut params, layer_idx, layer);
        }
        Self::push_named_param(&mut params, "norm.weight".to_string(), self.norm.weight());
        // Tied: lm_head shares embed_tokens's weight/TensorId, so it has no
        // independent entry. This also fixes save (no duplicate copy
        // written) and load (a tied checkpoint lacking `lm_head.weight`
        // no longer trips the missing-tensor check).
        if !self.cfg.tie_word_embeddings {
            Self::push_named_param(
                &mut params,
                "lm_head.weight".to_string(),
                self.lm_head.weight(),
            );
        }
        params
    }

    /// The subset of [`LitGPT::named_parameters`] the optimizer owns.
    pub fn named_trainable_parameters(
        &self,
    ) -> Vec<(
        String,
        boostr::autograd::TensorId,
        &boostr::autograd::Var<R>,
    )> {
        self.named_parameters()
            .into_iter()
            .filter(|(_, _, var)| var.requires_grad())
            .collect()
    }

    fn rebuild_block_with_parameters(
        cfg: &OxidizrModelConfig,
        layer_idx: usize,
        layer: &Block<R>,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
        device: &R::Device,
        frozen: &FrozenParams,
    ) -> Result<Block<R>> {
        let bcfg = cfg.to_boostr()?;
        match &layer.block_type {
            BlockType::Mamba { block, norm } => Block::with_ids(
                &bcfg,
                layer_idx,
                Some((
                    Self::mamba1_weights_with_ids(block, params, frozen)?,
                    Self::tensor_with_stable_id(norm.weight(), params)?,
                )),
                None,
                None,
                None,
                None,
                device,
                frozen,
            ),
            BlockType::Mamba2 { block, norm } => Block::with_ids(
                &bcfg,
                layer_idx,
                None,
                Some((
                    Self::mamba2_weights_with_ids(block, params, frozen)?,
                    Self::tensor_with_stable_id(norm.weight(), params)?,
                )),
                None,
                None,
                None,
                device,
                frozen,
            ),
            BlockType::Mamba3 { block, norm } => Block::with_ids(
                &bcfg,
                layer_idx,
                None,
                None,
                Some((
                    Self::mamba3_weights_with_ids(block, params, frozen)?,
                    Self::tensor_with_stable_id(norm.weight(), params)?,
                    Self::tensor_with_stable_id(norm.bias(), params)?,
                )),
                None,
                None,
                device,
                frozen,
            ),
            BlockType::Attention {
                attn,
                ffn,
                input_layernorm,
                post_attention_layernorm,
            } => {
                let (mlp, moe) = match ffn {
                    FfnLayer::Mlp(mlp) => {
                        (Some(Self::mlp_weights_with_ids(cfg, mlp, params)?), None)
                    }
                    FfnLayer::Moe(moe) => (None, Some(Self::moe_weights_with_ids(moe, params)?)),
                };
                let attn_named = attn.mla.named_parameters();
                Block::with_ids(
                    &bcfg,
                    layer_idx,
                    None,
                    None,
                    None,
                    Some((
                        Self::pair_from_named(&attn_named, params, "q_down.weight")?,
                        Self::pair_from_named(&attn_named, params, "q_up.weight")?,
                        Self::pair_from_named(&attn_named, params, "q_norm.weight")?,
                        Self::pair_from_named(&attn_named, params, "kv_compress.weight")?,
                        Self::pair_from_named(&attn_named, params, "kv_norm.weight")?,
                        Self::pair_from_named(&attn_named, params, "kv_decompress.weight")?,
                        Self::pair_from_named(&attn_named, params, "o_proj.weight")?,
                        mlp,
                        moe,
                        Self::tensor_with_stable_id(input_layernorm.weight(), params)?,
                        Self::tensor_with_stable_id(post_attention_layernorm.weight(), params)?,
                    )),
                    None,
                    device,
                    frozen,
                )
            }
            BlockType::LegacyAttention {
                attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            } => {
                let (gate_proj, up_proj, down_proj) = Self::mlp_weights_with_ids(cfg, mlp, params)?;
                Block::with_ids(
                    &bcfg,
                    layer_idx,
                    None,
                    None,
                    None,
                    None,
                    Some((
                        Self::attn_proj_with_ids(cfg, "q_proj", &attn.q_proj, params)?,
                        Self::attn_proj_with_ids(cfg, "k_proj", &attn.k_proj, params)?,
                        Self::attn_proj_with_ids(cfg, "v_proj", &attn.v_proj, params)?,
                        Self::attn_proj_with_ids(cfg, "o_proj", &attn.o_proj, params)?,
                        Self::tensor_with_stable_id(&attn.rope.cos, params)?,
                        Self::tensor_with_stable_id(&attn.rope.sin, params)?,
                        gate_proj,
                        up_proj,
                        down_proj,
                        Self::tensor_with_stable_id(input_layernorm.weight(), params)?,
                        Self::tensor_with_stable_id(post_attention_layernorm.weight(), params)?,
                    )),
                    device,
                    frozen,
                )
            }
        }
    }

    fn mamba1_weights_with_ids(
        block: &boostr::model::Mamba1<R>,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
        frozen: &FrozenParams,
    ) -> Result<boostr::model::mamba::Mamba1WeightsWithIds<R>> {
        let named = <boostr::model::Mamba1<R> as boostr::nn::Module<R>>::named_parameters(block);
        let config = block.config();
        Ok(boostr::model::mamba::Mamba1WeightsWithIds {
            in_proj: Self::linear_from_named(
                &named,
                params,
                "in_proj.weight",
                "in_proj.bias",
                frozen,
            )?,
            conv1d: Self::conv1d_from_named(
                &named,
                params,
                "conv1d.weight",
                "conv1d.bias",
                Self::conv1d_hyperparams_for(config.d_conv, config.conv_channels()),
                frozen,
            )?,
            x_proj: Self::linear_from_named(
                &named,
                params,
                "x_proj.weight",
                "x_proj.bias",
                frozen,
            )?,
            dt_proj: Self::linear_from_named(
                &named,
                params,
                "dt_proj.weight",
                "dt_proj.bias",
                frozen,
            )?,
            out_proj: Self::linear_from_named(
                &named,
                params,
                "out_proj.weight",
                "out_proj.bias",
                frozen,
            )?,
            a_log: Self::pair_from_named(&named, params, "a_log")?,
            d_param: if config.use_d {
                Some(Self::pair_from_named(&named, params, "d_param")?)
            } else {
                None
            },
        })
    }

    fn mamba2_weights_with_ids(
        block: &boostr::model::Mamba2<R>,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
        frozen: &FrozenParams,
    ) -> Result<boostr::model::mamba::Mamba2WeightsWithIds<R>> {
        let named = <boostr::model::Mamba2<R> as boostr::nn::Module<R>>::named_parameters(block);
        let config = block.config();
        let norm =
            if let Some(norm) = Self::optional_pair_from_named(&named, params, "norm.weight")? {
                let trainable = frozen.trainable(norm.1);
                Some(RmsNorm::with_id(norm.0, norm.1, 1e-5, trainable))
            } else {
                None
            };

        Ok(boostr::model::mamba::Mamba2WeightsWithIds {
            in_proj: Self::linear_from_named(
                &named,
                params,
                "in_proj.weight",
                "in_proj.bias",
                frozen,
            )?,
            conv1d: Self::conv1d_from_named(
                &named,
                params,
                "conv1d.weight",
                "conv1d.bias",
                Self::conv1d_hyperparams_for(config.d_conv, config.conv_channels()),
                frozen,
            )?,
            out_proj: Self::linear_from_named(
                &named,
                params,
                "out_proj.weight",
                "out_proj.bias",
                frozen,
            )?,
            a_log: Self::pair_from_named(&named, params, "a_log")?,
            dt_bias: Self::optional_pair_from_named(&named, params, "dt_bias")?,
            d_param: Self::optional_pair_from_named(&named, params, "d_param")?,
            norm,
        })
    }

    fn mamba3_weights_with_ids(
        block: &boostr::model::Mamba3<R>,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
        frozen: &FrozenParams,
    ) -> Result<boostr::model::mamba::Mamba3WeightsWithIds<R>> {
        let named = <boostr::model::Mamba3<R> as boostr::nn::Module<R>>::named_parameters(block);
        let config = block.config();
        let theta_proj = if config.use_complex_rope
            || Self::optional_named_var(&named, "theta_proj.weight").is_some()
        {
            Some(Self::linear_from_named(
                &named,
                params,
                "theta_proj.weight",
                "theta_proj.bias",
                frozen,
            )?)
        } else {
            None
        };
        let conv1d = if config.use_conv {
            Some(Self::conv1d_from_named(
                &named,
                params,
                "conv1d.weight",
                "conv1d.bias",
                Self::conv1d_hyperparams_for(config.d_conv, config.conv_channels()),
                frozen,
            )?)
        } else {
            None
        };
        let (mimo_x_up, mimo_x_down) = if config.mimo_rank > 0 {
            (
                Some(Self::linear_from_named(
                    &named,
                    params,
                    "mimo_x_up.weight",
                    "mimo_x_up.bias",
                    frozen,
                )?),
                Some(Self::linear_from_named(
                    &named,
                    params,
                    "mimo_x_down.weight",
                    "mimo_x_down.bias",
                    frozen,
                )?),
            )
        } else {
            (None, None)
        };
        let bc_norm = Self::pair_from_named(&named, params, "bc_norm.weight")?;
        let norm = Self::pair_from_named(&named, params, "norm.weight")?;
        let bc_norm_trainable = frozen.trainable(bc_norm.1);
        let norm_trainable = frozen.trainable(norm.1);

        Ok(boostr::model::mamba::Mamba3WeightsWithIds {
            in_proj: Self::linear_from_named(
                &named,
                params,
                "in_proj.weight",
                "in_proj.bias",
                frozen,
            )?,
            out_proj: Self::linear_from_named(
                &named,
                params,
                "out_proj.weight",
                "out_proj.bias",
                frozen,
            )?,
            lambda_proj: Self::linear_from_named(
                &named,
                params,
                "lambda_proj.weight",
                "lambda_proj.bias",
                frozen,
            )?,
            theta_proj,
            b_bias: Self::pair_from_named(&named, params, "b_bias")?,
            c_bias: Self::pair_from_named(&named, params, "c_bias")?,
            dt_bias: if config.use_dt_bias {
                Some(Self::pair_from_named(&named, params, "dt_bias")?)
            } else {
                None
            },
            a_log: Self::pair_from_named(&named, params, "a_log")?,
            d_param: if config.use_d {
                Some(Self::pair_from_named(&named, params, "d_param")?)
            } else {
                None
            },
            bc_norm: RmsNorm::with_id(bc_norm.0, bc_norm.1, 1e-6, bc_norm_trainable),
            norm: RmsNorm::with_id(norm.0, norm.1, 1e-6, norm_trainable),
            conv1d,
            mimo_x_up,
            mimo_x_down,
        })
    }

    /// Snapshot the MLP's three projections for the rebuild path, adapters
    /// included. Routing through `attn_proj_with_ids` is what carries every
    /// `lora_a`/`lora_b` id across a step instead of minting a fresh one.
    fn mlp_weights_with_ids(
        cfg: &OxidizrModelConfig,
        mlp: &Mlp<R>,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
    ) -> Result<(AttnProjWithIds<R>, AttnProjWithIds<R>, AttnProjWithIds<R>)> {
        Ok((
            Self::attn_proj_with_ids(cfg, "gate_proj", &mlp.gate_proj, params)?,
            Self::attn_proj_with_ids(cfg, "up_proj", &mlp.up_proj, params)?,
            Self::attn_proj_with_ids(cfg, "down_proj", &mlp.down_proj, params)?,
        ))
    }

    #[allow(clippy::type_complexity)]
    fn moe_weights_with_ids(
        moe: &MoeLayer<R>,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
    ) -> Result<(
        Vec<(
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
        )>,
        (
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
        ),
        (Tensor<R>, boostr::autograd::TensorId),
    )> {
        let mut experts = Vec::new();
        for expert in moe.moe.experts() {
            experts.push((
                Self::tensor_with_stable_id(expert.gate_proj().weight(), params)?,
                Self::tensor_with_stable_id(expert.up_proj().weight(), params)?,
                Self::tensor_with_stable_id(expert.down_proj().weight(), params)?,
            ));
        }
        let shared_expert =
            moe.moe
                .shared_expert()
                .ok_or_else(|| boostr::error::Error::ModelError {
                    reason: "MoE layer is missing the shared expert required by oxidizr".into(),
                })?;
        let shared_expert = (
            Self::tensor_with_stable_id(shared_expert.gate_proj().weight(), params)?,
            Self::tensor_with_stable_id(shared_expert.up_proj().weight(), params)?,
            Self::tensor_with_stable_id(shared_expert.down_proj().weight(), params)?,
        );
        let router_gate = Self::tensor_with_stable_id(moe.moe.router().gate().weight(), params)?;
        Ok((experts, shared_expert, router_gate))
    }

    fn conv1d_hyperparams_for(config_d_conv: usize, conv_channels: usize) -> Conv1dHyperParams {
        Conv1dHyperParams {
            stride: 1,
            padding: boostr::PaddingMode::Custom(config_d_conv - 1, 0, 0, 0),
            dilation: 1,
            groups: conv_channels,
        }
    }

    fn linear_from_named(
        named: &[(String, &boostr::autograd::Var<R>)],
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
        weight_name: &str,
        bias_name: &str,
        frozen: &FrozenParams,
    ) -> Result<Linear<R>> {
        let weight = Self::pair_from_named(named, params, weight_name)?;
        let bias = Self::optional_pair_from_named(named, params, bias_name)?;
        // boostr's Linear takes one flag for weight and bias together.
        let trainable = Self::weight_and_bias_trainable(
            frozen,
            weight_name,
            weight.1,
            bias_name,
            bias.as_ref().map(|(_, id)| *id),
        )?;
        Ok(Linear::with_ids(weight.0, weight.1, bias, trainable))
    }

    fn conv1d_from_named(
        named: &[(String, &boostr::autograd::Var<R>)],
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
        weight_name: &str,
        bias_name: &str,
        conv: Conv1dHyperParams,
        frozen: &FrozenParams,
    ) -> Result<boostr::nn::Conv1d<R>> {
        let weight = Self::pair_from_named(named, params, weight_name)?;
        let bias = Self::optional_pair_from_named(named, params, bias_name)?;
        // boostr's Conv1d takes one flag for weight and bias together.
        let trainable = Self::weight_and_bias_trainable(
            frozen,
            weight_name,
            weight.1,
            bias_name,
            bias.as_ref().map(|(_, id)| *id),
        )?;
        Ok(boostr::nn::Conv1d::with_ids(
            weight.0,
            weight.1,
            bias,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            trainable,
        ))
    }

    /// Trainability for a `Linear`/`Conv1d` whose weight and bias share one
    /// flag in boostr's API.
    fn weight_and_bias_trainable(
        frozen: &FrozenParams,
        weight_name: &str,
        weight_id: boostr::autograd::TensorId,
        bias_name: &str,
        bias_id: Option<boostr::autograd::TensorId>,
    ) -> Result<bool> {
        match bias_id {
            None => Ok(frozen.trainable(weight_id)),
            Some(bias_id) => frozen.group_trainable(
                weight_name,
                &[(weight_name, weight_id), (bias_name, bias_id)],
            ),
        }
    }

    fn pair_from_named(
        named: &[(String, &boostr::autograd::Var<R>)],
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
        name: &str,
    ) -> Result<(Tensor<R>, boostr::autograd::TensorId)> {
        let var = Self::find_named_var(named, name)?;
        Self::tensor_with_stable_id(var, params)
    }

    fn optional_pair_from_named(
        named: &[(String, &boostr::autograd::Var<R>)],
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
        name: &str,
    ) -> Result<Option<(Tensor<R>, boostr::autograd::TensorId)>> {
        Self::optional_named_var(named, name)
            .map(|var| Self::tensor_with_stable_id(var, params))
            .transpose()
    }

    fn find_named_var<'a>(
        named: &'a [(String, &'a boostr::autograd::Var<R>)],
        name: &str,
    ) -> Result<&'a boostr::autograd::Var<R>> {
        Self::optional_named_var(named, name).ok_or_else(|| boostr::error::Error::ModelError {
            reason: format!("missing parameter '{name}' while rebuilding LitGPT"),
        })
    }

    /// The slice borrow is independent of `'a`: the returned reference borrows
    /// from the model, not from the temporary Vec the caller built. Unifying the
    /// two lifetimes would force that local Vec to outlive the call.
    fn optional_named_var<'a>(
        named: &[(String, &'a boostr::autograd::Var<R>)],
        name: &str,
    ) -> Option<&'a boostr::autograd::Var<R>> {
        named
            .iter()
            .find_map(|(param_name, var)| (param_name == name).then_some(*var))
    }

    /// Snapshot one optionally-adapted projection for the rebuild path.
    ///
    /// The adapter factors go through `tensor_with_stable_id` exactly like
    /// every other parameter, so the rebuilt layer reuses their ids instead of
    /// minting new ones — the difference between resuming a LoRA run and
    /// silently restarting it with detached optimizer state.
    fn attn_proj_with_ids(
        cfg: &OxidizrModelConfig,
        name: &str,
        proj: &MaybeLora<R>,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
    ) -> Result<AttnProjWithIds<R>> {
        let weight = Self::tensor_with_stable_id(proj.weight(), params)?;
        let Some((lora_a, lora_b)) = proj.adapters() else {
            return Ok(AttnProjWithIds { weight, lora: None });
        };
        let alpha = cfg.lora.as_ref().map(|lora| lora.alpha).ok_or_else(|| {
            boostr::error::Error::ModelError {
                reason: format!(
                    "'{name}' carries a LoRA adapter but the model config has no lora block, \
                     so the adapter's alpha cannot be recovered for the rebuild. Rebuild the \
                     model from the config that built it."
                ),
            }
        })?;
        Ok(AttnProjWithIds {
            weight,
            lora: Some(LoraWithIds {
                a: Self::tensor_with_stable_id(lora_a, params)?,
                b: Self::tensor_with_stable_id(lora_b, params)?,
                alpha,
            }),
        })
    }

    fn tensor_with_stable_id(
        var: &boostr::autograd::Var<R>,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
    ) -> Result<(Tensor<R>, boostr::autograd::TensorId)> {
        let id = var.id();
        match params.get(&id) {
            Some(tensor) => Ok((tensor.clone(), id)),
            None if var.requires_grad() => Err(boostr::error::Error::ModelError {
                reason: format!("missing trainable parameter {id} while rebuilding LitGPT"),
            }),
            None => Ok((var.tensor().clone(), id)),
        }
    }

    fn take_embedding<C>(
        mut vb: VarBuilder<'_, R>,
        vocab_size: usize,
        hidden_size: usize,
        dtype: DType,
        tied: bool,
        client: &C,
    ) -> Result<Embedding<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let param_name = Self::param_name(&vb, "weight");
        // A tied matrix is also the output projection, so `nn.Embedding`'s
        // N(0, 1) default is far too wide: logits come out enormous and the
        // first loss lands near 126 instead of ln(vocab) ~ 11.5. The
        // GPT-2/BERT truncated normal is the standard init for a matrix that
        // has to serve both roles. Untied embeddings keep PyTorch's default.
        let init = if tied {
            Init::TruncatedNormal {
                mean: 0.0,
                stdev: 0.02,
            }
        } else {
            Init::PyTorchEmbedding
        };
        let weight = take_or_init_tensor(
            &mut vb,
            "weight",
            &[vocab_size, hidden_size],
            dtype,
            init,
            client,
        )?;
        let weight_id = weight.id();
        Self::ensure_shape(&param_name, &weight, &[vocab_size, hidden_size])?;
        Ok(Embedding::with_id(weight, weight_id, INITIALIZED_TRAINABLE))
    }

    fn take_linear_no_bias<C>(
        mut vb: VarBuilder<'_, R>,
        expected_shape: &[usize],
        dtype: DType,
        client: &C,
    ) -> Result<Linear<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let param_name = Self::param_name(&vb, "weight");
        let weight = take_or_init_tensor(
            &mut vb,
            "weight",
            expected_shape,
            dtype,
            Init::PyTorchLinear,
            client,
        )?;
        let weight_id = weight.id();
        Self::ensure_shape(&param_name, &weight, expected_shape)?;
        Ok(Linear::with_ids(
            weight,
            weight_id,
            None,
            INITIALIZED_TRAINABLE,
        ))
    }

    fn take_rms_norm<C>(
        mut vb: VarBuilder<'_, R>,
        expected_len: usize,
        eps: f32,
        dtype: DType,
        client: &C,
    ) -> Result<RmsNorm<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let param_name = Self::param_name(&vb, "weight");
        let weight = take_or_init_tensor(
            &mut vb,
            "weight",
            &[expected_len],
            dtype,
            Init::Ones,
            client,
        )?;
        let weight_id = weight.id();
        Self::ensure_shape(&param_name, &weight, &[expected_len])?;
        Ok(RmsNorm::with_id(
            weight,
            weight_id,
            eps,
            INITIALIZED_TRAINABLE,
        ))
    }

    fn ensure_shape(name: &str, weight: &Tensor<R>, expected_shape: &[usize]) -> Result<()> {
        if weight.shape() != expected_shape {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "shape mismatch for '{name}': expected {:?}, got {:?}",
                    expected_shape,
                    weight.shape()
                ),
            });
        }
        Ok(())
    }

    fn param_name(vb: &VarBuilder<'_, R>, leaf: &str) -> String {
        if vb.prefix().is_empty() {
            leaf.to_string()
        } else {
            format!("{}.{}", vb.prefix(), leaf)
        }
    }

    fn push_named_param<'a>(
        params: &mut Vec<(
            String,
            boostr::autograd::TensorId,
            &'a boostr::autograd::Var<R>,
        )>,
        name: String,
        var: &'a boostr::autograd::Var<R>,
    ) {
        params.push((name, var.id(), var));
    }

    /// Name one projection's trainable parameters:
    /// `<prefix>.weight`, then `<prefix>.lora_a` / `<prefix>.lora_b` when it
    /// carries an adapter. `push_named_param` drops whichever of these is
    /// frozen, so a LoRA layer contributes only its factors.
    fn push_named_projection<'a>(
        params: &mut Vec<(
            String,
            boostr::autograd::TensorId,
            &'a boostr::autograd::Var<R>,
        )>,
        prefix: &str,
        proj: &'a MaybeLora<R>,
    ) {
        Self::push_named_param(params, format!("{prefix}.weight"), proj.weight());
        if let Some((lora_a, lora_b)) = proj.adapters() {
            Self::push_named_param(params, format!("{prefix}.lora_a"), lora_a);
            Self::push_named_param(params, format!("{prefix}.lora_b"), lora_b);
        }
    }

    fn extend_named_from_module<'a>(
        params: &mut Vec<(
            String,
            boostr::autograd::TensorId,
            &'a boostr::autograd::Var<R>,
        )>,
        prefix: &str,
        named: Vec<(String, &'a boostr::autograd::Var<R>)>,
    ) {
        for (name, var) in named {
            let checkpoint_name = match name.as_str() {
                "a_log" => "A_log",
                "d_param" => "D",
                _ => name.as_str(),
            };
            Self::push_named_param(params, format!("{prefix}.{checkpoint_name}"), var);
        }
    }

    fn extend_block_named_trainable<'a>(
        params: &mut Vec<(
            String,
            boostr::autograd::TensorId,
            &'a boostr::autograd::Var<R>,
        )>,
        layer_idx: usize,
        layer: &'a Block<R>,
    ) {
        let prefix = format!("layers.{layer_idx}");
        match &layer.block_type {
            BlockType::Mamba { block, norm } => {
                Self::push_named_param(params, format!("{prefix}.norm.weight"), norm.weight());
                Self::extend_named_from_module(
                    params,
                    &format!("{prefix}.mamba"),
                    <boostr::model::Mamba1<R> as boostr::nn::Module<R>>::named_parameters(block),
                );
            }
            BlockType::Mamba2 { block, norm } => {
                Self::push_named_param(params, format!("{prefix}.norm.weight"), norm.weight());
                Self::extend_named_from_module(
                    params,
                    &format!("{prefix}.mamba2"),
                    <boostr::model::Mamba2<R> as boostr::nn::Module<R>>::named_parameters(block),
                );
            }
            BlockType::Mamba3 { block, norm } => {
                Self::push_named_param(params, format!("{prefix}.norm.weight"), norm.weight());
                Self::push_named_param(params, format!("{prefix}.norm.bias"), norm.bias());
                Self::extend_named_from_module(
                    params,
                    &format!("{prefix}.mamba3"),
                    <boostr::model::Mamba3<R> as boostr::nn::Module<R>>::named_parameters(block),
                );
            }
            BlockType::Attention {
                attn,
                ffn,
                input_layernorm,
                post_attention_layernorm,
            } => {
                Self::push_named_param(
                    params,
                    format!("{prefix}.input_layernorm.weight"),
                    input_layernorm.weight(),
                );
                let attn_named = attn.mla.named_parameters();
                for (boostr_name, oxidizr_name) in [
                    ("q_down.weight", "q_a_proj.weight"),
                    ("q_up.weight", "q_b_proj.weight"),
                    ("q_norm.weight", "q_a_layernorm.weight"),
                    ("kv_compress.weight", "kv_a_proj_with_mqa.weight"),
                    ("kv_norm.weight", "kv_a_layernorm.weight"),
                    ("kv_decompress.weight", "kv_b_proj.weight"),
                    ("o_proj.weight", "o_proj.weight"),
                ] {
                    if let Some(var) = Self::optional_named_var(&attn_named, boostr_name) {
                        Self::push_named_param(
                            params,
                            format!("{prefix}.self_attn.{oxidizr_name}"),
                            var,
                        );
                    }
                }
                Self::push_named_param(
                    params,
                    format!("{prefix}.post_attention_layernorm.weight"),
                    post_attention_layernorm.weight(),
                );
                Self::extend_ffn_named_trainable(params, &prefix, ffn);
            }
            BlockType::LegacyAttention {
                attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            } => {
                Self::push_named_param(
                    params,
                    format!("{prefix}.input_layernorm.weight"),
                    input_layernorm.weight(),
                );
                for (name, proj) in [
                    ("q_proj", &attn.q_proj),
                    ("k_proj", &attn.k_proj),
                    ("v_proj", &attn.v_proj),
                    ("o_proj", &attn.o_proj),
                ] {
                    Self::push_named_projection(
                        params,
                        &format!("{prefix}.self_attn.{name}"),
                        proj,
                    );
                }
                Self::push_named_param(
                    params,
                    format!("{prefix}.post_attention_layernorm.weight"),
                    post_attention_layernorm.weight(),
                );
                for (name, proj) in [
                    ("gate_proj", &mlp.gate_proj),
                    ("up_proj", &mlp.up_proj),
                    ("down_proj", &mlp.down_proj),
                ] {
                    Self::push_named_projection(params, &format!("{prefix}.mlp.{name}"), proj);
                }
            }
        }
    }

    fn extend_ffn_named_trainable<'a>(
        params: &mut Vec<(
            String,
            boostr::autograd::TensorId,
            &'a boostr::autograd::Var<R>,
        )>,
        layer_prefix: &str,
        ffn: &'a FfnLayer<R>,
    ) {
        match ffn {
            FfnLayer::Mlp(mlp) => {
                for (name, proj) in [
                    ("gate_proj", &mlp.gate_proj),
                    ("up_proj", &mlp.up_proj),
                    ("down_proj", &mlp.down_proj),
                ] {
                    Self::push_named_projection(
                        params,
                        &format!("{layer_prefix}.mlp.{name}"),
                        proj,
                    );
                }
            }
            FfnLayer::Moe(moe) => {
                Self::push_named_param(
                    params,
                    format!("{layer_prefix}.moe.router.gate.weight"),
                    moe.moe.router().gate().weight(),
                );
                for (expert_idx, expert) in moe.moe.experts().iter().enumerate() {
                    Self::push_named_param(
                        params,
                        format!("{layer_prefix}.moe.experts.{expert_idx}.gate_proj.weight"),
                        expert.gate_proj().weight(),
                    );
                    Self::push_named_param(
                        params,
                        format!("{layer_prefix}.moe.experts.{expert_idx}.up_proj.weight"),
                        expert.up_proj().weight(),
                    );
                    Self::push_named_param(
                        params,
                        format!("{layer_prefix}.moe.experts.{expert_idx}.down_proj.weight"),
                        expert.down_proj().weight(),
                    );
                }
                if let Some(shared_expert) = moe.moe.shared_expert() {
                    Self::push_named_param(
                        params,
                        format!("{layer_prefix}.moe.shared_expert.gate_proj.weight"),
                        shared_expert.gate_proj().weight(),
                    );
                    Self::push_named_param(
                        params,
                        format!("{layer_prefix}.moe.shared_expert.up_proj.weight"),
                        shared_expert.up_proj().weight(),
                    );
                    Self::push_named_param(
                        params,
                        format!("{layer_prefix}.moe.shared_expert.down_proj.weight"),
                        shared_expert.down_proj().weight(),
                    );
                }
            }
        }
    }

    /// Check if tensor contains NaN/Inf values (debug helper)
    /// NOTE: Disabled for performance - enable when debugging NaN issues
    #[allow(unused_variables)]
    fn check_nan(_tensor: &Tensor<R>, _name: &str) -> Result<()> {
        // Debug code disabled for performance - uncomment to enable NaN checking:
        // let vals = tensor.contiguous()?.to_vec::<f32>();
        // let nan_count = vals.iter().filter(|x| x.is_nan()).count();
        // let inf_count = vals.iter().filter(|x| x.is_infinite()).count();
        // let neg_inf_count = vals.iter().filter(|x| **x == f32::NEG_INFINITY).count();
        // let pos_inf_count = vals.iter().filter(|x| **x == f32::INFINITY).count();
        // // Also check for very large values (potential memory corruption)
        // const LARGE_THRESHOLD: f32 = 1e10;
        // let large_count = vals.iter().filter(|x| x.is_finite() && x.abs() > LARGE_THRESHOLD).count();
        // if nan_count > 0 || inf_count > 0 || large_count > 0 {
        //     eprintln!("[NaN DEBUG] {} has {} NaN, {} Inf ({} +Inf, {} -Inf), {} LARGE (>1e10) out of {} values",
        //         name, nan_count, inf_count, pos_inf_count, neg_inf_count, large_count, vals.len());
        //     // Also print some stats
        //     let max_val = vals.iter().filter(|x| x.is_finite()).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        //     let min_val = vals.iter().filter(|x| x.is_finite()).fold(f32::INFINITY, |a, &b| a.min(b));
        //     eprintln!("[NaN DEBUG] {} finite range: [{:.4}, {:.4}]", name, min_val, max_val);
        // }
        Ok(())
    }
}

/// Write every parameter the model needs to reconstruct itself.
///
/// `params` is the optimizer's map, so it holds only trainable tensors. A
/// frozen parameter (a LoRA-adapted base weight) is never updated, so the
/// model's own `Var` still carries its current value and is the source here.
/// Saving only `params` would emit an adapter-only checkpoint that no load can
/// repair.
pub fn save_checkpoint_from_params<R>(
    path: &str,
    model: &LitGPT<R>,
    params: &HashMap<TensorId, Tensor<R>>,
) -> Result<()>
where
    R: boostr::Runtime<DType = DType>,
    R::Client: boostr::RuntimeClient<R> + boostr::TensorOps<R> + boostr::TypeConversionOps<R>,
{
    let cpu_device = boostr::CpuDevice::new();
    let parameters = model.named_parameters();
    let mut tensors = HashMap::with_capacity(parameters.len());

    for (name, id, var) in &parameters {
        let tensor = match params.get(id) {
            Some(tensor) => tensor,
            None if !var.requires_grad() => var.tensor(),
            None => {
                return Err(boostr::error::Error::ModelError {
                    reason: format!(
                        "trainable parameter '{name}' ({id}) is absent from the optimizer \
                         parameter map while saving checkpoint '{path}'. The optimizer \
                         lost track of it; rebuild the model and optimizer from the \
                         same parameter set before saving."
                    ),
                });
            }
        };
        let cpu_tensor = stage_tensor_for_safetensors::<R>(tensor, &cpu_device)?;
        tensors.insert(name.clone(), cpu_tensor);
    }

    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "oxidizr-litgpt".to_string());
    metadata.insert("parameter_count".to_string(), tensors.len().to_string());

    boostr::format::safetensors::save_safetensors(path, &tensors, Some(&metadata))
}

pub fn load_checkpoint_into_model<R>(model: &mut LitGPT<R>, path: &str) -> Result<()>
where
    R: boostr::Runtime<DType = DType>,
    R::Client: boostr::RuntimeClient<R>
        + boostr::RandomOps<R>
        + boostr::ScalarOps<R>
        + boostr::BinaryOps<R>
        + boostr::CompareOps<R>
        + boostr::TensorOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::ReduceOps<R>
        + boostr::ShapeOps<R>,
{
    let cfg = model.cfg.clone();
    let device = model.embed_tokens.weight().tensor().device().clone();
    let client = R::default_client(&device);

    let mut safetensors = boostr::format::SafeTensors::open(path)?;
    let mut loaded = safetensors.load_all::<R>(&device)?;

    // Grow embed_tokens/lm_head row counts before they enter the VarMap: once
    // wrapped in a trainable `Var` by `LitGPT::new`, the pretrained and grown
    // rows must already be one contiguous tensor, since
    // `VarBuilder::take_or_init_tensor` hard-errors on a shape mismatch.
    for name in ["embed_tokens.weight", "lm_head.weight"] {
        if let Some(weight) = loaded.get(name) {
            let grown = crate::vocab::fit_vocab_rows(
                &client,
                weight,
                cfg.vocab_size,
                cfg.grow_vocab,
                name,
            )?;
            loaded.insert(name.to_string(), grown);
        }
    }

    // Every expected name, not just the trainable ones: `LitGPT::new` below
    // routes through `VarBuilder::take_or_init_tensor`, which silently
    // random-initializes any tensor the varmap lacks. Without this check an
    // incomplete checkpoint loads "successfully" and produces garbage.
    let mut missing = model
        .named_parameters()
        .into_iter()
        .map(|(name, _, _)| name)
        .filter(|name| !loaded.contains_key(name))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        missing.sort();
        return Err(boostr::error::Error::ModelError {
            reason: format!(
                "checkpoint '{path}' is missing {} tensor(s) the model requires: {}. \
                 Loading it would random-initialize them. Re-save the checkpoint from a \
                 run whose config matches this one, or point --resume at a complete \
                 checkpoint.",
                missing.len(),
                missing.join(", ")
            ),
        });
    }

    let target_dtype = cfg.dtype.to_dtype();
    let mut varmap = boostr::VarMap::<R>::new();
    for (name, tensor) in loaded {
        let tensor = if tensor.dtype() == target_dtype {
            tensor
        } else {
            client.cast(&tensor, target_dtype)?
        };
        varmap.insert(name, tensor);
    }

    let mut vb = VarBuilder::new(&mut varmap, &device);
    *model = LitGPT::new(&cfg, &mut vb, &client)?;
    Ok(())
}

fn stage_tensor_for_safetensors<R>(
    tensor: &Tensor<R>,
    cpu_device: &boostr::CpuDevice,
) -> Result<Tensor<boostr::CpuRuntime>>
where
    R: boostr::Runtime<DType = DType>,
    R::Client: boostr::RuntimeClient<R> + boostr::TensorOps<R> + boostr::TypeConversionOps<R>,
{
    let client = R::default_client(tensor.device());
    let shape = tensor.shape().to_vec();
    let contiguous = tensor.contiguous()?;
    let f32_tensor = if contiguous.dtype() == DType::F32 {
        contiguous
    } else {
        client.cast(&contiguous, DType::F32)?
    };
    let f32_tensor = f32_tensor.contiguous()?;
    let values = f32_tensor.to_vec::<f32>();

    Tensor::<boostr::CpuRuntime>::try_from_slice(&values, &shape, cpu_device)
        .map_err(boostr::error::Error::Numr)
}

#[cfg(test)]
mod tie_word_embeddings_tests {
    use super::*;
    use boostr::{CpuDevice, CpuRuntime};

    fn tiny_cfg(tie_word_embeddings: bool) -> OxidizrModelConfig {
        OxidizrModelConfig {
            name: "tie-test".to_string(),
            hidden_size: 8,
            num_layers: 0,
            num_heads: 2,
            kv_heads: None,
            kv_latent_dim: None,
            q_latent_dim: None,
            d_rope: None,
            num_experts: None,
            experts_per_tok: None,
            shared_expert_enabled: None,
            intermediate_size: Some(16),
            mamba_layers: None,
            mamba2_num_heads: None,
            mamba2_head_dim: None,
            mamba2_state_size: None,
            mamba2_chunk_size: None,
            mamba2_n_groups: None,
            mamba2_conv_kernel: None,
            mamba2_expand: None,
            mamba3_enabled: None,
            mamba3_complex_rope: None,
            mamba3_mimo_rank: None,
            mamba3_use_conv: None,
            vocab_size: 12,
            grow_vocab: false,
            tie_word_embeddings,
            tokenizer_vocab: "custom".to_string(),
            max_seq_len: 8,
            rope_theta: 10000.0,
            rope_scaling: None,
            sliding_window: None,
            speech_vocab: None,
            lora: None,
            dtype: crate::config::DTypePrecision::F32,
        }
    }

    /// Build a `LitGPT` via `with_ids` with zero transformer layers, so the
    /// tying invariant can be checked without exercising attention/RoPE/MLP
    /// construction — those are unrelated to this mechanism.
    fn tiny_model(tie_word_embeddings: bool) -> LitGPT<CpuRuntime> {
        let cfg = tiny_cfg(tie_word_embeddings);
        let device = CpuDevice::new();
        let embed =
            Tensor::<CpuRuntime>::zeros(&[cfg.vocab_size, cfg.hidden_size], DType::F32, &device);
        let norm = Tensor::<CpuRuntime>::zeros(&[cfg.hidden_size], DType::F32, &device);
        // A distinct tensor/id from embed_tokens, standing in for whatever a
        // checkpoint load would produce for an UNTIED model. When tied,
        // `with_ids` must discard this and re-alias from embed_tokens instead.
        let lm_head =
            Tensor::<CpuRuntime>::zeros(&[cfg.vocab_size, cfg.hidden_size], DType::F32, &device);

        LitGPT::with_ids(
            &cfg,
            (embed, TensorId::new()),
            Vec::new(),
            (norm, TensorId::new()),
            (lm_head, TensorId::new()),
            FrozenParams::new(),
        )
        .unwrap()
    }

    #[test]
    fn tied_lm_head_shares_embed_tokens_tensor_id() {
        let model = tiny_model(true);
        assert_eq!(
            model.embed_tokens.weight().id(),
            model.lm_head.weight().id(),
            "tied lm_head must alias embed_tokens's TensorId, not mint its own"
        );
    }

    #[test]
    fn untied_lm_head_has_a_distinct_tensor_id() {
        let model = tiny_model(false);
        assert_ne!(
            model.embed_tokens.weight().id(),
            model.lm_head.weight().id(),
            "untied lm_head keeps its own independent TensorId"
        );
    }

    #[test]
    fn tied_named_trainable_parameters_omits_lm_head_weight() {
        let model = tiny_model(true);
        let names: Vec<String> = model
            .named_trainable_parameters()
            .into_iter()
            .map(|(name, _, _)| name)
            .collect();
        assert!(names.contains(&"embed_tokens.weight".to_string()));
        assert!(!names.contains(&"lm_head.weight".to_string()));
    }

    #[test]
    fn untied_named_trainable_parameters_includes_lm_head_weight() {
        let model = tiny_model(false);
        let names: Vec<String> = model
            .named_trainable_parameters()
            .into_iter()
            .map(|(name, _, _)| name)
            .collect();
        assert!(names.contains(&"lm_head.weight".to_string()));
    }

    #[test]
    fn tied_trainable_parameters_has_one_entry_for_the_shared_weight() {
        let model = tiny_model(true);
        let map: HashMap<boostr::autograd::TensorId, &boostr::autograd::Var<CpuRuntime>> =
            model.trainable_parameters().into_iter().collect();
        // embed_tokens.weight (shared with lm_head) + norm.weight: two
        // distinct TensorIds, not three — the optimizer's TensorId-keyed
        // HashMap must see the tied parameter exactly once.
        assert_eq!(map.len(), 2);
        assert!(map.contains_key(&model.embed_tokens.weight().id()));
    }
}

/// `sliding_window`/`rope_scaling` on `AttentionConfig` must reach the
/// dense-attention block's own `var_*`-based construction path (`src/model.rs`),
/// not just boostr's `Llama` builder that oxidizr does not use.
#[cfg(test)]
mod attention_extras_reach_dense_attention_tests {
    use super::*;
    use boostr::model::config::{AttentionConfig, RopeScalingConfig};
    use boostr::{CpuClient, CpuDevice, CpuRuntime};

    fn attn_cfg(
        sliding_window: Option<usize>,
        rope_scaling: Option<RopeScalingConfig>,
    ) -> AttentionConfig {
        AttentionConfig {
            num_heads: 2,
            num_kv_heads: None,
            head_dim: None,
            rope_theta: 10000.0,
            rope_scaling,
            kv_latent_dim: None,
            q_latent_dim: None,
            d_rope: None,
            sliding_window,
            use_alibi: false,
        }
    }

    fn model_cfg(attention: AttentionConfig) -> ModelConfig {
        ModelConfig {
            model_type: "llama".to_string(),
            vocab_size: 16,
            hidden_size: 8,
            num_layers: 1,
            max_seq_len: 8,
            intermediate_size: None,
            rms_norm_eps: default_rms_norm_eps(),
            attention: Some(attention),
            ssm: None,
            moe: None,
            hybrid_layers: None,
            tie_word_embeddings: false,
            grow_vocab: false,
            vision: None,
            audio: None,
        }
    }

    fn build_attention(attention: AttentionConfig) -> CausalSelfAttention<CpuRuntime> {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let cfg = model_cfg(attention);
        let mut varmap = boostr::VarMap::<CpuRuntime>::new();
        let vb = VarBuilder::new(&mut varmap, &device);
        CausalSelfAttention::<CpuRuntime>::new(&cfg, vb, &client, DType::F32, None).unwrap()
    }

    #[test]
    fn sliding_window_set_reaches_the_attention_block() {
        let attn = build_attention(attn_cfg(Some(4096), None));
        assert_eq!(attn.sliding_window, 4096);
    }

    #[test]
    fn sliding_window_unset_disables_the_window() {
        let attn = build_attention(attn_cfg(None, None));
        assert_eq!(
            attn.sliding_window, 0,
            "unset sliding_window must map to boostr's 0 = disabled sentinel"
        );
    }

    #[test]
    fn rope_scaling_changes_the_rotary_cache_versus_unscaled() {
        let device = CpuDevice::new();
        let head_dim = 4; // hidden_size(8) / num_heads(2), matching build_attention's cfg
        let plain =
            RotaryEmbedding::<CpuRuntime>::new(head_dim, 8, 10000.0, None, &device).unwrap();
        let scaling = RopeScalingConfig {
            scaling_type: "linear".to_string(),
            factor: 4.0,
            original_max_position_embeddings: None,
            low_freq_factor: None,
            high_freq_factor: None,
            attention_factor: None,
            beta_fast: None,
            beta_slow: None,
        };
        let scaled =
            RotaryEmbedding::<CpuRuntime>::new(head_dim, 8, 10000.0, Some(&scaling), &device)
                .unwrap();

        let plain_cos: Vec<f32> = plain.cos.tensor().to_vec();
        let scaled_cos: Vec<f32> = scaled.cos.tensor().to_vec();
        assert_ne!(
            plain_cos, scaled_cos,
            "rope_scaling must change the precomputed cos cache reaching \
             oxidizr's own attention block, not just boostr's unused Llama builder"
        );
    }
}

/// Freezing a parameter must survive `rebuild_with_parameters` — the call the
/// training loop makes on every step. Before `FrozenParams` was threaded
/// through the reconstruction path, every `*_with_ids` site passed a literal
/// `true`, so a frozen weight came back trainable on step 1.
#[cfg(test)]
mod frozen_parameter_tests {
    use super::*;
    use boostr::{CpuClient, CpuDevice, CpuRuntime};

    /// The legacy GQA + MLP block: one layer, no MLA, no MoE, so the model
    /// exercises a real `Block::with_ids` reconstruction without needing an
    /// SSM or expert config.
    const FROZEN_PARAM: &str = "layers.0.self_attn.q_proj.weight";

    fn cfg() -> OxidizrModelConfig {
        OxidizrModelConfig {
            name: "freeze-test".to_string(),
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            kv_heads: None,
            kv_latent_dim: None,
            q_latent_dim: None,
            d_rope: None,
            num_experts: None,
            experts_per_tok: None,
            shared_expert_enabled: None,
            intermediate_size: Some(16),
            mamba_layers: None,
            mamba2_num_heads: None,
            mamba2_head_dim: None,
            mamba2_state_size: None,
            mamba2_chunk_size: None,
            mamba2_n_groups: None,
            mamba2_conv_kernel: None,
            mamba2_expand: None,
            mamba3_enabled: None,
            mamba3_complex_rope: None,
            mamba3_mimo_rank: None,
            mamba3_use_conv: None,
            vocab_size: 12,
            grow_vocab: false,
            tie_word_embeddings: false,
            tokenizer_vocab: "custom".to_string(),
            max_seq_len: 8,
            rope_theta: 10000.0,
            rope_scaling: None,
            sliding_window: None,
            speech_vocab: None,
            lora: None,
            dtype: crate::config::DTypePrecision::F32,
        }
    }

    fn build_model() -> LitGPT<CpuRuntime> {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut varmap = boostr::VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut varmap, &device);
        LitGPT::<CpuRuntime>::new(&cfg(), &mut vb, &client).expect("model builds")
    }

    /// The `q_proj` weight `Var` itself, reached through the private block
    /// fields so `requires_grad` can be read directly — `trainable_parameters`
    /// would hide a frozen parameter and prove nothing.
    fn q_proj_var(model: &LitGPT<CpuRuntime>) -> &boostr::autograd::Var<CpuRuntime> {
        match &model.layers[0].block_type {
            BlockType::LegacyAttention { attn, .. } => attn.q_proj.weight(),
            _ => panic!("freeze-test config must produce a LegacyAttention block"),
        }
    }

    fn named_id(model: &LitGPT<CpuRuntime>, name: &str) -> boostr::autograd::TensorId {
        model
            .named_trainable_parameters()
            .into_iter()
            .find(|(param_name, _, _)| param_name == name)
            .map(|(_, id, _)| id)
            .unwrap_or_else(|| panic!("model has no trainable parameter named '{name}'"))
    }

    /// The map the trainer hands to `rebuild_with_parameters`: every
    /// optimizer-tracked (i.e. trainable) parameter keyed by its `TensorId`.
    fn optimizer_params(
        model: &LitGPT<CpuRuntime>,
    ) -> HashMap<boostr::autograd::TensorId, Tensor<CpuRuntime>> {
        model
            .trainable_parameters()
            .into_iter()
            .map(|(id, var)| (id, var.tensor().clone()))
            .collect()
    }

    fn values(tensor: &Tensor<CpuRuntime>) -> Vec<f32> {
        tensor.contiguous().expect("contiguous").to_vec::<f32>()
    }

    fn trainable_ids(
        model: &LitGPT<CpuRuntime>,
    ) -> std::collections::HashSet<boostr::autograd::TensorId> {
        model
            .trainable_parameters()
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    fn model_with_q_proj_frozen() -> LitGPT<CpuRuntime> {
        let model = build_model();
        let frozen = FrozenParams::from_ids([named_id(&model, FROZEN_PARAM)]);
        model
            .with_frozen_parameters(frozen)
            .expect("freezing q_proj rebuilds the model")
    }

    #[test]
    fn freezing_clears_requires_grad_on_the_parameter() {
        let model = model_with_q_proj_frozen();
        assert!(
            !q_proj_var(&model).requires_grad(),
            "a frozen parameter must be reconstructed with requires_grad = false"
        );
    }

    #[test]
    fn frozen_parameter_stays_frozen_across_rebuild_with_parameters() {
        let model = model_with_q_proj_frozen();
        let frozen_id = q_proj_var(&model).id();
        let params = optimizer_params(&model);
        assert!(
            !params.contains_key(&frozen_id),
            "a frozen parameter must not reach the optimizer's parameter map"
        );

        // The exact call the training loop makes on every step.
        let rebuilt = model
            .rebuild_with_parameters(&params)
            .expect("rebuild succeeds");

        assert!(
            !q_proj_var(&rebuilt).requires_grad(),
            "'{FROZEN_PARAM}' came back trainable after rebuild_with_parameters"
        );
        assert!(
            !trainable_ids(&rebuilt).contains(&frozen_id),
            "'{FROZEN_PARAM}' must stay out of trainable_parameters() after a rebuild"
        );
        assert_eq!(
            q_proj_var(&rebuilt).id(),
            frozen_id,
            "freezing must not change the parameter's TensorId"
        );
        assert_eq!(
            rebuilt.frozen_parameters(),
            model.frozen_parameters(),
            "the frozen set must be carried forward by the rebuild"
        );
    }

    #[test]
    fn frozen_parameter_value_survives_the_rebuild_unchanged() {
        let model = model_with_q_proj_frozen();
        let before = values(q_proj_var(&model).tensor());

        let rebuilt = model
            .rebuild_with_parameters(&optimizer_params(&model))
            .expect("rebuild succeeds");
        let after = values(q_proj_var(&rebuilt).tensor());

        assert_eq!(
            before, after,
            "a frozen parameter's tensor must come through the \
             tensor_with_stable_id fallback unchanged, not from the optimizer map"
        );
        assert!(
            before.iter().any(|v| *v != 0.0),
            "the test weight must be non-trivial for the comparison to mean anything"
        );
    }

    #[test]
    fn named_trainable_parameters_omits_the_frozen_parameter() {
        // named_trainable_parameters drives checkpoint save, so a frozen
        // parameter must not appear there either.
        let model = model_with_q_proj_frozen();
        let names: Vec<String> = model
            .named_trainable_parameters()
            .into_iter()
            .map(|(name, _, _)| name)
            .collect();
        assert!(
            !names.contains(&FROZEN_PARAM.to_string()),
            "'{FROZEN_PARAM}' must be absent from named_trainable_parameters()"
        );
        assert!(
            names.contains(&"layers.0.self_attn.k_proj.weight".to_string()),
            "unfrozen siblings must still be listed"
        );
    }

    #[test]
    fn nothing_frozen_leaves_every_parameter_trainable_across_a_rebuild() {
        let model = build_model();
        assert!(
            model.frozen_parameters().is_empty(),
            "LitGPT::new must freeze nothing"
        );

        let before_ids = trainable_ids(&model);
        let before_names: Vec<String> = model
            .named_trainable_parameters()
            .into_iter()
            .map(|(name, _, _)| name)
            .collect();

        let rebuilt = model
            .rebuild_with_parameters(&optimizer_params(&model))
            .expect("rebuild succeeds");

        let after_names: Vec<String> = rebuilt
            .named_trainable_parameters()
            .into_iter()
            .map(|(name, _, _)| name)
            .collect();

        assert_eq!(before_ids, trainable_ids(&rebuilt));
        assert_eq!(before_names, after_names);
        assert!(
            q_proj_var(&rebuilt).requires_grad(),
            "with nothing frozen every parameter stays trainable"
        );
    }

    #[test]
    fn freezing_one_parameter_leaves_the_others_trainable() {
        let plain = build_model();
        let plain_count = plain.trainable_parameters().len();

        let frozen_model = model_with_q_proj_frozen();
        assert_eq!(
            frozen_model.trainable_parameters().len(),
            plain_count - 1,
            "freezing one parameter must remove exactly one entry"
        );
    }

    #[test]
    fn partially_freezing_a_shared_flag_group_is_rejected() {
        // boostr's LayerNorm/Linear/Conv1d and the Mamba raw parameters share
        // one trainability flag, so a half-frozen group has no valid answer.
        let frozen = FrozenParams::from_ids([TensorId::new()]);
        let a = TensorId::new();
        let err = frozen
            .group_trainable("norm (LayerNorm)", &[("norm.weight", a), ("norm.bias", a)])
            .err();
        assert!(
            err.is_none(),
            "an entirely unfrozen group must not be an error"
        );

        let mut partial = FrozenParams::new();
        let weight_id = TensorId::new();
        let bias_id = TensorId::new();
        partial.freeze(weight_id);
        let message = partial
            .group_trainable(
                "norm (LayerNorm)",
                &[("norm.weight", weight_id), ("norm.bias", bias_id)],
            )
            .expect_err("a half-frozen group must be rejected")
            .to_string();
        assert!(message.contains("norm.weight"), "{message}");
        assert!(message.contains("norm.bias"), "{message}");
    }
}

/// LoRA must add adapters and freeze the projections it adapts, and must keep
/// doing both across `rebuild_with_parameters` — the call the training loop
/// makes on every step. A mistake here silently converts LoRA back into full
/// fine-tuning, or severs the adapter from the gradient graph.
#[cfg(test)]
mod lora_tests {
    use super::*;
    use boostr::{CpuClient, CpuDevice, CpuRuntime};

    const TARGETS: [&str; 4] = ["q_proj", "k_proj", "v_proj", "o_proj"];
    const MLP_TARGETS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

    fn cfg(lora: Option<LoraConfig>) -> OxidizrModelConfig {
        OxidizrModelConfig {
            name: "lora-test".to_string(),
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            kv_heads: None,
            kv_latent_dim: None,
            q_latent_dim: None,
            d_rope: None,
            num_experts: None,
            experts_per_tok: None,
            shared_expert_enabled: None,
            intermediate_size: Some(16),
            mamba_layers: None,
            mamba2_num_heads: None,
            mamba2_head_dim: None,
            mamba2_state_size: None,
            mamba2_chunk_size: None,
            mamba2_n_groups: None,
            mamba2_conv_kernel: None,
            mamba2_expand: None,
            mamba3_enabled: None,
            mamba3_complex_rope: None,
            mamba3_mimo_rank: None,
            mamba3_use_conv: None,
            vocab_size: 12,
            grow_vocab: false,
            tie_word_embeddings: false,
            tokenizer_vocab: "custom".to_string(),
            max_seq_len: 8,
            rope_theta: 10000.0,
            rope_scaling: None,
            sliding_window: None,
            speech_vocab: None,
            lora,
            dtype: crate::config::DTypePrecision::F32,
        }
    }

    fn lora_cfg() -> LoraConfig {
        LoraConfig {
            rank: 2,
            alpha: 4.0,
            target_modules: TARGETS.iter().map(|name| name.to_string()).collect(),
        }
    }

    fn build(lora: Option<LoraConfig>) -> LitGPT<CpuRuntime> {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut varmap = boostr::VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut varmap, &device);
        LitGPT::<CpuRuntime>::new(&cfg(lora), &mut vb, &client).expect("model builds")
    }

    fn names(model: &LitGPT<CpuRuntime>) -> Vec<String> {
        model
            .named_trainable_parameters()
            .into_iter()
            .map(|(name, _, _)| name)
            .collect()
    }

    fn trainable_ids(
        model: &LitGPT<CpuRuntime>,
    ) -> std::collections::HashSet<boostr::autograd::TensorId> {
        model
            .trainable_parameters()
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    fn optimizer_params(
        model: &LitGPT<CpuRuntime>,
    ) -> HashMap<boostr::autograd::TensorId, Tensor<CpuRuntime>> {
        model
            .trainable_parameters()
            .into_iter()
            .map(|(id, var)| (id, var.tensor().clone()))
            .collect()
    }

    fn projections(model: &LitGPT<CpuRuntime>) -> Vec<&MaybeLora<CpuRuntime>> {
        match &model.layers[0].block_type {
            BlockType::LegacyAttention { attn, .. } => {
                vec![&attn.q_proj, &attn.k_proj, &attn.v_proj, &attn.o_proj]
            }
            _ => panic!("lora-test config must produce a LegacyAttention block"),
        }
    }

    /// The MLP's three projections, in `MLP_TARGETS` order.
    fn mlp_projections(model: &LitGPT<CpuRuntime>) -> Vec<&MaybeLora<CpuRuntime>> {
        match &model.layers[0].block_type {
            BlockType::LegacyAttention { mlp, .. } => mlp.projections().to_vec(),
            _ => panic!("lora-test config must produce a LegacyAttention block"),
        }
    }

    fn lora_cfg_for(targets: &[&str]) -> LoraConfig {
        LoraConfig {
            rank: 2,
            alpha: 4.0,
            target_modules: targets.iter().map(|name| name.to_string()).collect(),
        }
    }

    /// Every adapted projection, paired with the dotted prefix its parameters
    /// are named under.
    fn adapted_with_prefixes(model: &LitGPT<CpuRuntime>) -> Vec<(String, &MaybeLora<CpuRuntime>)> {
        let mut pairs = Vec::new();
        for (name, proj) in TARGETS.iter().zip(projections(model)) {
            pairs.push((format!("layers.0.self_attn.{name}"), proj));
        }
        for (name, proj) in MLP_TARGETS.iter().zip(mlp_projections(model)) {
            pairs.push((format!("layers.0.mlp.{name}"), proj));
        }
        pairs
    }

    #[test]
    fn adapters_are_trainable_and_the_wrapped_bases_are_not() {
        let model = build(Some(lora_cfg()));
        let listed = names(&model);
        for target in TARGETS {
            assert!(
                listed.contains(&format!("layers.0.self_attn.{target}.lora_a")),
                "{target}.lora_a must be trainable: {listed:?}"
            );
            assert!(
                listed.contains(&format!("layers.0.self_attn.{target}.lora_b")),
                "{target}.lora_b must be trainable: {listed:?}"
            );
            assert!(
                !listed.contains(&format!("layers.0.self_attn.{target}.weight")),
                "{target}.weight is the wrapped base — LoRA must freeze it, not train it"
            );
        }
        // The un-adapted siblings are untouched: LoRA freezes what it wraps,
        // nothing more.
        assert!(listed.contains(&"layers.0.mlp.gate_proj.weight".to_string()));
        assert!(listed.contains(&"embed_tokens.weight".to_string()));
    }

    #[test]
    fn base_weights_carry_requires_grad_false() {
        let model = build(Some(lora_cfg()));
        for proj in projections(&model) {
            assert!(
                proj.adapters().is_some(),
                "every targeted projection must be adapted"
            );
            assert!(
                !proj.weight().requires_grad(),
                "a wrapped base must be reconstructed with requires_grad = false"
            );
        }
    }

    #[test]
    fn trainable_set_and_ids_are_identical_across_a_rebuild() {
        // This is exactly what `assert_stable_parameter_ids` enforces at
        // runtime: a rebuilt model whose trainable-id set differs from the
        // optimizer's cached set is a hard error every step.
        let model = build(Some(lora_cfg()));
        let before_ids = trainable_ids(&model);
        let before_names = names(&model);

        let rebuilt = model
            .rebuild_with_parameters(&optimizer_params(&model))
            .expect("rebuild succeeds");

        assert_eq!(before_ids, trainable_ids(&rebuilt));
        assert_eq!(before_names, names(&rebuilt));
    }

    #[test]
    fn adapters_still_require_grad_and_bases_stay_frozen_after_a_rebuild() {
        let model = build(Some(lora_cfg()));
        let rebuilt = model
            .rebuild_with_parameters(&optimizer_params(&model))
            .expect("rebuild succeeds");

        for (before, after) in projections(&model).into_iter().zip(projections(&rebuilt)) {
            let (a_before, b_before) = before.adapters().expect("adapted before");
            let (a_after, b_after) = after.adapters().expect("adapted after");
            assert_eq!(a_before.id(), a_after.id(), "lora_a must keep its TensorId");
            assert_eq!(b_before.id(), b_after.id(), "lora_b must keep its TensorId");
            assert!(a_after.requires_grad(), "lora_a must stay trainable");
            assert!(b_after.requires_grad(), "lora_b must stay trainable");
            assert!(
                !after.weight().requires_grad(),
                "the wrapped base must stay frozen"
            );
        }
    }

    #[test]
    fn rebuild_preserves_rank_and_scaling() {
        let model = build(Some(lora_cfg()));
        let rebuilt = model
            .rebuild_with_parameters(&optimizer_params(&model))
            .expect("rebuild succeeds");

        for proj in projections(&rebuilt) {
            match proj {
                MaybeLora::Lora(lora) => {
                    assert_eq!(lora.rank(), 2);
                    // alpha / rank = 4.0 / 2
                    assert!((lora.scaling() - 2.0).abs() < 1e-6, "{}", lora.scaling());
                }
                MaybeLora::Plain(_) => panic!("a targeted projection came back un-adapted"),
            }
        }
    }

    /// The anti-severed-graph test, at MODEL level.
    ///
    /// boostr's own `LoraLinear` test covers the layer in isolation; this one
    /// covers the whole forward pass, where the adapter sits behind reshapes,
    /// RoPE, and flash attention. If any of that drops the adapter's grad_fn,
    /// `lora_b` receives no gradient and every adapter silently never trains —
    /// with no error, no NaN, and a loss that still falls because the rest of
    /// the network learns.
    #[test]
    fn a_gradient_reaches_lora_b_through_the_model() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut varmap = boostr::VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut varmap, &device);
        let model = LitGPT::<CpuRuntime>::new(&cfg(Some(lora_cfg())), &mut vb, &client)
            .expect("model builds");

        // i64, matching what `data.rs` uploads (it casts the u32 stream first).
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4], &[1, 4], &device);
        let output = model.forward(&client, &input_ids).expect("forward");
        let loss =
            boostr::autograd::var_sum(&output.logits, &[0, 1, 2], false, &client).expect("reduce");
        let grads = boostr::autograd::backward(&loss, &client).expect("backward");

        for (proj, name) in projections(&model).into_iter().zip(TARGETS) {
            let (lora_a, lora_b) = proj.adapters().expect("adapted");
            // lora_b is zero-initialised, so d(loss)/d(lora_a) is zero at step
            // 0 by construction; lora_b is the factor that must receive signal
            // immediately.
            let grad = grads
                .get(lora_b.id())
                .unwrap_or_else(|| panic!("{name}.lora_b received no gradient"));
            let values: Vec<f32> = grad.contiguous().expect("contiguous").to_vec();
            let magnitude: f32 = values.iter().map(|v| v.abs()).sum();
            assert!(
                magnitude > 1e-8,
                "{name}.lora_b gradient is all zeros ({magnitude}) — the LoRA graph is severed"
            );
            assert!(
                grads.get(lora_a.id()).is_some(),
                "{name}.lora_a must be reachable from the loss"
            );
        }
    }

    #[test]
    fn lora_disabled_leaves_the_parameter_set_unchanged() {
        let model = build(None);
        assert!(
            model.frozen_parameters().is_empty(),
            "without a lora block nothing may be frozen"
        );
        for proj in projections(&model) {
            assert!(proj.adapters().is_none(), "no adapter without a lora block");
            assert!(proj.weight().requires_grad());
        }
        let listed = names(&model);
        for target in TARGETS {
            assert!(listed.contains(&format!("layers.0.self_attn.{target}.weight")));
        }
        assert!(
            !listed.iter().any(|name| name.contains("lora_")),
            "a non-LoRA model must expose no lora parameters: {listed:?}"
        );
    }

    #[test]
    fn only_the_targeted_projections_are_adapted() {
        let model = build(Some(LoraConfig {
            rank: 2,
            alpha: 4.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        }));
        let listed = names(&model);
        for target in ["q_proj", "v_proj"] {
            assert!(listed.contains(&format!("layers.0.self_attn.{target}.lora_a")));
            assert!(!listed.contains(&format!("layers.0.self_attn.{target}.weight")));
        }
        for untargeted in ["k_proj", "o_proj"] {
            assert!(!listed.contains(&format!("layers.0.self_attn.{untargeted}.lora_a")));
            assert!(
                listed.contains(&format!("layers.0.self_attn.{untargeted}.weight")),
                "an untargeted projection must stay a plain trainable Linear"
            );
        }
    }

    #[test]
    fn an_unsupported_target_module_is_rejected_by_name() {
        let message = cfg(Some(LoraConfig {
            rank: 2,
            alpha: 4.0,
            // MLA's `kv_b_proj` is real but unwrapped: naming it must fail
            // rather than yield a model with one adapter fewer than asked.
            target_modules: vec!["q_proj".to_string(), "kv_b_proj".to_string()],
        }))
        .to_boostr()
        .expect_err("an unsupported target must be rejected")
        .to_string();
        assert!(
            message.contains("kv_b_proj"),
            "the error must name the offending target: {message}"
        );
        for supported in TARGETS {
            assert!(
                message.contains(supported),
                "the error must list what IS supported: {message}"
            );
        }
    }

    #[test]
    fn rank_zero_is_rejected() {
        let message = cfg(Some(LoraConfig {
            rank: 0,
            alpha: 4.0,
            target_modules: vec!["q_proj".to_string()],
        }))
        .to_boostr()
        .expect_err("rank 0 must be rejected")
        .to_string();
        assert!(message.contains("lora.rank"), "{message}");
    }

    #[test]
    fn empty_target_modules_is_rejected() {
        let message = cfg(Some(LoraConfig {
            rank: 2,
            alpha: 4.0,
            target_modules: Vec::new(),
        }))
        .to_boostr()
        .expect_err("an empty target list must be rejected")
        .to_string();
        assert!(message.contains("lora.target_modules"), "{message}");
    }

    #[test]
    fn absent_lora_block_leaves_the_config_hash_unchanged() {
        // `compute_model_config_hash` hashes serialized JSON. A plain
        // `Option` would serialize as `null` and change the hash of every
        // pre-existing config, failing resume validation on weights that are
        // in fact compatible.
        let json = serde_json::to_string(&cfg(None)).expect("serializes");
        // Match the KEY, not the bare substring: the fixture's `name` is
        // "lora-test", which contains "lora" and would pass/fail spuriously.
        assert!(
            !json.contains("\"lora\":"),
            "an absent lora block must not reach the hashed JSON: {json}"
        );
        assert_ne!(
            crate::config::compute_model_config_hash(&cfg(None)),
            crate::config::compute_model_config_hash(&cfg(Some(lora_cfg()))),
            "a PRESENT lora block must change the hash — it changes which \
             parameters exist, so the two checkpoints are incompatible"
        );
    }

    // --- MLP projections ---

    #[test]
    fn mlp_adapters_are_trainable_and_the_wrapped_bases_are_not() {
        let model = build(Some(lora_cfg_for(&MLP_TARGETS)));
        let listed = names(&model);
        for target in MLP_TARGETS {
            assert!(
                listed.contains(&format!("layers.0.mlp.{target}.lora_a")),
                "mlp.{target}.lora_a must be trainable: {listed:?}"
            );
            assert!(
                listed.contains(&format!("layers.0.mlp.{target}.lora_b")),
                "mlp.{target}.lora_b must be trainable: {listed:?}"
            );
            assert!(
                !listed.contains(&format!("layers.0.mlp.{target}.weight")),
                "mlp.{target}.weight is the wrapped base — LoRA must freeze it, not train it"
            );
        }
        // The attention block is untargeted here and must stay untouched.
        for untargeted in TARGETS {
            assert!(
                listed.contains(&format!("layers.0.self_attn.{untargeted}.weight")),
                "an untargeted projection must stay a plain trainable Linear: {listed:?}"
            );
        }
    }

    #[test]
    fn mlp_base_weights_carry_requires_grad_false() {
        let model = build(Some(lora_cfg_for(&MLP_TARGETS)));
        for proj in mlp_projections(&model) {
            assert!(
                proj.adapters().is_some(),
                "every targeted MLP projection must be adapted"
            );
            assert!(
                !proj.weight().requires_grad(),
                "a wrapped MLP base must be reconstructed with requires_grad = false"
            );
        }
    }

    #[test]
    fn mlp_trainable_set_and_ids_are_identical_across_a_rebuild() {
        let model = build(Some(lora_cfg_for(&MLP_TARGETS)));
        let before_ids = trainable_ids(&model);
        let before_names = names(&model);

        let rebuilt = model
            .rebuild_with_parameters(&optimizer_params(&model))
            .expect("rebuild succeeds");

        assert_eq!(before_ids, trainable_ids(&rebuilt));
        assert_eq!(before_names, names(&rebuilt));

        for (before, after) in mlp_projections(&model)
            .into_iter()
            .zip(mlp_projections(&rebuilt))
        {
            let (a_before, b_before) = before.adapters().expect("adapted before");
            let (a_after, b_after) = after.adapters().expect("adapted after");
            assert_eq!(a_before.id(), a_after.id(), "lora_a must keep its TensorId");
            assert_eq!(b_before.id(), b_after.id(), "lora_b must keep its TensorId");
            assert!(a_after.requires_grad(), "lora_a must stay trainable");
            assert!(b_after.requires_grad(), "lora_b must stay trainable");
            assert!(
                !after.weight().requires_grad(),
                "the wrapped MLP base must stay frozen"
            );
        }
    }

    #[test]
    fn attention_and_mlp_can_be_targeted_together() {
        let model = build(Some(lora_cfg_for(&["q_proj", "down_proj"])));
        let listed = names(&model);
        for adapted in ["layers.0.self_attn.q_proj", "layers.0.mlp.down_proj"] {
            assert!(
                listed.contains(&format!("{adapted}.lora_a")),
                "{adapted} must be adapted: {listed:?}"
            );
            assert!(
                !listed.contains(&format!("{adapted}.weight")),
                "{adapted}.weight is a wrapped base and must be frozen"
            );
        }
        for untouched in [
            "layers.0.self_attn.k_proj",
            "layers.0.mlp.gate_proj",
            "layers.0.mlp.up_proj",
        ] {
            assert!(
                !listed.contains(&format!("{untouched}.lora_a")),
                "{untouched} was not targeted and must carry no adapter"
            );
            assert!(
                listed.contains(&format!("{untouched}.weight")),
                "{untouched}.weight must stay a plain trainable Linear"
            );
        }
    }

    /// The anti-severed-graph test with every supported target adapted at
    /// once: an adapter the SwiGLU fusion or the residual path drops receives
    /// no gradient and silently never trains.
    #[test]
    fn a_gradient_reaches_every_adapted_projections_lora_b() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut varmap = boostr::VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut varmap, &device);
        let targets: Vec<&str> = TARGETS.iter().chain(MLP_TARGETS.iter()).copied().collect();
        let model = LitGPT::<CpuRuntime>::new(&cfg(Some(lora_cfg_for(&targets))), &mut vb, &client)
            .expect("model builds");

        // i64, matching what `data.rs` uploads (it casts the u32 stream first).
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4], &[1, 4], &device);
        let output = model.forward(&client, &input_ids).expect("forward");
        let loss =
            boostr::autograd::var_sum(&output.logits, &[0, 1, 2], false, &client).expect("reduce");
        let grads = boostr::autograd::backward(&loss, &client).expect("backward");

        for (name, proj) in adapted_with_prefixes(&model) {
            let (lora_a, lora_b) = proj
                .adapters()
                .unwrap_or_else(|| panic!("{name} must be adapted"));
            // lora_b is zero-initialised, so d(loss)/d(lora_a) is zero at step
            // 0 by construction; lora_b is the factor that must receive signal
            // immediately.
            let grad = grads
                .get(lora_b.id())
                .unwrap_or_else(|| panic!("{name}.lora_b received no gradient"));
            let values: Vec<f32> = grad.contiguous().expect("contiguous").to_vec();
            let magnitude: f32 = values.iter().map(|v| v.abs()).sum();
            assert!(
                magnitude > 1e-8,
                "{name}.lora_b gradient is all zeros ({magnitude}) — the LoRA graph is severed"
            );
            assert!(
                grads.get(lora_a.id()).is_some(),
                "{name}.lora_a must be reachable from the loss"
            );
        }
    }

    #[test]
    fn an_mlp_target_is_rejected_for_a_moe_model() {
        let mut moe_cfg = cfg(Some(lora_cfg_for(&["q_proj", "gate_proj"])));
        moe_cfg.num_experts = Some(2);
        moe_cfg.experts_per_tok = Some(1);
        let message = moe_cfg
            .to_boostr()
            .expect_err("an MLP target must be rejected while num_experts is set")
            .to_string();
        assert!(
            message.contains("gate_proj"),
            "the error must name the offending target: {message}"
        );
        assert!(
            message.contains("num_experts"),
            "the error must name what to change: {message}"
        );
    }
}

#[cfg(test)]
mod checkpoint_completeness_tests {
    use super::*;
    use boostr::{CpuClient, CpuDevice, CpuRuntime};
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A LoRA-adapted base weight: frozen, so absent from the optimizer's
    /// parameter map, and therefore the tensor a trainable-only save drops.
    const FROZEN_BASE: &str = "layers.0.self_attn.q_proj.weight";

    fn cfg(lora: Option<crate::config::LoraConfig>) -> OxidizrModelConfig {
        OxidizrModelConfig {
            name: "checkpoint-test".to_string(),
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            kv_heads: None,
            kv_latent_dim: None,
            q_latent_dim: None,
            d_rope: None,
            num_experts: None,
            experts_per_tok: None,
            shared_expert_enabled: None,
            intermediate_size: Some(16),
            mamba_layers: None,
            mamba2_num_heads: None,
            mamba2_head_dim: None,
            mamba2_state_size: None,
            mamba2_chunk_size: None,
            mamba2_n_groups: None,
            mamba2_conv_kernel: None,
            mamba2_expand: None,
            mamba3_enabled: None,
            mamba3_complex_rope: None,
            mamba3_mimo_rank: None,
            mamba3_use_conv: None,
            vocab_size: 12,
            grow_vocab: false,
            tie_word_embeddings: false,
            tokenizer_vocab: "custom".to_string(),
            max_seq_len: 8,
            rope_theta: 10000.0,
            rope_scaling: None,
            sliding_window: None,
            speech_vocab: None,
            lora,
            dtype: crate::config::DTypePrecision::F32,
        }
    }

    fn lora_cfg() -> crate::config::LoraConfig {
        crate::config::LoraConfig {
            rank: 2,
            alpha: 4.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        }
    }

    fn build(cfg: &OxidizrModelConfig) -> LitGPT<CpuRuntime> {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut varmap = boostr::VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut varmap, &device);
        LitGPT::<CpuRuntime>::new(cfg, &mut vb, &client).expect("model builds")
    }

    /// The map the trainer owns: optimizer-tracked (trainable) tensors only.
    fn optimizer_params(
        model: &LitGPT<CpuRuntime>,
    ) -> HashMap<boostr::autograd::TensorId, Tensor<CpuRuntime>> {
        model
            .trainable_parameters()
            .into_iter()
            .map(|(id, var)| (id, var.tensor().clone()))
            .collect()
    }

    fn values(tensor: &Tensor<CpuRuntime>) -> Vec<f32> {
        tensor.contiguous().expect("contiguous").to_vec::<f32>()
    }

    fn all_values(model: &LitGPT<CpuRuntime>) -> BTreeMap<String, Vec<f32>> {
        model
            .named_parameters()
            .into_iter()
            .map(|(name, _, var)| (name, values(var.tensor())))
            .collect()
    }

    fn temp_path(tag: &str) -> String {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir()
            .join(format!(
                "oxidizr-checkpoint-{tag}-{}-{unique}.safetensors",
                std::process::id()
            ))
            .to_string_lossy()
            .into_owned()
    }

    /// Rewrite `path` without `drop_name`, emulating a truncated or
    /// hand-edited checkpoint.
    fn rewrite_without(path: &str, drop_name: &str) {
        let device = CpuDevice::new();
        let mut safetensors = boostr::format::SafeTensors::open(path).expect("open checkpoint");
        let mut tensors = safetensors
            .load_all::<CpuRuntime>(&device)
            .expect("load checkpoint");
        assert!(
            tensors.remove(drop_name).is_some(),
            "'{drop_name}' must be present before it can be removed"
        );
        boostr::format::safetensors::save_safetensors(path, &tensors, None).expect("rewrite");
    }

    /// Write only the trainable parameters — exactly what the checkpoint
    /// writer produced before frozen tensors were included.
    fn save_trainable_only(path: &str, model: &LitGPT<CpuRuntime>) {
        let cpu_device = CpuDevice::new();
        let tensors: HashMap<String, Tensor<CpuRuntime>> = model
            .named_trainable_parameters()
            .into_iter()
            .map(|(name, _, var)| {
                (
                    name,
                    stage_tensor_for_safetensors::<CpuRuntime>(var.tensor(), &cpu_device)
                        .expect("stage tensor"),
                )
            })
            .collect();
        boostr::format::safetensors::save_safetensors(path, &tensors, None).expect("save");
    }

    fn checkpoint_names(path: &str) -> BTreeMap<String, Vec<f32>> {
        let device = CpuDevice::new();
        let mut safetensors = boostr::format::SafeTensors::open(path).expect("open checkpoint");
        safetensors
            .load_all::<CpuRuntime>(&device)
            .expect("load checkpoint")
            .into_iter()
            .map(|(name, tensor)| (name, values(&tensor)))
            .collect()
    }

    fn round_trip(cfg: &OxidizrModelConfig, tag: &str) -> (LitGPT<CpuRuntime>, LitGPT<CpuRuntime>) {
        let saved = build(cfg);
        let path = temp_path(tag);
        save_checkpoint_from_params(&path, &saved, &optimizer_params(&saved)).expect("save");

        let mut restored = build(cfg);
        load_checkpoint_into_model(&mut restored, &path).expect("load");
        let _ = std::fs::remove_file(&path);
        (saved, restored)
    }

    #[test]
    fn lora_checkpoint_round_trip_restores_frozen_bases() {
        let cfg = cfg(Some(lora_cfg()));
        let (saved, restored) = round_trip(&cfg, "lora");

        let saved_values = all_values(&saved);
        assert!(
            saved_values.contains_key(FROZEN_BASE),
            "the frozen base weight must be part of the saved parameter set"
        );
        assert_eq!(
            saved_values,
            all_values(&restored),
            "every parameter must survive the round trip, frozen bases included"
        );
    }

    #[test]
    fn mlp_lora_checkpoint_round_trip_restores_every_parameter() {
        const MLP_BASE: &str = "layers.0.mlp.gate_proj.weight";
        let cfg = cfg(Some(crate::config::LoraConfig {
            rank: 2,
            alpha: 4.0,
            target_modules: vec![
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
        }));
        let (saved, restored) = round_trip(&cfg, "mlp-lora");

        let saved_values = all_values(&saved);
        assert!(
            saved_values.contains_key(MLP_BASE),
            "the frozen MLP base weight must be part of the saved parameter set"
        );
        assert!(
            saved_values.contains_key("layers.0.mlp.gate_proj.lora_b"),
            "the MLP adapter must be part of the saved parameter set"
        );
        assert_eq!(
            saved_values,
            all_values(&restored),
            "every parameter must survive the round trip, frozen MLP bases included"
        );
    }

    #[test]
    fn round_trip_without_lora_is_unchanged() {
        let cfg = cfg(None);
        let (saved, restored) = round_trip(&cfg, "plain");
        assert_eq!(all_values(&saved), all_values(&restored));
    }

    #[test]
    fn frozen_parameters_stay_frozen_after_load() {
        let cfg = cfg(Some(lora_cfg()));
        let (saved, restored) = round_trip(&cfg, "frozen");

        let frozen_names = |model: &LitGPT<CpuRuntime>| {
            model
                .named_parameters()
                .into_iter()
                .filter(|(_, _, var)| !var.requires_grad())
                .map(|(name, _, _)| name)
                .collect::<Vec<_>>()
        };
        let saved_frozen = frozen_names(&saved);
        assert!(
            saved_frozen.contains(&FROZEN_BASE.to_string()),
            "LoRA must freeze '{FROZEN_BASE}'; got {saved_frozen:?}"
        );
        assert_eq!(
            saved_frozen,
            frozen_names(&restored),
            "the frozen set must be reconstructed by the load, not lost"
        );
        assert!(!restored.frozen_parameters().is_empty());
    }

    #[test]
    fn incomplete_checkpoint_is_rejected() {
        let cfg = cfg(None);
        let saved = build(&cfg);
        let path = temp_path("incomplete");
        save_checkpoint_from_params(&path, &saved, &optimizer_params(&saved)).expect("save");

        const DROPPED: &str = "layers.0.mlp.up_proj.weight";
        rewrite_without(&path, DROPPED);

        let mut restored = build(&cfg);
        let result = load_checkpoint_into_model(&mut restored, &path);
        let _ = std::fs::remove_file(&path);

        let error = match result {
            Ok(()) => panic!("an incomplete checkpoint must not load silently"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains(DROPPED),
            "the error must name the missing tensor; got: {error}"
        );
        assert!(
            error.contains(&path),
            "the error must name the checkpoint file; got: {error}"
        );
    }

    #[test]
    fn lora_checkpoint_missing_its_frozen_base_is_rejected() {
        let cfg = cfg(Some(lora_cfg()));
        let saved = build(&cfg);
        let path = temp_path("adapter-only");
        // The pre-fix writer's output: adapters only, every frozen base absent.
        save_trainable_only(&path, &saved);

        let mut restored = build(&cfg);
        let result = load_checkpoint_into_model(&mut restored, &path);
        let _ = std::fs::remove_file(&path);

        let error = match result {
            Ok(()) => panic!("an adapter-only checkpoint must not load silently"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains(FROZEN_BASE),
            "the error must name the missing base weight; got: {error}"
        );
    }

    #[test]
    fn full_fine_tune_checkpoint_still_loads() {
        let cfg = cfg(None);
        let saved = build(&cfg);
        let path = temp_path("legacy");
        // Nothing is frozen without LoRA, so this is byte-for-byte the
        // parameter set every existing checkpoint carries.
        save_trainable_only(&path, &saved);

        let expected = all_values(&saved);
        assert_eq!(
            checkpoint_names(&path).keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>(),
            "a full fine-tune checkpoint must carry exactly the model's parameter set"
        );

        let mut restored = build(&cfg);
        load_checkpoint_into_model(&mut restored, &path).expect("legacy checkpoint must load");
        let _ = std::fs::remove_file(&path);
        assert_eq!(expected, all_values(&restored));
    }
}
