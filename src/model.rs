use crate::prelude::{DType, Embedding, Init, Linear, Result, RmsNorm, Tensor, VarBuilder};

use crate::config::ModelConfig;
use boostr::tensor::TensorId;
// `cast` is a TypeConversionOps method; the trait must be in scope to call it.
use boostr::TypeConversionOps;
use std::collections::HashMap;

/// A weight tensor paired with the stable parameter id it was registered
/// under. Used throughout the `*_weights_with_ids`/`with_ids` rebuild path
/// so a checkpoint's tensors can be re-associated with boostr's `TensorId`
/// keys without re-running initialization.
type WeightWithId<R> = (Tensor<R>, boostr::autograd::TensorId);

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
    fn new(dim: usize, max_seq_len: usize, theta: f32, device: &R::Device) -> Result<Self> {
        let rope = boostr::nn::RoPE::<R>::precompute_freqs(max_seq_len, dim, theta, None, device);
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
    ) -> Result<Self> {
        let mla_config = Self::mla_config(cfg)?;
        let qk_dim = mla_config.qk_head_dim();

        let weights = boostr::nn::MlaWeights {
            q_down: Some(Self::linear_with_ids(
                "q_a_proj.weight",
                q_a_proj.0,
                q_a_proj.1,
                &[mla_config.q_lora_rank, mla_config.hidden_size],
            )?),
            q_up: Self::linear_with_ids(
                "q_b_proj.weight",
                q_b_proj.0,
                q_b_proj.1,
                &[mla_config.num_heads * qk_dim, mla_config.q_lora_rank],
            )?,
            q_norm: Some(Self::rms_norm_with_id(
                "q_a_layernorm.weight",
                q_a_layernorm.0,
                q_a_layernorm.1,
                mla_config.q_lora_rank,
                mla_config.norm_eps,
            )?),
            kv_compress: Self::linear_with_ids(
                "kv_a_proj_with_mqa.weight",
                kv_a_proj_with_mqa.0,
                kv_a_proj_with_mqa.1,
                &[
                    mla_config.kv_lora_rank + mla_config.rope_head_dim,
                    mla_config.hidden_size,
                ],
            )?,
            kv_norm: Some(Self::rms_norm_with_id(
                "kv_a_layernorm.weight",
                kv_a_layernorm.0,
                kv_a_layernorm.1,
                mla_config.kv_lora_rank,
                mla_config.norm_eps,
            )?),
            kv_decompress: Self::linear_with_ids(
                "kv_b_proj.weight",
                kv_b_proj.0,
                kv_b_proj.1,
                &[
                    mla_config.num_heads * (mla_config.head_dim + mla_config.head_dim_v),
                    mla_config.kv_lora_rank,
                ],
            )?,
            o_proj: Self::linear_with_ids(
                "o_proj.weight",
                o_proj.0,
                o_proj.1,
                &[
                    mla_config.hidden_size,
                    mla_config.num_heads * mla_config.head_dim_v,
                ],
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
        let head_dim = Self::head_dim(cfg)?;
        let kv_latent_dim = Self::required_dim("kv_latent_dim", cfg.kv_latent_dim)?;
        let q_latent_dim = Self::required_dim("q_latent_dim", cfg.q_latent_dim)?;
        let d_rope = Self::required_dim("d_rope", cfg.d_rope)?;

        let config = boostr::nn::MlaConfig {
            hidden_size: cfg.hidden_size,
            num_heads: cfg.num_heads,
            head_dim,
            head_dim_v: head_dim,
            kv_lora_rank: kv_latent_dim,
            q_lora_rank: q_latent_dim,
            rope_head_dim: d_rope,
            max_seq_len: cfg.max_seq_len,
            rope_theta: cfg.rope_theta,
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
            boostr::nn::Linear::with_ids(weight, weight_id, None, true),
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
        Ok(boostr::nn::RmsNorm::with_id(weight, weight_id, eps, true))
    }

    fn linear_with_ids(
        name: &str,
        weight: Tensor<R>,
        weight_id: boostr::autograd::TensorId,
        expected_shape: &[usize],
    ) -> Result<boostr::nn::MaybeQuantLinear<R>> {
        Self::ensure_shape(name, &weight, expected_shape)?;
        Ok(boostr::nn::MaybeQuantLinear::Standard(
            boostr::nn::Linear::with_ids(weight, weight_id, None, true),
        ))
    }

    fn rms_norm_with_id(
        name: &str,
        weight: Tensor<R>,
        weight_id: boostr::autograd::TensorId,
        expected_len: usize,
        eps: f32,
    ) -> Result<boostr::nn::RmsNorm<R>> {
        Self::ensure_shape(name, &weight, &[expected_len])?;
        Ok(boostr::nn::RmsNorm::with_id(weight, weight_id, eps, true))
    }

    fn head_dim(cfg: &ModelConfig) -> Result<usize> {
        if cfg.num_heads == 0 {
            return Err(boostr::error::Error::ModelError {
                reason: "MLA num_heads must be non-zero".into(),
            });
        }
        if !cfg.hidden_size.is_multiple_of(cfg.num_heads) {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "MLA hidden_size ({}) must be divisible by num_heads ({})",
                    cfg.hidden_size, cfg.num_heads
                ),
            });
        }
        Ok(cfg.hidden_size / cfg.num_heads)
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

// --- Causal Self Attention with Grouped Query Attention ---

struct CausalSelfAttention<R: boostr::Runtime> {
    q_proj: boostr::nn::Linear<R>,
    k_proj: boostr::nn::Linear<R>,
    v_proj: boostr::nn::Linear<R>,
    o_proj: boostr::nn::Linear<R>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: RotaryEmbedding<R>,
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
    ) -> Result<Self>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let head_dim = Self::head_dim(cfg)?;
        let kv_heads = cfg.kv_heads.unwrap_or(cfg.num_heads);
        Self::validate_heads(cfg.num_heads, kv_heads)?;
        let kv_hidden = kv_heads * head_dim;

        let q_proj = Self::linear_no_bias(
            vb.pp("q_proj"),
            &[cfg.hidden_size, cfg.hidden_size],
            dtype,
            client,
        )?;
        let k_proj = Self::linear_no_bias(
            vb.pp("k_proj"),
            &[kv_hidden, cfg.hidden_size],
            dtype,
            client,
        )?;
        let v_proj = Self::linear_no_bias(
            vb.pp("v_proj"),
            &[kv_hidden, cfg.hidden_size],
            dtype,
            client,
        )?;
        let o_proj = Self::linear_no_bias(
            vb.pp("o_proj"),
            &[cfg.hidden_size, cfg.hidden_size],
            dtype,
            client,
        )?;
        let rope = RotaryEmbedding::new(head_dim, cfg.max_seq_len, cfg.rope_theta, vb.device())?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_heads,
            num_kv_heads: kv_heads,
            head_dim,
            rope,
        })
    }

    fn with_ids(
        cfg: &ModelConfig,
        q_proj: (Tensor<R>, boostr::autograd::TensorId),
        k_proj: (Tensor<R>, boostr::autograd::TensorId),
        v_proj: (Tensor<R>, boostr::autograd::TensorId),
        o_proj: (Tensor<R>, boostr::autograd::TensorId),
        rope_cos: (Tensor<R>, boostr::autograd::TensorId),
        rope_sin: (Tensor<R>, boostr::autograd::TensorId),
    ) -> Result<Self> {
        let head_dim = Self::head_dim(cfg)?;
        let kv_heads = cfg.kv_heads.unwrap_or(cfg.num_heads);
        Self::validate_heads(cfg.num_heads, kv_heads)?;
        let kv_hidden = kv_heads * head_dim;

        Ok(Self {
            q_proj: Self::linear_with_ids(
                "q_proj.weight",
                q_proj.0,
                q_proj.1,
                &[cfg.hidden_size, cfg.hidden_size],
            )?,
            k_proj: Self::linear_with_ids(
                "k_proj.weight",
                k_proj.0,
                k_proj.1,
                &[kv_hidden, cfg.hidden_size],
            )?,
            v_proj: Self::linear_with_ids(
                "v_proj.weight",
                v_proj.0,
                v_proj.1,
                &[kv_hidden, cfg.hidden_size],
            )?,
            o_proj: Self::linear_with_ids(
                "o_proj.weight",
                o_proj.0,
                o_proj.1,
                &[cfg.hidden_size, cfg.hidden_size],
            )?,
            num_heads: cfg.num_heads,
            num_kv_heads: kv_heads,
            head_dim,
            rope: RotaryEmbedding::with_ids(
                rope_cos.0, rope_cos.1, rope_sin.0, rope_sin.1, head_dim,
            ),
        })
    }

    fn forward<C>(
        &self,
        client: &C,
        x: &boostr::autograd::Var<R>,
    ) -> Result<boostr::autograd::Var<R>>
    where
        C: boostr::RuntimeClient<R> + boostr::RoPEOps<R> + boostr::TensorOps<R>,
        R::Client: boostr::FlashAttentionOps<R> + boostr::TensorOps<R>,
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
            0,
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
        if cfg.num_heads == 0 {
            return Err(boostr::error::Error::ModelError {
                reason: "attention num_heads must be non-zero".into(),
            });
        }
        if !cfg.hidden_size.is_multiple_of(cfg.num_heads) {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "hidden_size ({}) must be divisible by num_heads ({})",
                    cfg.hidden_size, cfg.num_heads
                ),
            });
        }
        Ok(cfg.hidden_size / cfg.num_heads)
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
        let param_name = if vb.prefix().is_empty() {
            "weight".to_string()
        } else {
            format!("{}.weight", vb.prefix())
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
        Ok(boostr::nn::Linear::with_ids(weight, weight_id, None, true))
    }

    fn linear_with_ids(
        name: &str,
        weight: Tensor<R>,
        weight_id: boostr::autograd::TensorId,
        expected_shape: &[usize],
    ) -> Result<boostr::nn::Linear<R>> {
        Self::ensure_shape(name, &weight, expected_shape)?;
        Ok(boostr::nn::Linear::with_ids(weight, weight_id, None, true))
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
    gate_proj: boostr::nn::Linear<R>,
    up_proj: boostr::nn::Linear<R>,
    down_proj: boostr::nn::Linear<R>,
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

        Ok(Self {
            gate_proj: Self::linear_no_bias(
                vb.pp("gate_proj"),
                &[intermediate_size, cfg.hidden_size],
                dtype,
                client,
            )?,
            up_proj: Self::linear_no_bias(
                vb.pp("up_proj"),
                &[intermediate_size, cfg.hidden_size],
                dtype,
                client,
            )?,
            down_proj: Self::linear_no_bias(
                vb.pp("down_proj"),
                &[cfg.hidden_size, intermediate_size],
                dtype,
                client,
            )?,
        })
    }

    fn with_ids(
        cfg: &ModelConfig,
        gate_proj: (Tensor<R>, boostr::autograd::TensorId),
        up_proj: (Tensor<R>, boostr::autograd::TensorId),
        down_proj: (Tensor<R>, boostr::autograd::TensorId),
    ) -> Result<Self> {
        let intermediate_size = cfg.intermediate_size.unwrap_or(cfg.hidden_size * 4);

        Ok(Self {
            gate_proj: Self::linear_with_ids(
                "gate_proj.weight",
                gate_proj.0,
                gate_proj.1,
                &[intermediate_size, cfg.hidden_size],
            )?,
            up_proj: Self::linear_with_ids(
                "up_proj.weight",
                up_proj.0,
                up_proj.1,
                &[intermediate_size, cfg.hidden_size],
            )?,
            down_proj: Self::linear_with_ids(
                "down_proj.weight",
                down_proj.0,
                down_proj.1,
                &[cfg.hidden_size, intermediate_size],
            )?,
        })
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
            + boostr::TypeConversionOps<R>,
        R::Client: boostr::TensorOps<R> + boostr::ScalarOps<R> + boostr::TypeConversionOps<R>,
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
        let param_name = if vb.prefix().is_empty() {
            "weight".to_string()
        } else {
            format!("{}.weight", vb.prefix())
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
        Ok(boostr::nn::Linear::with_ids(weight, weight_id, None, true))
    }

    fn linear_with_ids(
        name: &str,
        weight: Tensor<R>,
        weight_id: boostr::autograd::TensorId,
        expected_shape: &[usize],
    ) -> Result<boostr::nn::Linear<R>> {
        Self::ensure_shape(name, &weight, expected_shape)?;
        Ok(boostr::nn::Linear::with_ids(weight, weight_id, None, true))
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
    ) -> Result<Self> {
        let gate_proj = Self::linear_with_ids(
            "gate_proj.weight",
            gate_proj.0,
            gate_proj.1,
            &[intermediate, hidden],
        )?;
        let up_proj = Self::linear_with_ids(
            "up_proj.weight",
            up_proj.0,
            up_proj.1,
            &[intermediate, hidden],
        )?;
        let down_proj = Self::linear_with_ids(
            "down_proj.weight",
            down_proj.0,
            down_proj.1,
            &[hidden, intermediate],
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
        Ok(boostr::nn::Linear::with_ids(weight, weight_id, None, true))
    }

    fn linear_with_ids(
        name: &str,
        weight: Tensor<R>,
        weight_id: boostr::autograd::TensorId,
        expected_shape: &[usize],
    ) -> Result<boostr::nn::Linear<R>> {
        Self::ensure_shape(name, &weight, expected_shape)?;
        Ok(boostr::nn::Linear::with_ids(weight, weight_id, None, true))
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
    ) -> Result<Self> {
        let gate = Expert::<R>::linear_with_ids(
            "gate.weight",
            gate.0,
            gate.1,
            &[config.num_experts, hidden],
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
    ) -> Result<Self> {
        let (num_experts, experts_per_tok, intermediate) = Self::required_moe_dims(cfg)?;
        if experts.len() != num_experts {
            return Err(boostr::error::Error::ModelError {
                reason: format!(
                    "MoE expected {num_experts} experts from cfg.num_experts, got {}",
                    experts.len()
                ),
            });
        }

        let mut boostr_experts = Vec::with_capacity(num_experts);
        for (gate_proj, up_proj, down_proj) in experts {
            let expert =
                Expert::with_ids(cfg.hidden_size, intermediate, gate_proj, up_proj, down_proj)?;
            boostr_experts.push(expert.into_inner());
        }

        let shared_expert = Expert::with_ids(
            cfg.hidden_size,
            intermediate,
            shared_expert.0,
            shared_expert.1,
            shared_expert.2,
        )?
        .into_inner();
        let router = Router::with_ids(
            cfg.hidden_size,
            Self::router_config(num_experts, experts_per_tok),
            router_gate,
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
        let num_experts = Self::required_positive("num_experts", cfg.num_experts)?;
        let experts_per_tok = Self::required_positive("experts_per_tok", cfg.experts_per_tok)?;
        let intermediate = Self::required_positive("intermediate_size", cfg.intermediate_size)?;
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
    fn new<C>(cfg: &ModelConfig, vb: VarBuilder<'_, R>, client: &C, dtype: DType) -> Result<Self>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        if cfg.num_experts.is_some() {
            // MoE mode
            Ok(FfnLayer::Moe(Box::new(MoeLayer::new(
                cfg, vb, client, dtype,
            )?)))
        } else {
            // Simple MLP mode (no MoE)
            Ok(FfnLayer::Mlp(Box::new(Mlp::new(cfg, vb, client, dtype)?)))
        }
    }

    #[allow(clippy::type_complexity)]
    fn with_ids(
        cfg: &ModelConfig,
        mlp: Option<(
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
        )>,
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
    ) -> Result<Self> {
        if cfg.num_experts.is_some() {
            let (experts, shared_expert, router_gate) =
                moe.ok_or_else(|| boostr::error::Error::ModelError {
                    reason: "FfnLayer::with_ids requires MoE weights when cfg.num_experts is set"
                        .into(),
                })?;
            Ok(FfnLayer::Moe(Box::new(MoeLayer::with_ids(
                cfg,
                experts,
                shared_expert,
                router_gate,
            )?)))
        } else {
            let (gate_proj, up_proj, down_proj) =
                mlp.ok_or_else(|| boostr::error::Error::ModelError {
                    reason: "FfnLayer::with_ids requires MLP weights when cfg.num_experts is unset"
                        .into(),
                })?;
            Ok(FfnLayer::Mlp(Box::new(Mlp::with_ids(
                cfg, gate_proj, up_proj, down_proj,
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
    ) -> Result<Self>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let eps = 1e-5;

        // Determine if this should be a Mamba layer
        let is_mamba = cfg
            .mamba_layers
            .as_ref()
            .map(|layers| layers.contains(&layer_idx))
            .unwrap_or(false);

        // Determine if using Mamba3 (Trapezoidal + Complex RoPE + MIMO)
        let use_mamba3 = cfg.mamba3_enabled.unwrap_or(false);

        // Determine if using Mamba2 (State Space Duality)
        let use_mamba2 = cfg.mamba2_num_heads.is_some();

        // Determine if using MLA (Multi-Head Latent Attention) or legacy GQA
        let use_mla = cfg.kv_latent_dim.is_some();

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
                    true,
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
                    true,
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
                    true,
                )?),
                norm: Self::take_rms_norm(vb.pp("norm"), cfg.hidden_size, eps, dtype, client)?,
            }
        } else if use_mla {
            // MLA attention layer (with MLP or MoE based on config)
            let ffn_prefix = if cfg.num_experts.is_some() {
                "moe"
            } else {
                "mlp"
            };
            BlockType::Attention {
                attn: Box::new(MultiHeadLatentAttention::new(
                    cfg,
                    vb.pp("self_attn"),
                    client,
                    dtype,
                )?),
                ffn: FfnLayer::new(cfg, vb.pp(ffn_prefix), client, dtype)?,
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
                )?),
                mlp: Box::new(Mlp::new(cfg, vb.pp("mlp"), client, dtype)?),
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
            Option<(
                (Tensor<R>, boostr::autograd::TensorId),
                (Tensor<R>, boostr::autograd::TensorId),
                (Tensor<R>, boostr::autograd::TensorId),
            )>,
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
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
            (Tensor<R>, boostr::autograd::TensorId),
        )>,
        device: &R::Device,
    ) -> Result<Self> {
        let eps = 1e-5;

        // Determine if this should be a Mamba layer
        let is_mamba = cfg
            .mamba_layers
            .as_ref()
            .map(|layers| layers.contains(&layer_idx))
            .unwrap_or(false);

        // Determine if using Mamba3 (Trapezoidal + Complex RoPE + MIMO)
        let use_mamba3 = cfg.mamba3_enabled.unwrap_or(false);

        // Determine if using Mamba2 (State Space Duality)
        let use_mamba2 = cfg.mamba2_num_heads.is_some();

        // Determine if using MLA (Multi-Head Latent Attention) or legacy GQA
        let use_mla = cfg.kv_latent_dim.is_some();

        let block_type = if is_mamba && use_mamba3 && use_mamba2 {
            let (weights, norm_weight, norm_bias) =
                mamba3.ok_or_else(|| boostr::error::Error::ModelError {
                    reason:
                        "Block::with_ids requires Mamba3 weights and pre-norm for a Mamba3 layer"
                            .into(),
                })?;
            BlockType::Mamba3 {
                block: Box::new(boostr::model::Mamba3::with_ids(
                    Self::mamba3_config(cfg)?,
                    weights,
                    true,
                )),
                norm: Self::layer_norm_with_ids(
                    "norm",
                    norm_weight,
                    norm_bias,
                    cfg.hidden_size,
                    MAMBA_PRENORM_EPS,
                )?,
            }
        } else if is_mamba && use_mamba2 {
            let (weights, norm) = mamba2.ok_or_else(|| boostr::error::Error::ModelError {
                reason: "Block::with_ids requires Mamba2 weights and pre-norm for a Mamba2 layer"
                    .into(),
            })?;
            BlockType::Mamba2 {
                block: Box::new(boostr::model::Mamba2::with_ids(
                    Self::mamba2_config(cfg)?,
                    weights,
                    true,
                )),
                norm: Self::rms_norm_with_id(
                    "norm.weight",
                    norm.0,
                    norm.1,
                    cfg.hidden_size,
                    MAMBA_PRENORM_EPS,
                )?,
            }
        } else if is_mamba {
            let (weights, norm) = mamba1.ok_or_else(|| boostr::error::Error::ModelError {
                reason: "Block::with_ids requires Mamba1 weights and norm for a Mamba1 layer"
                    .into(),
            })?;
            BlockType::Mamba {
                block: Box::new(boostr::model::Mamba1::with_ids(
                    Self::mamba1_config(cfg)?,
                    weights,
                    true,
                )),
                norm: Self::rms_norm_with_id("norm.weight", norm.0, norm.1, cfg.hidden_size, eps)?,
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
                )?),
                ffn: FfnLayer::with_ids(cfg, mlp, moe)?,
                input_layernorm: Self::rms_norm_with_id(
                    "input_layernorm.weight",
                    input_layernorm.0,
                    input_layernorm.1,
                    cfg.hidden_size,
                    eps,
                )?,
                post_attention_layernorm: Self::rms_norm_with_id(
                    "post_attention_layernorm.weight",
                    post_attention_layernorm.0,
                    post_attention_layernorm.1,
                    cfg.hidden_size,
                    eps,
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
                    cfg, q_proj, k_proj, v_proj, o_proj, rope_cos, rope_sin,
                )?),
                mlp: Box::new(Mlp::with_ids(cfg, gate_proj, up_proj, down_proj)?),
                input_layernorm: Self::rms_norm_with_id(
                    "input_layernorm.weight",
                    input_layernorm.0,
                    input_layernorm.1,
                    cfg.hidden_size,
                    eps,
                )?,
                post_attention_layernorm: Self::rms_norm_with_id(
                    "post_attention_layernorm.weight",
                    post_attention_layernorm.0,
                    post_attention_layernorm.1,
                    cfg.hidden_size,
                    eps,
                )?,
            }
        };

        Ok(Self { block_type })
    }

    fn forward<C>(&self, client: &C, x: &boostr::autograd::Var<R>) -> Result<BlockOutput<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
            + boostr::CompareOps<R>
            + boostr::ConvOps<R>
            + boostr::QuantMatmulOps<R>
            + boostr::RoPEOps<R>
            + boostr::TypeConversionOps<R>,
        R::Client: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
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

    fn mamba1_config(cfg: &ModelConfig) -> Result<boostr::model::Mamba1Config> {
        Self::reject_unexpressed_mamba1_value("mamba2_num_heads", cfg.mamba2_num_heads)?;
        Self::reject_unexpressed_mamba1_value("mamba2_head_dim", cfg.mamba2_head_dim)?;
        Self::reject_unexpressed_mamba1_value("mamba2_n_groups", cfg.mamba2_n_groups)?;
        if cfg.mamba3_enabled.unwrap_or(false) {
            return Err(boostr::error::Error::ModelError {
                reason: "Mamba1Config has no field for cfg.mamba3_enabled".into(),
            });
        }
        Self::reject_unexpressed_mamba1_value("mamba3_complex_rope", cfg.mamba3_complex_rope)?;
        Self::reject_unexpressed_mamba1_value("mamba3_mimo_rank", cfg.mamba3_mimo_rank)?;
        Self::reject_unexpressed_mamba1_value("mamba3_use_conv", cfg.mamba3_use_conv)?;

        let mut config = boostr::model::Mamba1Config::new(cfg.hidden_size);
        if let Some(state_size) = cfg.mamba2_state_size {
            config.d_state = state_size;
        }
        if let Some(chunk_size) = cfg.mamba2_chunk_size {
            config.chunk_size = chunk_size;
        }
        if let Some(expand) = cfg.mamba2_expand {
            config.expand = expand;
        }
        if let Some(conv_kernel) = cfg.mamba2_conv_kernel {
            config.d_conv = conv_kernel;
        }
        config.validate()?;
        Ok(config)
    }

    fn mamba2_config(cfg: &ModelConfig) -> Result<boostr::model::Mamba2Config> {
        let config = boostr::model::Mamba2Config {
            d_model: cfg.hidden_size,
            d_state: Self::required_mamba_value("mamba2_state_size", cfg.mamba2_state_size)?,
            nheads: Self::required_mamba_value("mamba2_num_heads", cfg.mamba2_num_heads)?,
            headdim: Self::required_mamba_value("mamba2_head_dim", cfg.mamba2_head_dim)?,
            expand: Self::required_mamba_value("mamba2_expand", cfg.mamba2_expand)?,
            ngroups: Self::required_mamba_value("mamba2_n_groups", cfg.mamba2_n_groups)?,
            d_conv: Self::required_mamba_value("mamba2_conv_kernel", cfg.mamba2_conv_kernel)?,
            chunk_size: Self::required_mamba_value("mamba2_chunk_size", cfg.mamba2_chunk_size)?,
            use_dt_bias: true,
            use_d: true,
            dt_softplus: true,
        };
        config.validate()?;
        Ok(config)
    }

    fn mamba3_config(cfg: &ModelConfig) -> Result<boostr::model::Mamba3Config> {
        let config = boostr::model::Mamba3Config {
            d_model: cfg.hidden_size,
            d_state: Self::required_mamba_value("mamba2_state_size", cfg.mamba2_state_size)?,
            nheads: Self::required_mamba_value("mamba2_num_heads", cfg.mamba2_num_heads)?,
            headdim: Self::required_mamba_value("mamba2_head_dim", cfg.mamba2_head_dim)?,
            expand: Self::required_mamba_value("mamba2_expand", cfg.mamba2_expand)?,
            ngroups: Self::required_mamba_value("mamba2_n_groups", cfg.mamba2_n_groups)?,
            d_conv: cfg.mamba2_conv_kernel.unwrap_or(4),
            chunk_size: Self::required_mamba_value("mamba2_chunk_size", cfg.mamba2_chunk_size)?,
            dt_softplus: true,
            use_d: true,
            use_dt_bias: true,
            use_complex_rope: cfg.mamba3_complex_rope.unwrap_or(true),
            mimo_rank: cfg.mamba3_mimo_rank.unwrap_or(0),
            use_conv: cfg.mamba3_use_conv.unwrap_or(false),
            time_step_min: 0.001,
            time_step_max: 4.0,
        };
        config.validate()?;
        Ok(config)
    }

    fn required_mamba_value(name: &str, value: Option<usize>) -> Result<usize> {
        let value = value.ok_or_else(|| boostr::error::Error::ModelError {
            reason: format!("Mamba config requires cfg.{name}"),
        })?;
        if value == 0 {
            return Err(boostr::error::Error::ModelError {
                reason: format!("Mamba cfg.{name} must be non-zero"),
            });
        }
        Ok(value)
    }

    fn reject_unexpressed_mamba1_value<T>(name: &str, value: Option<T>) -> Result<()> {
        if value.is_some() {
            return Err(boostr::error::Error::ModelError {
                reason: format!("Mamba1Config has no field for cfg.{name}"),
            });
        }
        Ok(())
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
        Ok(RmsNorm::with_id(weight, weight_id, eps, true))
    }

    fn layer_norm_with_ids(
        name: &str,
        weight: (Tensor<R>, boostr::autograd::TensorId),
        bias: (Tensor<R>, boostr::autograd::TensorId),
        expected_len: usize,
        eps: f32,
    ) -> Result<boostr::nn::LayerNorm<R>> {
        Self::ensure_shape(&format!("{name}.weight"), &weight.0, &[expected_len])?;
        Self::ensure_shape(&format!("{name}.bias"), &bias.0, &[expected_len])?;
        Ok(boostr::nn::LayerNorm::with_ids(
            weight.0, weight.1, bias.0, bias.1, eps, true,
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
            weight, weight_id, bias, bias_id, eps, true,
        ))
    }

    fn rms_norm_with_id(
        name: &str,
        weight: Tensor<R>,
        weight_id: boostr::autograd::TensorId,
        expected_len: usize,
        eps: f32,
    ) -> Result<RmsNorm<R>> {
        Self::ensure_shape(name, &weight, &[expected_len])?;
        Ok(RmsNorm::with_id(weight, weight_id, eps, true))
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
    cfg: ModelConfig,
    embed_tokens: Embedding<R>,
    layers: Vec<Block<R>>,
    norm: RmsNorm<R>,
    lm_head: Linear<R>,
    /// Model dtype (F32/F16/BF16 as configured)
    /// RmsNorm and SSM computations use F32 internally for numerical stability
    #[allow(dead_code)]
    compute_dtype: DType,
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
    pub fn new<C>(cfg: &ModelConfig, vb: &mut VarBuilder<'_, R>, client: &C) -> Result<Self>
    where
        C: boostr::RuntimeClient<R>
            + boostr::RandomOps<R>
            + boostr::ScalarOps<R>
            + boostr::BinaryOps<R>
            + boostr::CompareOps<R>
            + boostr::TensorOps<R>,
    {
        let compute_dtype = cfg.dtype.to_dtype();
        let embed = Self::take_embedding(
            vb.pp("embed_tokens"),
            cfg.vocab_size,
            cfg.hidden_size,
            compute_dtype,
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
                cfg,
                i,
                vb.pp(&format!("layers.{}", i)),
                client,
                compute_dtype,
            )?);
        }

        let norm =
            Self::take_rms_norm(vb.pp("norm"), cfg.hidden_size, 1e-5, compute_dtype, client)?;
        let lm_head = Self::take_linear_no_bias(
            vb.pp("lm_head"),
            &[cfg.vocab_size, cfg.hidden_size],
            compute_dtype,
            client,
        )?;

        Self::with_ids(
            cfg,
            (embed.weight().tensor().clone(), embed.weight().id()),
            layers,
            (norm.weight().tensor().clone(), norm.weight().id()),
            (lm_head.weight().tensor().clone(), lm_head.weight().id()),
        )
    }

    fn with_ids(
        cfg: &ModelConfig,
        embed_tokens: (Tensor<R>, boostr::autograd::TensorId),
        layers: Vec<Block<R>>,
        norm: (Tensor<R>, boostr::autograd::TensorId),
        lm_head: (Tensor<R>, boostr::autograd::TensorId),
    ) -> Result<Self> {
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

        Ok(Self {
            cfg: cfg.clone(),
            embed_tokens: Embedding::with_id(embed_tokens.0, embed_tokens.1, true),
            layers,
            norm: RmsNorm::with_id(norm.0, norm.1, 1e-5, true),
            lm_head: Linear::with_ids(lm_head.0, lm_head.1, None, true),
            compute_dtype: cfg.dtype.to_dtype(),
        })
    }

    pub fn rebuild_with_parameters(
        &self,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
    ) -> Result<Self> {
        let device = self.embed_tokens.weight().tensor().device();
        let mut layers = Vec::with_capacity(self.layers.len());
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            layers.push(Self::rebuild_block_with_parameters(
                &self.cfg, layer_idx, layer, params, device,
            )?);
        }

        Self::with_ids(
            &self.cfg,
            Self::tensor_with_stable_id(self.embed_tokens.weight(), params)?,
            layers,
            Self::tensor_with_stable_id(self.norm.weight(), params)?,
            Self::tensor_with_stable_id(self.lm_head.weight(), params)?,
        )
    }

    pub fn forward<C>(&self, client: &C, input_ids: &Tensor<R>) -> Result<ModelOutput<R>>
    where
        C: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::ScalarOps<R>
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
        params.extend(self.lm_head.trainable_parameters());
        params
    }

    pub fn named_trainable_parameters(
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
        Self::push_named_param(
            &mut params,
            "lm_head.weight".to_string(),
            self.lm_head.weight(),
        );
        params
    }

    fn rebuild_block_with_parameters(
        cfg: &ModelConfig,
        layer_idx: usize,
        layer: &Block<R>,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
        device: &R::Device,
    ) -> Result<Block<R>> {
        match &layer.block_type {
            BlockType::Mamba { block, norm } => Block::with_ids(
                cfg,
                layer_idx,
                Some((
                    Self::mamba1_weights_with_ids(block, params)?,
                    Self::tensor_with_stable_id(norm.weight(), params)?,
                )),
                None,
                None,
                None,
                None,
                device,
            ),
            BlockType::Mamba2 { block, norm } => Block::with_ids(
                cfg,
                layer_idx,
                None,
                Some((
                    Self::mamba2_weights_with_ids(block, params)?,
                    Self::tensor_with_stable_id(norm.weight(), params)?,
                )),
                None,
                None,
                None,
                device,
            ),
            BlockType::Mamba3 { block, norm } => Block::with_ids(
                cfg,
                layer_idx,
                None,
                None,
                Some((
                    Self::mamba3_weights_with_ids(block, params)?,
                    Self::tensor_with_stable_id(norm.weight(), params)?,
                    Self::tensor_with_stable_id(norm.bias(), params)?,
                )),
                None,
                None,
                device,
            ),
            BlockType::Attention {
                attn,
                ffn,
                input_layernorm,
                post_attention_layernorm,
            } => {
                let (mlp, moe) = match ffn {
                    FfnLayer::Mlp(mlp) => (Some(Self::mlp_weights_with_ids(mlp, params)?), None),
                    FfnLayer::Moe(moe) => (None, Some(Self::moe_weights_with_ids(moe, params)?)),
                };
                let attn_named = attn.mla.named_parameters();
                Block::with_ids(
                    cfg,
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
                )
            }
            BlockType::LegacyAttention {
                attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            } => Block::with_ids(
                cfg,
                layer_idx,
                None,
                None,
                None,
                None,
                Some((
                    Self::tensor_with_stable_id(attn.q_proj.weight(), params)?,
                    Self::tensor_with_stable_id(attn.k_proj.weight(), params)?,
                    Self::tensor_with_stable_id(attn.v_proj.weight(), params)?,
                    Self::tensor_with_stable_id(attn.o_proj.weight(), params)?,
                    Self::tensor_with_stable_id(&attn.rope.cos, params)?,
                    Self::tensor_with_stable_id(&attn.rope.sin, params)?,
                    Self::tensor_with_stable_id(mlp.gate_proj.weight(), params)?,
                    Self::tensor_with_stable_id(mlp.up_proj.weight(), params)?,
                    Self::tensor_with_stable_id(mlp.down_proj.weight(), params)?,
                    Self::tensor_with_stable_id(input_layernorm.weight(), params)?,
                    Self::tensor_with_stable_id(post_attention_layernorm.weight(), params)?,
                )),
                device,
            ),
        }
    }

    fn mamba1_weights_with_ids(
        block: &boostr::model::Mamba1<R>,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
    ) -> Result<boostr::model::mamba::Mamba1WeightsWithIds<R>> {
        let named = <boostr::model::Mamba1<R> as boostr::nn::Module<R>>::named_parameters(block);
        let config = block.config();
        Ok(boostr::model::mamba::Mamba1WeightsWithIds {
            in_proj: Self::linear_from_named(&named, params, "in_proj.weight", "in_proj.bias")?,
            conv1d: Self::conv1d_from_named(
                &named,
                params,
                "conv1d.weight",
                "conv1d.bias",
                Self::conv1d_hyperparams_for(config.d_conv, config.conv_channels()),
            )?,
            x_proj: Self::linear_from_named(&named, params, "x_proj.weight", "x_proj.bias")?,
            dt_proj: Self::linear_from_named(&named, params, "dt_proj.weight", "dt_proj.bias")?,
            out_proj: Self::linear_from_named(&named, params, "out_proj.weight", "out_proj.bias")?,
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
    ) -> Result<boostr::model::mamba::Mamba2WeightsWithIds<R>> {
        let named = <boostr::model::Mamba2<R> as boostr::nn::Module<R>>::named_parameters(block);
        let config = block.config();
        let norm =
            if let Some(norm) = Self::optional_pair_from_named(&named, params, "norm.weight")? {
                Some(RmsNorm::with_id(norm.0, norm.1, 1e-5, true))
            } else {
                None
            };

        Ok(boostr::model::mamba::Mamba2WeightsWithIds {
            in_proj: Self::linear_from_named(&named, params, "in_proj.weight", "in_proj.bias")?,
            conv1d: Self::conv1d_from_named(
                &named,
                params,
                "conv1d.weight",
                "conv1d.bias",
                Self::conv1d_hyperparams_for(config.d_conv, config.conv_channels()),
            )?,
            out_proj: Self::linear_from_named(&named, params, "out_proj.weight", "out_proj.bias")?,
            a_log: Self::pair_from_named(&named, params, "a_log")?,
            dt_bias: Self::optional_pair_from_named(&named, params, "dt_bias")?,
            d_param: Self::optional_pair_from_named(&named, params, "d_param")?,
            norm,
        })
    }

    fn mamba3_weights_with_ids(
        block: &boostr::model::Mamba3<R>,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
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
                )?),
                Some(Self::linear_from_named(
                    &named,
                    params,
                    "mimo_x_down.weight",
                    "mimo_x_down.bias",
                )?),
            )
        } else {
            (None, None)
        };
        let bc_norm = Self::pair_from_named(&named, params, "bc_norm.weight")?;
        let norm = Self::pair_from_named(&named, params, "norm.weight")?;

        Ok(boostr::model::mamba::Mamba3WeightsWithIds {
            in_proj: Self::linear_from_named(&named, params, "in_proj.weight", "in_proj.bias")?,
            out_proj: Self::linear_from_named(&named, params, "out_proj.weight", "out_proj.bias")?,
            lambda_proj: Self::linear_from_named(
                &named,
                params,
                "lambda_proj.weight",
                "lambda_proj.bias",
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
            bc_norm: RmsNorm::with_id(bc_norm.0, bc_norm.1, 1e-6, true),
            norm: RmsNorm::with_id(norm.0, norm.1, 1e-6, true),
            conv1d,
            mimo_x_up,
            mimo_x_down,
        })
    }

    fn mlp_weights_with_ids(
        mlp: &Mlp<R>,
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
    ) -> Result<(WeightWithId<R>, WeightWithId<R>, WeightWithId<R>)> {
        Ok((
            Self::tensor_with_stable_id(mlp.gate_proj.weight(), params)?,
            Self::tensor_with_stable_id(mlp.up_proj.weight(), params)?,
            Self::tensor_with_stable_id(mlp.down_proj.weight(), params)?,
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
    ) -> Result<Linear<R>> {
        let weight = Self::pair_from_named(named, params, weight_name)?;
        let bias = Self::optional_pair_from_named(named, params, bias_name)?;
        Ok(Linear::with_ids(weight.0, weight.1, bias, true))
    }

    fn conv1d_from_named(
        named: &[(String, &boostr::autograd::Var<R>)],
        params: &std::collections::HashMap<boostr::autograd::TensorId, Tensor<R>>,
        weight_name: &str,
        bias_name: &str,
        conv: Conv1dHyperParams,
    ) -> Result<boostr::nn::Conv1d<R>> {
        let weight = Self::pair_from_named(named, params, weight_name)?;
        let bias = Self::optional_pair_from_named(named, params, bias_name)?;
        Ok(boostr::nn::Conv1d::with_ids(
            weight.0,
            weight.1,
            bias,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            true,
        ))
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
        let weight = take_or_init_tensor(
            &mut vb,
            "weight",
            &[vocab_size, hidden_size],
            dtype,
            Init::PyTorchEmbedding,
            client,
        )?;
        let weight_id = weight.id();
        Self::ensure_shape(&param_name, &weight, &[vocab_size, hidden_size])?;
        Ok(Embedding::with_id(weight, weight_id, true))
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
        Ok(Linear::with_ids(weight, weight_id, None, true))
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
        Ok(RmsNorm::with_id(weight, weight_id, eps, true))
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
        if var.requires_grad() {
            params.push((name, var.id(), var));
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
                Self::push_named_param(
                    params,
                    format!("{prefix}.self_attn.q_proj.weight"),
                    attn.q_proj.weight(),
                );
                Self::push_named_param(
                    params,
                    format!("{prefix}.self_attn.k_proj.weight"),
                    attn.k_proj.weight(),
                );
                Self::push_named_param(
                    params,
                    format!("{prefix}.self_attn.v_proj.weight"),
                    attn.v_proj.weight(),
                );
                Self::push_named_param(
                    params,
                    format!("{prefix}.self_attn.o_proj.weight"),
                    attn.o_proj.weight(),
                );
                Self::push_named_param(
                    params,
                    format!("{prefix}.post_attention_layernorm.weight"),
                    post_attention_layernorm.weight(),
                );
                Self::push_named_param(
                    params,
                    format!("{prefix}.mlp.gate_proj.weight"),
                    mlp.gate_proj.weight(),
                );
                Self::push_named_param(
                    params,
                    format!("{prefix}.mlp.up_proj.weight"),
                    mlp.up_proj.weight(),
                );
                Self::push_named_param(
                    params,
                    format!("{prefix}.mlp.down_proj.weight"),
                    mlp.down_proj.weight(),
                );
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
                Self::push_named_param(
                    params,
                    format!("{layer_prefix}.mlp.gate_proj.weight"),
                    mlp.gate_proj.weight(),
                );
                Self::push_named_param(
                    params,
                    format!("{layer_prefix}.mlp.up_proj.weight"),
                    mlp.up_proj.weight(),
                );
                Self::push_named_param(
                    params,
                    format!("{layer_prefix}.mlp.down_proj.weight"),
                    mlp.down_proj.weight(),
                );
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

pub fn save_checkpoint_from_params<R>(
    path: &str,
    params: &HashMap<TensorId, Tensor<R>>,
    parameter_names: &[(String, TensorId)],
) -> Result<()>
where
    R: boostr::Runtime<DType = DType>,
    R::Client: boostr::RuntimeClient<R> + boostr::TensorOps<R> + boostr::TypeConversionOps<R>,
{
    let cpu_device = boostr::CpuDevice::new();
    let mut tensors = HashMap::with_capacity(parameter_names.len());

    for (name, id) in parameter_names {
        let tensor = params
            .get(id)
            .ok_or_else(|| boostr::error::Error::ModelError {
                reason: format!("missing parameter '{name}' ({id}) while saving checkpoint"),
            })?;
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
        + boostr::TypeConversionOps<R>,
{
    let cfg = model.cfg.clone();
    let device = model.embed_tokens.weight().tensor().device().clone();
    let client = R::default_client(&device);

    let mut safetensors = boostr::format::SafeTensors::open(path)?;
    let loaded = safetensors.load_all::<R>(&device)?;

    let mut missing = model
        .named_trainable_parameters()
        .into_iter()
        .map(|(name, _, _)| name)
        .filter(|name| !loaded.contains_key(name))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        missing.sort();
        return Err(boostr::error::Error::ModelError {
            reason: format!(
                "checkpoint '{path}' is missing {} trainable tensor(s): {}",
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
