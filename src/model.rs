use crate::prelude::{
    D, DType, Device, Embedding, Linear, Result, RmsNorm, Tensor, VarBuilder,
    embedding, linear_no_bias, ops, rms_norm,
};
use crate::config::ModelConfig;
use crate::mamba2::Mamba2Block;
use crate::mamba3::Mamba3Block;

// --- Rotary Positional Embeddings (RoPE) ---

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    _dim: usize,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq_len: usize, theta: f32, device: &Device) -> Result<Self> {
        let theta = theta as f64;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / dim as f64))
            .collect();

        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?
            .to_dtype(DType::F32)?; // Convert to F32 to match model dtype

        let t = Tensor::arange(0f32, max_seq_len as f32, device)?
            .reshape((max_seq_len, 1))?;

        let freqs = t.matmul(&inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;

        Ok(Self {
            sin: emb.sin()?,
            cos: emb.cos()?,
            _dim: dim,
        })
    }

    fn apply_rotary_emb(&self, q: &Tensor, k: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let (b_sz_q, n_heads_q, _seq_len_q, head_dim_q) = q.dims4()?;
        let (b_sz_k, n_heads_k, _seq_len_k, head_dim_k) = k.dims4()?;
        let q_dtype = q.dtype();  // Match model dtype
        let k_dtype = k.dtype();

        // Debug: Check input values
        LitGPT::check_nan(q, "rope_input_q")?;
        LitGPT::check_nan(k, "rope_input_k")?;

        // cos and sin are (seq_len, head_dim), need to broadcast to match Q and K separately
        let cos_q = self.cos
            .narrow(0, 0, seq_len)?
            .unsqueeze(0)?  // (1, seq_len, head_dim)
            .unsqueeze(0)?  // (1, 1, seq_len, head_dim)
            .broadcast_as((b_sz_q, n_heads_q, seq_len, head_dim_q))?
            .to_dtype(q_dtype)?;  // Match model dtype

        let sin_q = self.sin
            .narrow(0, 0, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b_sz_q, n_heads_q, seq_len, head_dim_q))?
            .to_dtype(q_dtype)?;  // Match model dtype

        let cos_k = self.cos
            .narrow(0, 0, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b_sz_k, n_heads_k, seq_len, head_dim_k))?
            .to_dtype(k_dtype)?;  // Match model dtype

        let sin_k = self.sin
            .narrow(0, 0, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b_sz_k, n_heads_k, seq_len, head_dim_k))?
            .to_dtype(k_dtype)?;  // Match model dtype

        let q_rot = Self::rotate_half(q)?;
        let k_rot = Self::rotate_half(k)?;
        LitGPT::check_nan(&q_rot, "rope_q_rot")?;
        LitGPT::check_nan(&k_rot, "rope_k_rot")?;

        let q_embed = ((q * &cos_q)? + (q_rot * &sin_q)?)?;
        let k_embed = ((k * &cos_k)? + (k_rot * &sin_k)?)?;

        Ok((q_embed, k_embed))
    }

    fn rotate_half(x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dim(D::Minus1)?;
        // Use narrow_tracked for gradient flow through the split
        let x1 = x.narrow_tracked(D::Minus1, 0, last_dim / 2)?;
        let x2 = x.narrow_tracked(D::Minus1, last_dim / 2, last_dim / 2)?;
        LitGPT::check_nan(&x1, "rotate_half_x1")?;
        LitGPT::check_nan(&x2, "rotate_half_x2")?;
        let x2_neg = x2.neg()?;
        LitGPT::check_nan(&x2_neg, "rotate_half_x2_neg")?;
        // cat() now records on tape automatically
        let result = Tensor::cat(&[&x2_neg, &x1], D::Minus1)?;
        LitGPT::check_nan(&result, "rotate_half_result")?;
        Ok(result)
    }

    /// Get cos tensor narrow'd to seq_len: [seq_len, dim]
    #[allow(dead_code)]
    fn get_cos(&self, seq_len: usize) -> Result<Tensor> {
        self.cos.narrow(0, 0, seq_len)
    }

    /// Get sin tensor narrow'd to seq_len: [seq_len, dim]
    #[allow(dead_code)]
    fn get_sin(&self, seq_len: usize) -> Result<Tensor> {
        self.sin.narrow(0, 0, seq_len)
    }
}

// --- Multi-Head Latent Attention (MLA) with Decoupled RoPE ---
// Attention mechanism that compresses KV cache significantly
// Key innovation: Compress KV into low-rank latent space, use separate RoPE heads

struct MultiHeadLatentAttention {
    // Compression projections (Down-project to latent space)
    w_dkv: Linear,  // hidden → kv_latent_dim
    w_dq: Linear,   // hidden → q_latent_dim

    // Decompression projections (Up-project from latent)
    w_uq: Linear,   // q_latent_dim → num_heads * head_dim
    w_uk: Linear,   // kv_latent_dim → num_heads * head_dim
    w_uv: Linear,   // kv_latent_dim → num_heads * head_dim

    // Decoupled RoPE "Peeking" Heads (CRITICAL for position encoding)
    w_qr: Linear,   // hidden → d_rope * num_heads
    w_kr: Linear,   // hidden → d_rope (shared across heads)

    // Output projection
    o_proj: Linear,

    // Configuration
    num_heads: usize,
    head_dim: usize,
    d_rope: usize,
    scale: f64,
    rope: RotaryEmbedding,

    // Cached causal mask (avoids recomputation every forward pass)
    causal_mask_cache: std::cell::RefCell<Option<Tensor>>,
}

impl MultiHeadLatentAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_heads;
        let d_rope = cfg.d_rope.expect("MLA requires d_rope");

        // CRITICAL: Scale must account for concatenated dimension (content + position)
        let combined_dim = head_dim + d_rope;
        let scale = 1.0 / (combined_dim as f64).sqrt();

        let kv_latent_dim = cfg.kv_latent_dim.expect("MLA requires kv_latent_dim");
        let q_latent_dim = cfg.q_latent_dim.expect("MLA requires q_latent_dim");

        let rope = RotaryEmbedding::new(
            d_rope,
            cfg.max_seq_len,
            cfg.rope_theta,
            vb.device()
        )?;

        Ok(Self {
            // Compression
            w_dkv: linear_no_bias(cfg.hidden_size, kv_latent_dim, vb.pp("w_dkv"))?,
            w_dq: linear_no_bias(cfg.hidden_size, q_latent_dim, vb.pp("w_dq"))?,

            // Decompression
            w_uq: linear_no_bias(q_latent_dim, cfg.num_heads * head_dim, vb.pp("w_uq"))?,
            w_uk: linear_no_bias(kv_latent_dim, cfg.num_heads * head_dim, vb.pp("w_uk"))?,
            w_uv: linear_no_bias(kv_latent_dim, cfg.num_heads * head_dim, vb.pp("w_uv"))?,

            // Decoupled RoPE heads
            w_qr: linear_no_bias(cfg.hidden_size, d_rope * cfg.num_heads, vb.pp("w_qr"))?,
            w_kr: linear_no_bias(cfg.hidden_size, d_rope, vb.pp("w_kr"))?,

            o_proj: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("o_proj"))?,

            num_heads: cfg.num_heads,
            head_dim,
            d_rope,
            scale,
            rope,
            causal_mask_cache: std::cell::RefCell::new(None),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;

        // 1. Compress input to latent space
        let c_kv = self.w_dkv.forward(x)?;  // [B, Seq, kv_latent_dim]
        let c_q = self.w_dq.forward(x)?;    // [B, Seq, q_latent_dim]

        // 2. Generate decoupled RoPE components (before RoPE application)
        // Use _tracked variants for gradient flow through view operations
        let q_rope_raw = self.w_qr.forward(x)?  // [B, Seq, d_rope * num_heads]
            .reshape_tracked(&[b_sz, seq_len, self.num_heads, self.d_rope])?
            .transpose_tracked(1, 2)?               // [B, num_heads, Seq, d_rope]
            .contiguous_tracked()?;

        let k_rope_raw = self.w_kr.forward(x)?;  // [B, Seq, d_rope] - keep raw for boostr

        // 4. Decompress latent to full Q, K, V
        // Use _tracked variants for gradient flow
        let q_content = self.w_uq.forward(&c_q)?
            .reshape_tracked(&[b_sz, seq_len, self.num_heads, self.head_dim])?
            .transpose_tracked(1, 2)?                // [B, num_heads, Seq, head_dim]
            .contiguous_tracked()?;

        let k_content = self.w_uk.forward(&c_kv)?
            .reshape_tracked(&[b_sz, seq_len, self.num_heads, self.head_dim])?
            .transpose_tracked(1, 2)?
            .contiguous_tracked()?;

        let v = self.w_uv.forward(&c_kv)?
            .reshape_tracked(&[b_sz, seq_len, self.num_heads, self.head_dim])?
            .transpose_tracked(1, 2)?
            .contiguous_tracked()?;

        // 3 & 5. Apply RoPE + combine content and position
        // RoPE application + concatenation
        let (q, k) = {
            // Broadcast k_rope for RoPE application
            // Use _tracked variants for gradient flow
            LitGPT::check_nan(&k_rope_raw, "mla_k_rope_raw")?;
            let k_rope = k_rope_raw
                .unsqueeze_tracked(1)?                  // [B, 1, Seq, d_rope]
                .broadcast_as_tracked((b_sz, self.num_heads, seq_len, self.d_rope))?;
            LitGPT::check_nan(&k_rope, "mla_k_rope_broadcast")?;
            LitGPT::check_nan(&q_rope_raw, "mla_q_rope_raw")?;

            // Apply RoPE to decoupled heads
            let (q_rope, k_rope) = self.rope.apply_rotary_emb(&q_rope_raw, &k_rope, seq_len)?;
            LitGPT::check_nan(&q_rope, "mla_q_rope_after_rope")?;
            LitGPT::check_nan(&k_rope, "mla_k_rope_after_rope")?;
            LitGPT::check_nan(&q_content, "mla_q_content")?;
            LitGPT::check_nan(&k_content, "mla_k_content")?;

            // Concatenate: Q = [q_content, q_rope], K = [k_content, k_rope]
            // cat() now records on tape automatically
            let q = Tensor::cat(&[&q_content, &q_rope], D::Minus1)?;  // [B, H, Seq, head_dim+d_rope]
            let k = Tensor::cat(&[&k_content, &k_rope], D::Minus1)?;

            (q, k)
        };

        // 6. Scaled Dot-Product Attention
        // Use transpose_tracked for gradient flow
        LitGPT::check_nan(&q, "mla_q")?;
        LitGPT::check_nan(&k, "mla_k")?;
        LitGPT::check_nan(&v, "mla_v")?;
        let att_scores = q.matmul(&k.transpose_tracked(2, 3)?)?;
        LitGPT::check_nan(&att_scores, "mla_att_scores_raw")?;

        let att_scores = (att_scores * self.scale)?;
        LitGPT::check_nan(&att_scores, "mla_att_scores_scaled")?;

        // Apply causal mask (cached to avoid recomputation)
        let mask = self.get_or_create_causal_mask(seq_len, att_scores.device())?
            .to_dtype(att_scores.dtype())?;  // Match model dtype
        LitGPT::check_nan(&mask, "mla_causal_mask")?;
        let att_scores = att_scores.broadcast_add(&mask)?;
        LitGPT::check_nan(&att_scores, "mla_att_scores_masked")?;

        // Softmax and apply to values
        let att_weights = ops::softmax_last_dim(&att_scores)?;
        LitGPT::check_nan(&att_weights, "mla_att_weights")?;

        let output = att_weights.matmul(&v)?;  // [B, H, Seq, head_dim]
        LitGPT::check_nan(&output, "mla_attn_v_out")?;

        // 7. Reshape and project output
        // Use _tracked variants for gradient flow
        let output = output
            .transpose_tracked(1, 2)?
            .reshape_tracked(&[b_sz, seq_len, self.num_heads * self.head_dim])?;

        let output = self.o_proj.forward(&output)?;
        LitGPT::check_nan(&output, "mla_o_proj")?;

        Ok(output)
    }

    /// Get or create cached causal mask for efficient reuse
    ///
    /// The causal mask is O(seq_len^2) memory, so we cache it to avoid
    /// recomputing on every forward pass.
    fn get_or_create_causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let mut cache = self.causal_mask_cache.borrow_mut();

        // Check if cached mask exists and has correct shape
        if let Some(ref cached_mask) = *cache {
            let cached_mask: &Tensor = cached_mask;
            if cached_mask.dims()[2] == seq_len {
                return Ok(cached_mask.clone());
            }
        }

        // Create new mask
        let mask_data: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    if j > i { f32::NEG_INFINITY } else { 0.0 }
                })
            })
            .collect();

        let mask_tensor = Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)?;
        *cache = Some(mask_tensor.clone());

        Ok(mask_tensor)
    }
}

// --- Causal Self Attention with Grouped Query Attention ---

struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    rope: RotaryEmbedding,
    // Cached causal mask (avoids recomputation)
    causal_mask_cache: std::cell::RefCell<Option<Tensor>>,
}

impl CausalSelfAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let kv_heads = cfg.kv_heads.expect("GQA requires kv_heads");
        let kv_size = kv_heads * head_dim;

        let rope = RotaryEmbedding::new(
            head_dim,
            cfg.max_seq_len,
            cfg.rope_theta,
            vb.device()
        )?;

        Ok(Self {
            q_proj: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(cfg.hidden_size, kv_size, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(cfg.hidden_size, kv_size, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("o_proj"))?,
            num_heads: cfg.num_heads,
            num_kv_heads: kv_heads,
            head_dim,
            scale,
            rope,
            causal_mask_cache: std::cell::RefCell::new(None),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;

        // Project to Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        // Q: [B, Seq, H, D] -> [B, H, Seq, D]
        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous_tracked()?;

        // K, V: [B, Seq, KV_H, D] -> [B, KV_H, Seq, D]
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous_tracked()?;

        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous_tracked()?;

        // Apply RoPE
        let (q, k) = self.rope.apply_rotary_emb(&q, &k, seq_len)?;

        // Expand K and V for Grouped Query Attention if needed
        let k = self.repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = self.repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        // Scaled Dot-Product Attention with cached causal mask
        let att_scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let att_scores = (att_scores * self.scale)?;

        // Apply causal mask for autoregressive generation (cached)
        let mask = self.get_or_create_causal_mask(seq_len, att_scores.device())?
            .to_dtype(att_scores.dtype())?;  // Match model dtype
        let att_scores = att_scores.broadcast_add(&mask)?;

        // Softmax and apply to values
        let att_weights = ops::softmax_last_dim(&att_scores)?;
        let output = att_weights.matmul(&v)?;

        // Reshape back: [B, H, Seq, D] -> [B, Seq, H, D] -> [B, Seq, Hidden]
        let output = output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        self.o_proj.forward(&output)
    }

    fn repeat_kv(&self, x: Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, n_kv_heads, seq_len, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(2)?
                .expand((b_sz, n_kv_heads, n_rep, seq_len, head_dim))?
                .reshape((b_sz, n_kv_heads * n_rep, seq_len, head_dim))?;
            Ok(x)
        }
    }

    /// Get or create cached causal mask for efficient reuse
    fn get_or_create_causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let mut cache = self.causal_mask_cache.borrow_mut();

        // Check if cached mask exists and has correct shape
        if let Some(ref cached_mask) = *cache {
            let cached_mask: &Tensor = cached_mask;
            if cached_mask.dims()[2] == seq_len {
                return Ok(cached_mask.clone());
            }
        }

        // Create new mask
        let mask_data: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    if j > i { f32::NEG_INFINITY } else { 0.0 }
                })
            })
            .collect();

        let mask_tensor = Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)?;
        *cache = Some(mask_tensor.clone());

        Ok(mask_tensor)
    }
}

// --- MLP with SwiGLU Activation ---

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size.unwrap_or(hidden * 4);

        Ok(Self {
            gate_proj: linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: silu(gate) * up
        let gate = ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let fused = (gate * up)?;
        self.down_proj.forward(&fused)
    }
}

// --- Fine-Grained Mixture of Experts (MoE) ---
// Key innovation: Many small experts + shared expert + Top-2 routing
// Prevents routing collapse and improves convergence

struct Expert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Expert {
    fn new(hidden: usize, intermediate: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU activation
        let gate = ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let fused = (gate * up)?;
        self.down_proj.forward(&fused)
    }
}

struct Router {
    gate: Linear,
}

impl Router {
    fn new(hidden: usize, num_experts: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: linear_no_bias(hidden, num_experts, vb.pp("gate"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.gate.forward(x)
    }
}

struct MoeLayer {
    experts: Vec<Expert>,
    shared_expert: Expert,  // Always active (critical for stability)
    router: Router,
    num_experts: usize,
    experts_per_tok: usize,  // Top-k routing (k=2 for POC)
}

impl MoeLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_experts = cfg.num_experts.expect("MoE requires num_experts");
        let experts_per_tok = cfg.experts_per_tok.expect("MoE requires experts_per_tok");
        let intermediate = cfg.intermediate_size.expect("MoE requires intermediate_size");

        let mut experts = Vec::new();
        for i in 0..num_experts {
            experts.push(Expert::new(
                cfg.hidden_size,
                intermediate,
                vb.pp(&format!("experts.{}", i))
            )?);
        }

        let shared_expert = Expert::new(
            cfg.hidden_size,
            intermediate,
            vb.pp("shared_expert")
        )?;

        let router = Router::new(cfg.hidden_size, num_experts, vb.pp("router"))?;

        Ok(Self {
            experts,
            shared_expert,
            router,
            num_experts,
            experts_per_tok,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b_sz, seq_len, hidden) = x.dims3()?;

        LitGPT::check_nan(x, "moe_input")?;

        // 1. Get router logits
        let router_logits = self.router.forward(x)?;  // [B, Seq, num_experts]
        LitGPT::check_nan(&router_logits, "moe_router_logits")?;

        // 2. Compute routing probabilities (for load balancing loss)
        let router_probs = ops::softmax_last_dim(&router_logits)?;
        LitGPT::check_nan(&router_probs, "moe_router_probs")?;

        // 3. Select top-k experts (still uses CPU for top-k, but greatly optimized routing)
        // Use reshape_tracked for gradient flow
        let flat_logits = router_logits.reshape_tracked(&[b_sz * seq_len, self.num_experts])?;
        LitGPT::check_nan(&flat_logits, "moe_flat_logits")?;
        let (top_k_values, top_k_indices) = self.top_k(&flat_logits, self.experts_per_tok)?;
        LitGPT::check_nan(&top_k_values, "moe_top_k_values")?;
        let top_k_weights = ops::softmax_last_dim(&top_k_values)?;
        LitGPT::check_nan(&top_k_weights, "moe_top_k_weights")?;

        // 4. OPTIMIZED: Batch-process all expert outputs, then use einsum-style combination
        // Use reshape_tracked for gradient flow
        let x_flat = x.reshape_tracked(&[b_sz * seq_len, hidden])?;

        // NOTE: We run ALL num_experts on ALL tokens, even though we only use top-k per token.
        // This is FASTER than selective computation due to GPU parallelization, but uses more memory.
        // Trade-off: ~75% wasted computation for k=2, but 10x faster than token-by-token routing.
        let mut expert_outputs = Vec::with_capacity(self.num_experts);
        for (i, expert) in self.experts.iter().enumerate() {
            let out = expert.forward(&x_flat)?;
            LitGPT::check_nan(&out, &format!("moe_expert_{}_out", i))?;
            expert_outputs.push(out);  // Each: [tokens, hidden]
        }
        // Use stack_tracked for gradient flow through expert outputs
        let all_expert_outs = Tensor::stack_tracked(&expert_outputs, 1)?;  // [tokens, num_experts, hidden]
        LitGPT::check_nan(&all_expert_outs, "moe_all_expert_outs")?;

        // GPU-native routing matrix construction (no CPU transfers!)
        // routing_weights: [tokens, num_experts] - sparse matrix with weights at top-k positions
        // Note: routing weights have zero gradients (discrete selection), gradients flow through expert outputs
        let routing_weights = self.build_routing_matrix_gpu(&top_k_indices, &top_k_weights)?;
        LitGPT::check_nan(&routing_weights, "moe_routing_weights")?;

        // Weighted sum: output[tok] = sum_over_experts(routing_weights[tok, exp] * expert_outs[tok, exp])
        // broadcasting: [tokens, experts, 1] * [tokens, experts, hidden] -> [tokens, experts, hidden]
        // Use unsqueeze_tracked for gradient flow
        let weighted_experts = all_expert_outs.broadcast_mul(&routing_weights.unsqueeze_tracked(2)?)?;  // [tokens, experts, hidden]
        LitGPT::check_nan(&weighted_experts, "moe_weighted_experts")?;

        let mut output = weighted_experts.sum(1)?;  // Sum over experts dim: [tokens, hidden]
        LitGPT::check_nan(&output, "moe_weighted_sum")?;

        // 5. Add shared expert output (always active)
        let shared_out = self.shared_expert.forward(&x_flat)?;
        LitGPT::check_nan(&shared_out, "moe_shared_expert_out")?;

        output = output.add(&shared_out)?;
        LitGPT::check_nan(&output, "moe_combined_out")?;

        // 6. Reshape back to [B, Seq, Hidden]
        // Use reshape_tracked for gradient flow
        let output = output.reshape_tracked(&[b_sz, seq_len, hidden])?;
        LitGPT::check_nan(&output, "moe_final_out")?;

        // 7. Compute load balancing loss
        // Minimize variance of expert usage to encourage even distribution
        let expert_usage = router_probs.mean(0)?.mean(0)?;  // [num_experts] - no keepdim

        // L2 loss: sum of squared usage (encourages even distribution)
        // Use contiguous_tracked to ensure proper gradient flow (avoid to_vec0 which breaks chain)
        let load_balance_loss = expert_usage.powf(2.0)?.mean_all()?.contiguous_tracked()?;

        Ok((output, load_balance_loss))
    }

    /// GPU-native top-k implementation using iterative argmax
    ///
    /// For k=2 (typical MoE), this avoids GPU→CPU→GPU transfers by:
    /// 1. Find argmax (first max index)
    /// 2. Mask that position to -inf
    /// 3. Find argmax again (second max index)
    /// 4. Gather values at those indices
    ///
    /// Performance: ~10-20x faster than CPU-based sorting for large batches
    fn top_k(&self, x: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
        let (batch_size, num_experts) = x.dims2()?;
        let device = x.device();
        let dtype = x.dtype();

        if k == 0 {
            // Edge case: no experts selected
            let empty_vals = Tensor::zeros((batch_size, 0), dtype, device)?;
            let empty_idxs = Tensor::zeros((batch_size, 0), DType::U32, device)?;
            return Ok((empty_vals, empty_idxs));
        }

        // Special optimized path for k=2 (most common case for MoE)
        if k == 2 && num_experts >= 2 {
            return self.top_2_gpu(x);
        }

        // General case: iterative argmax for arbitrary k
        // Each iteration: find max, mask it out, repeat
        let mut values_list = Vec::with_capacity(k);
        let mut indices_list = Vec::with_capacity(k);

        // Work with f32 for masking (convert back at end if needed)
        let mut masked = x.to_dtype(DType::F32)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?
            .broadcast_as((batch_size, num_experts))?;

        for _ in 0..k {
            // Find argmax along expert dimension (dim=1)
            let idx = masked.argmax_keepdim(1)?;  // [batch, 1]

            // Gather values at those indices
            let val = masked.gather(&idx, 1)?;  // [batch, 1]

            values_list.push(val);
            indices_list.push(idx.to_dtype(DType::U32)?);

            // Mask out selected indices by setting to -inf
            // Create one-hot mask: [batch, num_experts]
            let one_hot = Self::one_hot_from_indices(&idx, num_experts, device)?;
            // Where one_hot is 0 (not selected), keep original; where 1 (selected), use -inf
            let zero_tensor = Tensor::zeros(one_hot.shape().clone(), one_hot.dtype(), device)?;
            let keep_mask = one_hot.eq(&zero_tensor)?;  // U8 boolean: true where we keep original
            masked = keep_mask.where_cond(&masked, &neg_inf)?;
        }

        // Stack along k dimension: [batch, k]
        let values = Tensor::cat(&values_list, 1)?.to_dtype(dtype)?;
        let indices = Tensor::cat(&indices_list, 1)?;

        Ok((values, indices))
    }

    /// Optimized GPU-native top-2 for MoE routing
    /// Uses two argmax operations with masking - no CPU transfers
    fn top_2_gpu(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch_size, num_experts) = x.dims2()?;
        let device = x.device();
        let dtype = x.dtype();

        // Work in f32 for numerical stability with -inf masking
        let x_f32 = x.to_dtype(DType::F32)?;

        // First max
        let idx1 = x_f32.argmax_keepdim(1)?;  // [batch, 1]
        let val1 = x_f32.gather(&idx1, 1)?;   // [batch, 1]

        // Mask out first max by setting to -inf
        let one_hot1 = Self::one_hot_from_indices(&idx1, num_experts, device)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (batch_size, num_experts), device)?;
        // Where one_hot is 0 (not selected), keep x_f32; where 1 (selected), use -inf
        let zero_one_hot = Tensor::zeros(one_hot1.shape().clone(), one_hot1.dtype(), device)?;
        let keep_mask = one_hot1.eq(&zero_one_hot)?;  // U8 boolean: true where we keep original
        let masked = keep_mask.where_cond(&x_f32, &neg_inf)?;

        // Second max
        let idx2 = masked.argmax_keepdim(1)?;  // [batch, 1]
        let val2 = masked.gather(&idx2, 1)?;   // [batch, 1]

        // Stack: [batch, 2]
        let values = Tensor::cat(&[val1, val2], 1)?.to_dtype(dtype)?;
        let indices = Tensor::cat(&[idx1.to_dtype(DType::U32)?, idx2.to_dtype(DType::U32)?], 1)?;

        Ok((values, indices))
    }

    /// Create one-hot tensor from indices (GPU-native)
    /// indices: [batch, 1] with values in [0, num_classes)
    /// Returns: [batch, num_classes] with 1.0 at index positions, 0.0 elsewhere (F32)
    fn one_hot_from_indices(indices: &Tensor, num_classes: usize, device: &Device) -> Result<Tensor> {
        let batch_size = indices.dim(0)?;

        // Create range tensor [0, 1, 2, ..., num_classes-1] and broadcast to [batch, num_classes]
        // IMPORTANT: Use F32 because boostr's eq() only supports F32 tensors
        let range: Vec<f32> = (0..num_classes).map(|i| i as f32).collect();
        let range_tensor = Tensor::from_vec(range, num_classes, device)?
            .unsqueeze(0)?  // [1, num_classes]
            .broadcast_as((batch_size, num_classes))?;  // [batch, num_classes]

        // Compare with indices (convert to F32 first - boostr's eq() only supports F32)
        let indices_broadcast = indices.to_dtype(DType::F32)?
            .broadcast_as((batch_size, num_classes))?;

        // one_hot[i,j] = 1.0 if j == indices[i] else 0.0
        range_tensor.eq(&indices_broadcast)
    }

    /// GPU-native routing matrix construction
    ///
    /// Instead of transferring indices to CPU, builds routing matrix entirely on GPU
    /// using scatter-like operations with one-hot encoding.
    ///
    /// Args:
    ///   top_k_indices: [tokens, k] - indices of selected experts
    ///   top_k_weights: [tokens, k] - weights for selected experts (softmax normalized)
    ///   num_experts: total number of experts
    ///
    /// Returns:
    ///   routing_weights: [tokens, num_experts] - sparse matrix with weights at selected positions
    fn build_routing_matrix_gpu(
        &self,
        top_k_indices: &Tensor,
        top_k_weights: &Tensor,
    ) -> Result<Tensor> {
        let (num_tokens, k) = top_k_indices.dims2()?;
        let device = top_k_indices.device();
        let dtype = top_k_weights.dtype();

        // Initialize zeros: [tokens, num_experts]
        let mut routing = Tensor::zeros((num_tokens, self.num_experts), dtype, device)?;
        LitGPT::check_nan(&routing, "routing_init_zeros")?;

        // For each k position, scatter the weights to the correct expert column
        for ki in 0..k {
            // Get indices and weights for this k position
            let idx = top_k_indices.narrow(1, ki, 1)?;  // [tokens, 1]
            let wgt = top_k_weights.narrow(1, ki, 1)?;  // [tokens, 1]
            LitGPT::check_nan(&wgt, &format!("routing_wgt_k{}", ki))?;

            // Create one-hot for this k's indices: [tokens, num_experts]
            let one_hot = Self::one_hot_from_indices(&idx, self.num_experts, device)?
                .to_dtype(dtype)?;
            LitGPT::check_nan(&one_hot, &format!("routing_onehot_k{}", ki))?;

            // Multiply one-hot by weight and add to routing matrix
            // one_hot * wgt broadcasts: [tokens, num_experts] * [tokens, 1] -> [tokens, num_experts]
            let weighted_one_hot = one_hot.broadcast_mul(&wgt)?;
            LitGPT::check_nan(&weighted_one_hot, &format!("routing_weighted_k{}", ki))?;
            routing = routing.add(&weighted_one_hot)?;
            LitGPT::check_nan(&routing, &format!("routing_after_add_k{}", ki))?;
        }

        Ok(routing)
    }
}

// --- Hybrid Block (supports both Mamba and Attention) ---

use crate::mamba::MambaBlock;

// Output from a block, including optional auxiliary loss
pub struct BlockOutput {
    pub hidden: Tensor,
    pub aux_loss: Option<Tensor>,
}

enum BlockType {
    Mamba {
        block: MambaBlock,
        norm: RmsNorm,
    },
    Mamba2 {
        block: Mamba2Block,
    },
    Mamba3 {
        block: Mamba3Block,
    },
    Attention {
        attn: MultiHeadLatentAttention,
        ffn: MoeLayer,  // MoE instead of simple MLP
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
    },
    LegacyAttention {
        attn: CausalSelfAttention,
        mlp: Mlp,
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
    },
}

struct Block {
    block_type: BlockType,
}

impl Block {
    // Create appropriate block type based on configuration
    fn new(cfg: &ModelConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let eps = 1e-5;

        // Determine if this should be a Mamba layer
        let is_mamba = cfg.mamba_layers.as_ref()
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
            BlockType::Mamba3 {
                block: Mamba3Block::new(
                    cfg.hidden_size,
                    cfg.mamba2_num_heads.unwrap(),
                    cfg.mamba2_head_dim.unwrap(),
                    cfg.mamba2_state_size.unwrap(),
                    cfg.mamba2_chunk_size.unwrap(),
                    cfg.mamba2_n_groups.unwrap(),
                    cfg.mamba2_expand.unwrap(),
                    cfg.mamba3_complex_rope.unwrap_or(true),
                    cfg.mamba3_mimo_rank.unwrap_or(0),
                    cfg.mamba3_use_conv.unwrap_or(false),
                    cfg.mamba2_conv_kernel.unwrap_or(4),
                    vb.pp("mamba3"),
                )?,
            }
        } else if is_mamba && use_mamba2 {
            // Mamba2 layer (SSD - State Space Duality)
            BlockType::Mamba2 {
                block: Mamba2Block::new(
                    cfg.hidden_size,
                    cfg.mamba2_num_heads.unwrap(),
                    cfg.mamba2_head_dim.unwrap(),
                    cfg.mamba2_state_size.unwrap(),
                    cfg.mamba2_chunk_size.unwrap(),
                    cfg.mamba2_n_groups.unwrap(),
                    cfg.mamba2_conv_kernel.unwrap(),
                    cfg.mamba2_expand.unwrap(),
                    vb.pp("mamba2"),
                )?,
            }
        } else if is_mamba {
            // Mamba1 layer (original sequential)
            BlockType::Mamba {
                block: MambaBlock::new(cfg, vb.pp("mamba"))?,
                norm: rms_norm(cfg.hidden_size, eps, vb.pp("norm"))?,
            }
        } else if use_mla {
            // MLA + MoE attention layer
            BlockType::Attention {
                attn: MultiHeadLatentAttention::new(cfg, vb.pp("self_attn"))?,
                ffn: MoeLayer::new(cfg, vb.pp("moe"))?,
                input_layernorm: rms_norm(cfg.hidden_size, eps, vb.pp("input_layernorm"))?,
                post_attention_layernorm: rms_norm(cfg.hidden_size, eps, vb.pp("post_attention_layernorm"))?,
            }
        } else {
            // Legacy GQA + MLP attention layer (original nano)
            BlockType::LegacyAttention {
                attn: CausalSelfAttention::new(cfg, vb.pp("self_attn"))?,
                mlp: Mlp::new(cfg, vb.pp("mlp"))?,
                input_layernorm: rms_norm(cfg.hidden_size, eps, vb.pp("input_layernorm"))?,
                post_attention_layernorm: rms_norm(cfg.hidden_size, eps, vb.pp("post_attention_layernorm"))?,
            }
        };

        Ok(Self { block_type })
    }

    fn forward(&self, x: &Tensor) -> Result<BlockOutput> {
        match &self.block_type {
            BlockType::Mamba { block, norm } => {
                // Mamba1: single layer with normalization and residual
                let residual = x;
                let x = norm.forward(x)?;
                let x = block.forward(&x)?;
                let x = x.broadcast_add(&residual)?;

                Ok(BlockOutput {
                    hidden: x,
                    aux_loss: None,
                })
            }
            BlockType::Mamba2 { block } => {
                // Mamba2: has built-in normalization and residual
                let x = block.forward(x)?;

                Ok(BlockOutput {
                    hidden: x,
                    aux_loss: None,
                })
            }
            BlockType::Mamba3 { block } => {
                // Mamba3: has built-in normalization and residual (like Mamba2)
                let x = block.forward(x)?;

                Ok(BlockOutput {
                    hidden: x,
                    aux_loss: None,
                })
            }
            BlockType::Attention { attn, ffn, input_layernorm, post_attention_layernorm } => {
                // MLA + MoE architecture
                // Attention sub-block
                let residual = x;
                let x = input_layernorm.forward(x)?;
                LitGPT::check_nan(&x, "attn_input_norm")?;

                let x = attn.forward(&x)?;
                LitGPT::check_nan(&x, "attn_output")?;

                let x = x.broadcast_add(&residual)?;
                LitGPT::check_nan(&x, "attn_residual")?;

                // MoE sub-block
                let residual = &x;
                let x = post_attention_layernorm.forward(&x)?;
                LitGPT::check_nan(&x, "moe_input_norm")?;

                let (x, moe_loss) = ffn.forward(&x)?;  // Returns output + load balance loss
                LitGPT::check_nan(&x, "moe_output")?;

                let x = x.broadcast_add(&residual)?;
                LitGPT::check_nan(&x, "moe_residual")?;

                Ok(BlockOutput {
                    hidden: x,
                    aux_loss: Some(moe_loss),
                })
            }
            BlockType::LegacyAttention { attn, mlp, input_layernorm, post_attention_layernorm } => {
                // Legacy GQA + MLP (original nano)
                let residual = x;
                let x = input_layernorm.forward(x)?;
                let x = attn.forward(&x)?;
                let x = x.broadcast_add(&residual)?;

                let residual = &x;
                let x = post_attention_layernorm.forward(&x)?;
                let x = mlp.forward(&x)?;
                let x = x.broadcast_add(&residual)?;

                Ok(BlockOutput {
                    hidden: x,
                    aux_loss: None,
                })
            }
        }
    }
}

// --- Main LitGPT Model ---

pub struct ModelOutput {
    pub logits: Tensor,
    pub aux_losses: Vec<Tensor>,  // Collect all auxiliary losses (e.g., MoE load balancing)
}

pub struct LitGPT {
    embed_tokens: Embedding,
    layers: Vec<Block>,
    norm: RmsNorm,
    lm_head: Linear,
    /// Model dtype (F32/F16/BF16 as configured)
    /// RmsNorm and SSM computations use F32 internally for numerical stability
    #[allow(dead_code)]
    compute_dtype: DType,
}

impl LitGPT {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let embed = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::new();
        for i in 0..cfg.num_layers {
            layers.push(Block::new(cfg, i, vb.pp(&format!("layers.{}", i)))?);
        }

        let norm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;

        // Get compute dtype from config (BF16/F16/F32)
        let compute_dtype = cfg.dtype.to_dtype();

        Ok(Self {
            embed_tokens: embed,
            layers,
            norm,
            lm_head,
            compute_dtype,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<ModelOutput> {
        // Use contiguous_tracked() to ensure proper memory layout on CUDA (handles padding)
        // AND record on tape for gradient flow back to embed_tokens
        // Model uses configured dtype (F32/F16/BF16). RmsNorm and SSM ops convert to
        // F32 internally for numerical stability, then convert back.
        let mut x = self.embed_tokens.forward(input_ids)?.contiguous_tracked()?;

        // Debug: check for NaN after embedding
        Self::check_nan(&x, "embed_tokens")?;

        let mut aux_losses = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let output = layer.forward(&x)?;
            x = output.hidden.contiguous_tracked()?;  // Force contiguous after each layer

            // Debug: check for NaN after each layer
            Self::check_nan(&x, &format!("layer_{}", i))?;

            if let Some(loss) = output.aux_loss {
                aux_losses.push(loss);
            }
        }

        let x = self.norm.forward(&x)?;
        Self::check_nan(&x, "final_norm")?;

        let logits = self.lm_head.forward(&x)?;
        Self::check_nan(&logits, "lm_head")?;

        Ok(ModelOutput { logits, aux_losses })
    }

    /// Check if tensor contains NaN/Inf values (debug helper)
    /// NOTE: Disabled for performance - enable when debugging NaN issues
    #[allow(unused_variables)]
    fn check_nan(_tensor: &Tensor, _name: &str) -> Result<()> {
        // Debug code disabled for performance - uncomment to enable NaN checking:
        // if let Ok(vals) = tensor.flatten_all()?.to_vec::<f32>() {
        //     let nan_count = vals.iter().filter(|x| x.is_nan()).count();
        //     let inf_count = vals.iter().filter(|x| x.is_infinite()).count();
        //     let neg_inf_count = vals.iter().filter(|x| **x == f32::NEG_INFINITY).count();
        //     let pos_inf_count = vals.iter().filter(|x| **x == f32::INFINITY).count();
        //     // Also check for very large values (potential memory corruption)
        //     const LARGE_THRESHOLD: f32 = 1e10;
        //     let large_count = vals.iter().filter(|x| x.is_finite() && x.abs() > LARGE_THRESHOLD).count();
        //     if nan_count > 0 || inf_count > 0 || large_count > 0 {
        //         eprintln!("[NaN DEBUG] {} has {} NaN, {} Inf ({} +Inf, {} -Inf), {} LARGE (>1e10) out of {} values",
        //             name, nan_count, inf_count, pos_inf_count, neg_inf_count, large_count, vals.len());
        //         // Also print some stats
        //         let max_val = vals.iter().filter(|x| x.is_finite()).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        //         let min_val = vals.iter().filter(|x| x.is_finite()).fold(f32::INFINITY, |a, &b| a.min(b));
        //         eprintln!("[NaN DEBUG] {} finite range: [{:.4}, {:.4}]", name, min_val, max_val);
        //     }
        // }
        Ok(())
    }
}
