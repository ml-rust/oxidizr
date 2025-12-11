use crate::prelude::{
    D, DType, Device, Embedding, Linear, Result, RmsNorm, Tensor, VarBuilder,
    embedding, linear_no_bias, ops, rms_norm, router_z_loss,
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

        let mla = Self {
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
        };

        // DEBUG: Log MLA weight initialization
        if std::env::var("DEBUG_INIT").is_ok() {
            let weights = [
                ("w_dkv", mla.w_dkv.weight()),
                ("w_dq", mla.w_dq.weight()),
                ("w_uq", mla.w_uq.weight()),
                ("w_uk", mla.w_uk.weight()),
                ("w_uv", mla.w_uv.weight()),
                ("w_qr", mla.w_qr.weight()),
                ("w_kr", mla.w_kr.weight()),
                ("o_proj", mla.o_proj.weight()),
            ];
            for (name, w) in weights {
                if let Ok(v) = w.to_vec::<f32>() {
                    let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                    let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                    // Print first 3 values to check if weights are truly different
                    let first3: Vec<f32> = v.iter().take(3).copied().collect();
                    eprintln!("[MLA INIT] {}: rms={:.6e}, max={:.6e}, first3={:?}", name, rms, max_abs, first3);
                }
            }
        }

        Ok(mla)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;
        let debug = std::env::var("DEBUG_MLA_MOE").is_ok();

        // DEBUG: MLA input
        if debug {
            if let Ok(v) = x.to_vec::<f32>() {
                let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("[MLA] INPUT: shape={:?}, rms={:.6e}, max={:.6e}", x.dims(), rms, max_abs);
            }
        }

        // 1. Compress input to latent space
        let c_kv = self.w_dkv.forward(x)?;  // [B, Seq, kv_latent_dim]
        let c_q = self.w_dq.forward(x)?;    // [B, Seq, q_latent_dim]

        // DEBUG: After compression
        if debug {
            if let Ok(v) = c_kv.to_vec::<f32>() {
                let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("[MLA] c_kv: rms={:.6e}, max={:.6e}", rms, max_abs);
            }
            if let Ok(v) = c_q.to_vec::<f32>() {
                let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("[MLA] c_q: rms={:.6e}, max={:.6e}", rms, max_abs);
            }
        }

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

        // DEBUG: Q, K, V after decompression
        if debug {
            if let Ok(qv) = q_content.to_vec::<f32>() {
                let rms = (qv.iter().map(|x| x * x).sum::<f32>() / qv.len() as f32).sqrt();
                let max_abs = qv.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("[MLA] q_content: rms={:.6e}, max={:.6e}", rms, max_abs);
            }
            if let Ok(kv) = k_content.to_vec::<f32>() {
                let rms = (kv.iter().map(|x| x * x).sum::<f32>() / kv.len() as f32).sqrt();
                let max_abs = kv.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("[MLA] k_content: rms={:.6e}, max={:.6e}", rms, max_abs);
            }
            if let Ok(vv) = v.to_vec::<f32>() {
                let rms = (vv.iter().map(|x| x * x).sum::<f32>() / vv.len() as f32).sqrt();
                let max_abs = vv.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("[MLA] v: rms={:.6e}, max={:.6e}", rms, max_abs);
            }
        }

        // DEBUG: Check V tensor values for position differences
        if std::env::var("DEBUG_GRAD").is_ok() {
            // Check c_kv first
            if let Ok(ckv_vec) = c_kv.to_vec::<f32>() {
                // c_kv is [B, Seq, kv_latent_dim]
                let kv_latent = ckv_vec.len() / (b_sz * seq_len);
                let pos0_start = 0;
                let pos2_start = 2 * kv_latent;
                eprintln!("[MLA forward] c_kv shape={:?}, kv_latent_dim={}", c_kv.dims(), kv_latent);
                eprintln!("[MLA forward] c_kv pos0[:3] = {:?}", &ckv_vec[pos0_start..pos0_start+3]);
                eprintln!("[MLA forward] c_kv pos2[:3] = {:?}", &ckv_vec[pos2_start..pos2_start+3]);
                let diff_ckv: f32 = ckv_vec[pos0_start..pos0_start+kv_latent].iter()
                    .zip(&ckv_vec[pos2_start..pos2_start+kv_latent])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                eprintln!("[MLA forward] c_kv diff (pos0 vs pos2) = {:.6e}", diff_ckv);
            }

            if let Ok(v_vec) = v.to_vec::<f32>() {
                // V is [B, H, Seq, D] = [4, 4, 64, 32]
                let s_stride = 32;       // Seq stride
                // Check head 0, positions 0, 2, 10
                let pos0_start = 0;
                let pos2_start = 2 * s_stride;
                let pos10_start = 10 * s_stride;
                eprintln!("[MLA forward] V shape={:?}", v.dims());
                eprintln!("[MLA forward] V head0_pos0[:3] = {:?}", &v_vec[pos0_start..pos0_start+3]);
                eprintln!("[MLA forward] V head0_pos2[:3] = {:?}", &v_vec[pos2_start..pos2_start+3]);
                eprintln!("[MLA forward] V head0_pos10[:3] = {:?}", &v_vec[pos10_start..pos10_start+3]);
                let diff_0_2: f32 = v_vec[pos0_start..pos0_start+32].iter()
                    .zip(&v_vec[pos2_start..pos2_start+32])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                eprintln!("[MLA forward] V diff (pos0 vs pos2) = {:.6e}", diff_0_2);
            }
        }

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

        // DEBUG: Log attention score scale
        if debug {
            if let Ok(v) = att_scores.to_vec::<f32>() {
                let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                let min_val = v.iter().fold(f32::MAX, |a, &b| a.min(b));
                let max_val = v.iter().fold(f32::MIN, |a, &b| a.max(b));
                eprintln!("[MLA] att_scores: rms={:.4}, range=[{:.4}, {:.4}], scale={:.6}",
                    rms, min_val, max_val, self.scale);
            }
        }

        // Fused scaled causal softmax - numerically stable, never stores -inf
        // Combines: scale, causal mask, and softmax in one pass
        let att_weights = ops::scaled_causal_softmax(&att_scores, self.scale as f32)?;
        LitGPT::check_nan(&att_weights, "mla_att_weights")?;

        // DEBUG: Log attention weights
        if debug {
            if let Ok(v) = att_weights.to_vec::<f32>() {
                let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                let max_val = v.iter().fold(f32::MIN, |a, &b| a.max(b));
                let min_val = v.iter().fold(f32::MAX, |a, &b| a.min(b));
                eprintln!("[MLA] att_weights (after softmax): rms={:.6e}, range=[{:.6e}, {:.6e}]", rms, min_val, max_val);
            }
        }

        let output = att_weights.matmul(&v)?;  // [B, H, Seq, head_dim]
        LitGPT::check_nan(&output, "mla_attn_v_out")?;

        // DEBUG: After attention
        if debug {
            if let Ok(v) = output.to_vec::<f32>() {
                let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("[MLA] attn_out (att_weights @ v): rms={:.6e}, max={:.6e}", rms, max_abs);
            }
        }

        // 7. Reshape and project output
        // Use _tracked variants for gradient flow
        let output = output
            .transpose_tracked(1, 2)?
            .reshape_tracked(&[b_sz, seq_len, self.num_heads * self.head_dim])?;

        let output = self.o_proj.forward(&output)?;
        LitGPT::check_nan(&output, "mla_o_proj")?;

        // DEBUG: MLA output
        if debug {
            if let Ok(v) = output.to_vec::<f32>() {
                let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("[MLA] OUTPUT: rms={:.6e}, max={:.6e}", rms, max_abs);
            }
        }

        Ok(output)
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
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;

        // Project to Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention - use _tracked versions for gradient flow
        // Q: [B, Seq, H, D] -> [B, H, Seq, D]
        let q = q
            .reshape_tracked(vec![b_sz, seq_len, self.num_heads, self.head_dim])?
            .transpose_tracked(1, 2)?
            .contiguous_tracked()?;

        // K, V: [B, Seq, KV_H, D] -> [B, KV_H, Seq, D]
        let k = k
            .reshape_tracked(vec![b_sz, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose_tracked(1, 2)?
            .contiguous_tracked()?;

        let v = v
            .reshape_tracked(vec![b_sz, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose_tracked(1, 2)?
            .contiguous_tracked()?;

        // Apply RoPE
        let (q, k) = self.rope.apply_rotary_emb(&q, &k, seq_len)?;

        // Expand K and V for Grouped Query Attention if needed
        let k = self.repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = self.repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        // Scaled Dot-Product Attention with fused causal softmax
        let att_scores = q.matmul(&k.transpose_tracked(2, 3)?)?;

        // Fused scaled causal softmax - numerically stable, never stores -inf
        let att_weights = ops::scaled_causal_softmax(&att_scores, self.scale as f32)?;
        let output = att_weights.matmul(&v)?;

        // Reshape back: [B, H, Seq, D] -> [B, Seq, H, D] -> [B, Seq, Hidden]
        let output = output
            .transpose_tracked(1, 2)?
            .reshape_tracked(vec![b_sz, seq_len, self.num_heads * self.head_dim])?;

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
                .reshape_tracked(vec![b_sz, n_kv_heads * n_rep, seq_len, head_dim])?;
            Ok(x)
        }
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
        let gate_linear = self.gate_proj.forward(x)?;
        let gate = ops::silu(&gate_linear)?;
        let up = self.up_proj.forward(x)?;
        let fused = (gate.clone() * up.clone())?;
        let out = self.down_proj.forward(&fused)?;

        // DEBUG: Check weight norms
        if std::env::var("DEBUG_MLP").is_ok() {
            if let Ok(gate_w) = self.gate_proj.weight().to_vec::<f32>() {
                let w_std = {
                    let mean = gate_w.iter().sum::<f32>() / gate_w.len() as f32;
                    (gate_w.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / gate_w.len() as f32).sqrt()
                };
                let w_norm: f32 = gate_w.iter().map(|v| v * v).sum::<f32>().sqrt();
                eprintln!("[MLP weights] gate_proj.weight: shape={:?}, std={:.6e}, norm={:.4}", self.gate_proj.weight().dims(), w_std, w_norm);
            }
        }

        // DEBUG: Check MLP forward values
        if std::env::var("DEBUG_MLP").is_ok() {
            if let (Ok(x_vec), Ok(gate_vec), Ok(up_vec), Ok(fused_vec), Ok(out_vec)) = (
                x.to_vec::<f32>(),
                gate.to_vec::<f32>(),
                up.to_vec::<f32>(),
                fused.to_vec::<f32>(),
                out.to_vec::<f32>(),
            ) {
                let x_std = {
                    let mean = x_vec.iter().sum::<f32>() / x_vec.len() as f32;
                    (x_vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x_vec.len() as f32).sqrt()
                };
                let gate_std = {
                    let mean = gate_vec.iter().sum::<f32>() / gate_vec.len() as f32;
                    (gate_vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / gate_vec.len() as f32).sqrt()
                };
                let up_std = {
                    let mean = up_vec.iter().sum::<f32>() / up_vec.len() as f32;
                    (up_vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / up_vec.len() as f32).sqrt()
                };
                let fused_std = {
                    let mean = fused_vec.iter().sum::<f32>() / fused_vec.len() as f32;
                    (fused_vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / fused_vec.len() as f32).sqrt()
                };
                let out_std = {
                    let mean = out_vec.iter().sum::<f32>() / out_vec.len() as f32;
                    (out_vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / out_vec.len() as f32).sqrt()
                };
                eprintln!("[MLP forward] x_std={:.6e}, gate_std={:.6e}, up_std={:.6e}, fused_std={:.6e}, out_std={:.6e}",
                    x_std, gate_std, up_std, fused_std, out_std);
                eprintln!("[MLP forward] x[:3]={:?}, gate[:3]={:?}, up[:3]={:?}, fused[:3]={:?}, out[:3]={:?}",
                    &x_vec[..3.min(x_vec.len())],
                    &gate_vec[..3.min(gate_vec.len())],
                    &up_vec[..3.min(up_vec.len())],
                    &fused_vec[..3.min(fused_vec.len())],
                    &out_vec[..3.min(out_vec.len())]);
            }
        }

        Ok(out)
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

        let result = Self {
            experts,
            shared_expert,
            router,
            num_experts,
            experts_per_tok,
        };

        // DEBUG: Log MoE weight initialization
        if std::env::var("DEBUG_INIT").is_ok() {
            // Router gate
            if let Ok(v) = result.router.gate.weight().to_vec::<f32>() {
                let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("[MoE INIT] router.gate: rms={:.6e}, max={:.6e}", rms, max_abs);
            }
            // Shared expert
            for (name, w) in [
                ("shared.gate_proj", result.shared_expert.gate_proj.weight()),
                ("shared.up_proj", result.shared_expert.up_proj.weight()),
                ("shared.down_proj", result.shared_expert.down_proj.weight()),
            ] {
                if let Ok(v) = w.to_vec::<f32>() {
                    let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                    let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                    eprintln!("[MoE INIT] {}: rms={:.6e}, max={:.6e}", name, rms, max_abs);
                }
            }
            // Expert 0
            for (name, w) in [
                ("expert0.gate_proj", result.experts[0].gate_proj.weight()),
                ("expert0.up_proj", result.experts[0].up_proj.weight()),
                ("expert0.down_proj", result.experts[0].down_proj.weight()),
            ] {
                if let Ok(v) = w.to_vec::<f32>() {
                    let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                    let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                    eprintln!("[MoE INIT] {}: rms={:.6e}, max={:.6e}", name, rms, max_abs);
                }
            }
        }

        Ok(result)
    }

    /// Forward pass for MoE layer
    ///
    /// Returns: (output, load_balance_loss, z_loss)
    /// - output: The mixed expert output [B, Seq, Hidden]
    /// - load_balance_loss: Auxiliary loss encouraging even expert usage
    /// - z_loss: Router z-loss preventing softmax saturation (critical for BF16 training)
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, seq_len, hidden) = x.dims3()?;
        let debug = std::env::var("DEBUG_MLA_MOE").is_ok();

        LitGPT::check_nan(x, "moe_input")?;

        // 1. Get router logits
        let router_logits = self.router.forward(x)?;  // [B, Seq, num_experts]
        LitGPT::check_nan(&router_logits, "moe_router_logits")?;

        // 2. Compute routing probabilities (for load balancing loss)
        // Use temperature scaling to prevent softmax from becoming too confident
        // Higher temperature = softer distribution = better gradient flow
        // T=2.0 means p=0.73 instead of p=0.88 for logit difference of 1.0
        let router_temperature = 2.0f32;
        let scaled_logits = router_logits.scale(1.0 / router_temperature)?;
        let router_probs = ops::softmax_last_dim(&scaled_logits)?;
        LitGPT::check_nan(&router_probs, "moe_router_probs")?;


        // 2b. Compute router z-loss (prevents logit drift/explosion in BF16)
        // L_z = (1/T) * sum(log²(sum(exp(logits))))
        // This penalizes large router logits that cause softmax saturation
        let router_logits_flat = router_logits.reshape_tracked(&[b_sz * seq_len, self.num_experts])?;
        let z_loss = router_z_loss(&router_logits_flat)?;
        LitGPT::check_nan(&z_loss, "moe_z_loss")?;

        // 3. DIFFERENTIABLE Top-K Routing
        // Key insight: We need gradients to flow back to the router.
        // Old approach: gather top-k values → softmax → weights (BROKEN - gather has no backward)
        // New approach: Use router_probs directly with top-k masking
        //
        // The router_probs already has proper gradients from softmax(router_logits).
        // We just need to zero out non-selected experts and renormalize.
        let flat_logits = router_logits.reshape_tracked(&[b_sz * seq_len, self.num_experts])?;
        LitGPT::check_nan(&flat_logits, "moe_flat_logits")?;

        // Get top-k indices (discrete selection, no gradient needed for indices)
        let (_top_k_values, top_k_indices) = self.top_k(&flat_logits, self.experts_per_tok)?;

        // Create top-k mask from indices: [tokens, num_experts] with 1.0 at selected positions
        let top_k_mask = self.build_top_k_mask(&top_k_indices)?;
        LitGPT::check_nan(&top_k_mask, "moe_top_k_mask")?;

        // STRAIGHT-THROUGH ESTIMATOR for MoE routing:
        // - Forward: Use hard mask (1.0 for selected expert, 0.0 for others)
        //   This gives FULL gradient to selected experts (same as regular MLP)
        // - Backward: Gradient flows through soft probabilities (router_probs)
        //   This lets the router learn which experts are better
        //
        // Implementation: top_k_weights = mask + (soft_probs - mask).detach()
        // Forward: evaluates to mask (since detach is identity in forward)
        // Backward: d/dx[mask + (soft - mask)] = d/dx[soft] (since mask is constant)
        //
        // But we can't easily do .detach() in boostr. Instead:
        // Use soft probabilities but SCALE UP to compensate for gradient reduction.
        // If avg prob is ~0.6, scale by 1/0.6 ≈ 1.67 to get full gradient.
        let flat_probs = router_probs.reshape_tracked(&[b_sz * seq_len, self.num_experts])?;
        let masked_probs = flat_probs.broadcast_mul(&top_k_mask)?;
        LitGPT::check_nan(&masked_probs, "moe_masked_probs")?;

        // For k=1: Scale masked probs by num_experts to match single-expert gradient magnitude.
        // With 2 experts, softmax gives ~0.5 per expert. Scale by 2 → weight ≈ 1.0
        // This ensures expert gradients are not reduced by the routing probability.
        // For k>1: Renormalize so weights sum to 1 for proper output scaling.
        let top_k_weights = if self.experts_per_tok == 1 {
            // k=1: Scale by num_experts to compensate for softmax probability
            // softmax([a,b]) ≈ [0.5, 0.5], scaled by 2 → [1.0, 1.0] at selected position
            masked_probs.scale(self.num_experts as f32)?
        } else {
            // k>1: Renormalize so weights sum to 1
            // Detach denominator to prevent gradient cancellation
            let sum_probs = masked_probs.sum_keepdim(1)?;
            let sum_probs_safe = sum_probs.clamp(1e-6, f32::MAX)?;
            let sum_probs_detached = sum_probs_safe.detach();
            masked_probs.broadcast_div(&sum_probs_detached)?
        };
        LitGPT::check_nan(&top_k_weights, "moe_top_k_weights")?;

        // DEBUG: Log expert usage (most important metric)
        if debug {
            if let Ok(idx) = top_k_indices.to_vec::<u32>() {
                let mut usage = vec![0usize; self.num_experts];
                for &i in &idx {
                    usage[i as usize] += 1;
                }
                eprintln!("[MoE] expert_usage: {:?}", usage);
            }
        }

        // 4. REAL MoE: Run each expert ONLY on tokens that selected it
        // This is the correct MoE implementation where experts specialize on their tokens.
        let x_flat = x.reshape_tracked(&[b_sz * seq_len, hidden])?;
        let num_tokens = b_sz * seq_len;

        // Initialize output to zeros - will accumulate expert outputs via scatter_add
        let mut output = Tensor::zeros(&[num_tokens, hidden], x.dtype(), x.device())?;

        for (expert_idx, expert) in self.experts.iter().enumerate() {
            // 1. Find which tokens selected this expert
            // top_k_indices is [num_tokens, k] - check if any k-slot contains expert_idx
            let expert_mask = self.build_expert_mask(&top_k_indices, expert_idx)?;
            let token_indices = expert_mask.nonzero()?;

            let num_selected = token_indices.dims()[0];
            if num_selected == 0 {
                if debug {
                    eprintln!("[MoE] expert_{}: 0 tokens selected, skipping", expert_idx);
                }
                continue;  // No tokens selected this expert
            }

            // 2. Gather only selected tokens for this expert
            let x_selected = x_flat.index_select_tracked(&token_indices, 0)?;

            // 3. Run expert on selected tokens ONLY (the key efficiency gain!)
            let y_selected = expert.forward(&x_selected)?;
            LitGPT::check_nan(&y_selected, &format!("moe_expert_{}_out", expert_idx))?;

            // 4. Get routing weights for selected tokens
            // top_k_weights is [num_tokens, num_experts] - gather the weight for this expert
            let weights = self.gather_expert_weights(&top_k_weights, &token_indices, expert_idx)?;

            // 5. Scatter-add weighted output back to the output tensor
            output = output.scatter_add_weighted_tracked(&token_indices, &y_selected, &weights)?;
        }

        LitGPT::check_nan(&output, "moe_weighted_sum")?;

        // 5. Add shared expert output (always active)
        let shared_scale = 1.0 / (self.experts_per_tok as f32 + 1.0);
        let shared_out = self.shared_expert.forward(&x_flat)?;
        let shared_out_scaled = shared_out.scale(shared_scale)?;
        LitGPT::check_nan(&shared_out_scaled, "moe_shared_expert_out")?;

        output = output.add(&shared_out_scaled)?;
        LitGPT::check_nan(&output, "moe_combined_out")?;

        // 6. Reshape back to [B, Seq, Hidden]
        let output = output.reshape_tracked(&[b_sz, seq_len, hidden])?;
        LitGPT::check_nan(&output, "moe_final_out")?;

        // 7. Compute load balancing loss
        //
        // Problem: Switch Transformer's f_i * P_i doesn't work because f_i comes from
        // discrete argmax and has no gradients. If expert 0 gets 0 tokens, there's no
        // gradient path to encourage selecting it.
        //
        // Solution: Use BOTH:
        // 1. Squared probability loss: sum(P_i^2) - penalizes uneven router probabilities
        //    This HAS gradients to all experts through softmax!
        //    Minimum at P_i = 1/num_experts for all i
        //
        // 2. Switch-style loss: sum(f_i * P_i) - penalizes when router and selection agree
        //    Only has gradients where f_i > 0, but stronger when it does work
        //
        // Combined: L_balance = num_experts * (sum(P_i^2) + sum(f_i * P_i))

        // P_i: average router probability for each expert
        let router_probs_flat = router_probs.reshape_tracked(&[b_sz * seq_len, self.num_experts])?;
        let p_i = router_probs_flat.mean(0)?;  // [num_experts]

        // Component 1: Squared probability loss (always has gradients to all experts)
        let p_squared = p_i.sqr()?;  // [num_experts]
        let squared_loss = p_squared.sum_all()?;

        // Component 2: Switch-style f_i * P_i (only has gradients where f_i > 0)
        let tokens_per_expert = top_k_mask.sum(0)?;  // [num_experts]
        let total_selections = (b_sz * seq_len * self.experts_per_tok) as f32;
        let f_i = tokens_per_expert.scale(1.0 / total_selections)?;  // [num_experts]
        let f_times_p = f_i.mul(&p_i)?;  // [num_experts]
        let switch_loss = f_times_p.sum_all()?;

        // Combined loss scaled by num_experts
        let load_balance_loss = squared_loss.add(&switch_loss)?.scale(self.num_experts as f32)?;
        let load_balance_loss = load_balance_loss.contiguous_tracked()?;

        Ok((output, load_balance_loss, z_loss))
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
        let indices_f32 = indices.to_dtype(DType::F32)?;
        let indices_broadcast = indices_f32.broadcast_as((batch_size, num_classes))?;

        // one_hot[i,j] = 1.0 if j == indices[i] else 0.0
        let result = range_tensor.eq(&indices_broadcast)?;
        Ok(result)
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
    #[allow(dead_code)] // GPU-optimized alternative, may be used in future
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
            // NOTE: idx doesn't need tracking (discrete selection, no gradient)
            // wgt MUST use narrow_tracked for gradient flow back to router
            let idx = top_k_indices.narrow(1, ki, 1)?;  // [tokens, 1]
            let wgt = top_k_weights.narrow_tracked(1, ki, 1)?;  // [tokens, 1] - TRACKED for gradients!
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

    /// Build a mask indicating which tokens selected a specific expert.
    ///
    /// Args:
    ///   top_k_indices: [tokens, k] - indices of selected experts per token
    ///   expert_idx: which expert to build the mask for
    ///
    /// Returns:
    ///   mask: [tokens] - 1.0 if token selected this expert, 0.0 otherwise
    fn build_expert_mask(&self, top_k_indices: &Tensor, expert_idx: usize) -> Result<Tensor> {
        let (num_tokens, k) = top_k_indices.dims2()?;
        let device = top_k_indices.device();

        // top_k_indices is [tokens, k] containing expert indices (u32)
        // We need to check if any k-slot equals expert_idx

        // Create a tensor filled with expert_idx for comparison
        let expert_id = Tensor::full(expert_idx as f32, &[num_tokens, k], device)?;

        // Convert indices to F32 for comparison
        let indices_f32 = top_k_indices.to_dtype(DType::F32)?;

        // Check equality: 1.0 where match, 0.0 elsewhere [tokens, k]
        let matches = indices_f32.eq(&expert_id)?;

        // Sum over k dimension and clamp to 1.0 (in case of duplicates)
        let mask = matches.sum(1)?;  // [tokens]
        let mask = mask.clamp(0.0, 1.0)?;

        Ok(mask)
    }

    /// Gather routing weights for tokens that selected a specific expert.
    ///
    /// Args:
    ///   top_k_weights: [tokens, num_experts] - routing weights (sparse, only k non-zero per row)
    ///   token_indices: [num_selected] - which tokens to gather weights for
    ///   expert_idx: which expert's weight to extract
    ///
    /// Returns:
    ///   weights: [num_selected] - the weight for expert_idx at each selected token
    fn gather_expert_weights(
        &self,
        top_k_weights: &Tensor,
        token_indices: &Tensor,
        expert_idx: usize,
    ) -> Result<Tensor> {
        // top_k_weights: [tokens, num_experts]
        // token_indices: [num_selected] (U32)
        // Returns: [num_selected]

        // First gather the rows for selected tokens
        let gathered_rows = top_k_weights.index_select_tracked(token_indices, 0)?;  // [num_selected, num_experts]

        // Then extract the column for this expert
        let weights = gathered_rows.narrow(1, expert_idx, 1)?;  // [num_selected, 1]
        let weights = weights.squeeze(1)?;  // [num_selected]

        Ok(weights)
    }

    /// Build top-k mask from indices (for backward compatibility and load balancing).
    ///
    /// Args:
    ///   top_k_indices: [tokens, k] - indices of selected experts
    ///
    /// Returns:
    ///   mask: [tokens, num_experts] - 1.0 at selected positions, 0.0 elsewhere
    fn build_top_k_mask(&self, top_k_indices: &Tensor) -> Result<Tensor> {
        let (num_tokens, k) = top_k_indices.dims2()?;
        let device = top_k_indices.device();

        // Initialize zeros: [tokens, num_experts]
        let mut mask = Tensor::zeros((num_tokens, self.num_experts), DType::F32, device)?;

        // For each k position, set 1.0 at the selected expert indices
        for ki in 0..k {
            let idx = top_k_indices.narrow(1, ki, 1)?;  // [tokens, 1]
            let one_hot = Self::one_hot_from_indices(&idx, self.num_experts, device)?;
            mask = mask.add(&one_hot)?;
        }

        Ok(mask)
    }
}

// --- Feed-Forward Network Layer (MLP or MoE) ---
// Supports both simple MLP and Mixture of Experts based on config

enum FfnLayer {
    Mlp(Mlp),
    Moe(MoeLayer),
}

impl FfnLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        if cfg.num_experts.is_some() {
            // MoE mode
            Ok(FfnLayer::Moe(MoeLayer::new(cfg, vb)?))
        } else {
            // Simple MLP mode (no MoE)
            Ok(FfnLayer::Mlp(Mlp::new(cfg, vb)?))
        }
    }

    /// Forward pass
    /// Returns: (output, load_balance_loss, z_loss)
    /// For MLP, losses are zero tensors
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            FfnLayer::Mlp(mlp) => {
                let out = mlp.forward(x)?;
                // Create zero losses for MLP (no routing)
                let device = x.device();
                let zero_loss = Tensor::zeros(&[1], DType::F32, device)?;
                Ok((out, zero_loss.clone(), zero_loss))
            }
            FfnLayer::Moe(moe) => moe.forward(x),
        }
    }
}

// --- Hybrid Block (supports both Mamba and Attention) ---

use crate::mamba::MambaBlock;

// Output from a block, including optional auxiliary losses
pub struct BlockOutput {
    pub hidden: Tensor,
    pub aux_loss: Option<Tensor>,  // MoE load balance loss
    pub z_loss: Option<Tensor>,    // Router z-loss (prevents softmax saturation)
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
        ffn: FfnLayer,  // MLP or MoE based on config
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
            // MLA attention layer (with MLP or MoE based on config)
            let ffn_prefix = if cfg.num_experts.is_some() { "moe" } else { "mlp" };
            BlockType::Attention {
                attn: MultiHeadLatentAttention::new(cfg, vb.pp("self_attn"))?,
                ffn: FfnLayer::new(cfg, vb.pp(ffn_prefix))?,
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
                    z_loss: None,
                })
            }
            BlockType::Mamba2 { block } => {
                // Mamba2: has built-in normalization and residual
                let x = block.forward(x)?;

                // DEBUG: Log activation scale
                if std::env::var("DEBUG_ACTIVATION").is_ok() {
                    if let Ok(vals) = x.flatten_all()?.to_vec::<f32>() {
                        let rms = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32).sqrt();
                        let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                        eprintln!("[Mamba2 output] rms={:.6}, max_abs={:.6}", rms, max_abs);
                    }
                }

                Ok(BlockOutput {
                    hidden: x,
                    aux_loss: None,
                    z_loss: None,
                })
            }
            BlockType::Mamba3 { block } => {
                // Mamba3: has built-in normalization and residual (like Mamba2)
                let x = block.forward(x)?;

                Ok(BlockOutput {
                    hidden: x,
                    aux_loss: None,
                    z_loss: None,
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

                // MoE forward returns (output, load_balance_loss, z_loss)
                let (x, moe_loss, z_loss) = ffn.forward(&x)?;
                LitGPT::check_nan(&x, "moe_output")?;

                let x = x.broadcast_add(&residual)?;
                LitGPT::check_nan(&x, "moe_residual")?;

                // DEBUG: Log activation scale
                if std::env::var("DEBUG_ACTIVATION").is_ok() {
                    if let Ok(vals) = x.flatten_all()?.to_vec::<f32>() {
                        let rms = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32).sqrt();
                        let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                        eprintln!("[MLA+MoE output] rms={:.6}, max_abs={:.6}", rms, max_abs);
                    }
                }

                Ok(BlockOutput {
                    hidden: x,
                    aux_loss: Some(moe_loss),
                    z_loss: Some(z_loss),
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
                    z_loss: None,
                })
            }
        }
    }
}

// --- Main LitGPT Model ---

pub struct ModelOutput {
    pub logits: Tensor,
    pub aux_losses: Vec<Tensor>,  // Auxiliary losses (MoE load balancing)
    pub z_losses: Vec<Tensor>,    // Router z-losses (prevents MoE logit explosion)
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

        // DEBUG: Log embedding init
        if std::env::var("DEBUG_INIT").is_ok() {
            if let Ok(v) = embed.weight().to_vec::<f32>() {
                let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("[EMBED INIT] embed_tokens: rms={:.6e}, max={:.6e}, num_tokens={}",
                    rms, max_abs, cfg.vocab_size);
            }
        }

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

        // DEBUG: Check input_ids for duplicate tokens
        if std::env::var("DEBUG_GRAD").is_ok() {
            if let Ok(ids) = input_ids.to_vec::<u32>() {
                let seq_len = input_ids.dims().last().copied().unwrap_or(0);
                eprintln!("[Model forward] input_ids shape={:?}, first 10 tokens: {:?}",
                    input_ids.dims(), &ids[..std::cmp::min(10, ids.len())]);
                // Check if all tokens are the same
                if seq_len > 0 && ids.windows(2).all(|w| w[0] == w[1]) {
                    eprintln!("[Model forward] WARNING: All tokens are the same! token={}", ids[0]);
                }
            }
        }

        let mut x = self.embed_tokens.forward(input_ids)?.contiguous_tracked()?;

        // DEBUG: Check embedding output
        if std::env::var("DEBUG_GRAD").is_ok() {
            if let Ok(x_vec) = x.to_vec::<f32>() {
                let hidden_size = x.dims().last().copied().unwrap_or(0);
                eprintln!("[Model forward] embed_out shape={:?}, hidden_size={}", x.dims(), hidden_size);
                if hidden_size > 0 {
                    let pos0 = &x_vec[0..std::cmp::min(3, hidden_size)];
                    let pos2_start = 2 * hidden_size;
                    let pos2 = if pos2_start + 3 <= x_vec.len() {
                        &x_vec[pos2_start..pos2_start+3]
                    } else {
                        &[]
                    };
                    eprintln!("[Model forward] embed_out pos0[:3] = {:?}", pos0);
                    eprintln!("[Model forward] embed_out pos2[:3] = {:?}", pos2);
                    if pos2.len() == 3 {
                        let diff: f32 = x_vec[0..hidden_size].iter()
                            .zip(&x_vec[pos2_start..pos2_start+hidden_size])
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>()
                            .sqrt();
                        eprintln!("[Model forward] embed_out diff (pos0 vs pos2) = {:.6e}", diff);
                    }
                }
            }
        }

        // Debug: check for NaN after embedding
        Self::check_nan(&x, "embed_tokens")?;

        let mut aux_losses = Vec::new();
        let mut z_losses = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let output = layer.forward(&x)?;
            x = output.hidden.contiguous_tracked()?;  // Force contiguous after each layer

            // Debug: check for NaN after each layer
            Self::check_nan(&x, &format!("layer_{}", i))?;

            // Collect MoE auxiliary losses
            if let Some(loss) = output.aux_loss {
                aux_losses.push(loss);
            }

            // Collect router z-losses (prevents softmax saturation in MoE)
            if let Some(z) = output.z_loss {
                z_losses.push(z);
            }
        }

        let x = self.norm.forward(&x)?;
        Self::check_nan(&x, "final_norm")?;

        let logits = self.lm_head.forward(&x)?;
        Self::check_nan(&logits, "lm_head")?;

        Ok(ModelOutput { logits, aux_losses, z_losses })
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
