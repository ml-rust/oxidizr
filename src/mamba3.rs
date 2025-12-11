// Mamba3 Implementation
// Based on "Mamba-3: Improved Sequence Modeling Using State Space Principles" (ICLR 2026)
//
// Key innovations over Mamba-2:
// 1. Trapezoidal Discretization - More expressive recurrence (replaces short conv)
// 2. Complex-Valued SSM via RoPE - Data-dependent rotary embeddings for state tracking
// 3. MIMO (Multi-Input Multi-Output) - Increased arithmetic intensity for inference
//
// Dimension notation conventions:
// - B: batch size
// - L or seq_len: sequence length
// - H or num_heads: number of attention heads
// - D or head_dim: dimension per head
// - N or state_size: SSM state dimension
// - C or chunk_size: chunk size for SSD decomposition
// - R or mimo_rank: MIMO rank (0 = SISO)

use crate::prelude::{
    Conv1d, Conv1dConfig, D, DType, Error, LayerNorm, Linear, Result, Tensor, VarBuilder,
    conv1d, layer_norm, linear, ops,
};

// Numerical stability constants (same as Mamba2)
const EPS: f64 = 1e-6;
const MAX_EXP_INPUT: f32 = 50.0;
const MIN_EXP_INPUT: f32 = -50.0;
const MASK_NEG_VALUE: f32 = -1e9;

/// Softplus activation: softplus(x) = log(1 + exp(x))
fn softplus(x: &Tensor, beta: f64) -> Result<Tensor> {
    let beta_x = x.affine(beta, 0.0)?;
    let exp = beta_x.exp()?;
    let one_plus_exp = exp.affine(1.0, 1.0)?;
    one_plus_exp.log()?.affine(1.0 / beta, 0.0)
}

/// Pad tensor on a specific dimension
fn pad_1d(tensor: &Tensor, padding: (usize, usize), dim: usize, value: f32) -> Result<Tensor> {
    let (left, right) = padding;
    if left == 0 && right == 0 {
        return Ok(tensor.clone());
    }

    let shape = tensor.dims();
    let dtype = tensor.dtype();
    let mut left_shape = shape.to_vec();
    let mut right_shape = shape.to_vec();
    left_shape[dim] = left;
    right_shape[dim] = right;

    let mut parts = Vec::new();
    if left > 0 {
        parts.push(Tensor::full(value, left_shape.as_slice(), tensor.device())?.to_dtype(dtype)?);
    }
    parts.push(tensor.clone());
    if right > 0 {
        parts.push(Tensor::full(value, right_shape.as_slice(), tensor.device())?.to_dtype(dtype)?);
    }

    if parts.len() == 1 {
        Ok(parts.into_iter().next().unwrap())
    } else {
        Tensor::cat(&parts.iter().collect::<Vec<_>>(), dim)
    }
}

/// Pad tensor on seq_len dimension (dim=1)
fn pad_tensor_by_size(input: &Tensor, pad_size: usize) -> Result<Tensor> {
    if pad_size == 0 {
        return Ok(input.clone());
    }

    let shape = input.dims();
    match shape.len() {
        3 => {
            let zeros = Tensor::zeros(&[shape[0], pad_size, shape[2]], input.dtype(), input.device())?;
            Tensor::cat(&[input, &zeros], 1)
        }
        4 => {
            let zeros = Tensor::zeros(&[shape[0], pad_size, shape[2], shape[3]], input.dtype(), input.device())?;
            Tensor::cat(&[input, &zeros], 1)
        }
        _ => return Err(Error::Msg(format!("pad_tensor_by_size: unsupported tensor rank {}", shape.len()))),
    }
}

/// Reshape tensor into chunks for SSD algorithm
fn reshape_into_chunks(input: &Tensor, pad_size: usize, chunk_size: usize) -> Result<Tensor> {
    let padded = pad_tensor_by_size(input, pad_size)?;
    let shape = padded.dims();

    match shape.len() {
        3 => {
            let (b, l, d) = (shape[0], shape[1], shape[2]);
            padded.reshape(&[b, l / chunk_size, chunk_size, d])
        }
        4 => {
            let (b, l, h, d) = (shape[0], shape[1], shape[2], shape[3]);
            padded.reshape(&[b, l / chunk_size, chunk_size, h, d])
        }
        _ => return Err(Error::Msg(format!("reshape_into_chunks: unsupported tensor rank {}", shape.len()))),
    }
}

/// Stable segment sum using cumsum difference method
fn segment_sum(input: &Tensor) -> Result<Tensor> {
    let shape = input.dims();
    let chunk_size = shape[shape.len() - 1];

    if chunk_size == 0 {
        return Err(Error::Msg("segment_sum: chunk_size cannot be zero".to_string()));
    }

    let input = input.contiguous_tracked()?;
    let cumsum = input.cumsum(D::Minus1)?;

    let batch_dims: Vec<usize> = shape[..shape.len()-1].to_vec();
    let batch_size: usize = if batch_dims.is_empty() { 1 } else { batch_dims.iter().product() };

    let cumsum_flat = cumsum.reshape(&[batch_size, chunk_size])?;

    let mut diff_rows = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let row = cumsum_flat.narrow(0, b, 1)?;
        let row_i = row.transpose(0, 1)?;
        let row_j = row.clone();
        let diff = row_i.broadcast_sub(&row_j)?
            .clamp(MIN_EXP_INPUT, MAX_EXP_INPUT)?
            .contiguous_tracked()?;
        diff_rows.push(diff);
    }

    let refs: Vec<&Tensor> = diff_rows.iter().collect();
    let diff = Tensor::stack(&refs, 0)?;

    let mask_data: Vec<f32> = (0..chunk_size)
        .flat_map(|i| (0..chunk_size).map(move |j| if j <= i { 1.0 } else { 0.0 }))
        .collect();
    let device = diff.device();
    let dtype = diff.dtype();
    let mask = Tensor::from_vec(mask_data, &[1, chunk_size, chunk_size], device)?.to_dtype(dtype)?;

    let neg_large = Tensor::full(MASK_NEG_VALUE, &[1, chunk_size, chunk_size], device)?.to_dtype(dtype)?;

    let masked_diff = diff.broadcast_mul(&mask)?;
    let one_minus_mask = mask.affine(-1.0, 1.0)?;
    let masked_neg = neg_large.broadcast_mul(&one_minus_mask)?;
    let result = masked_diff.broadcast_add(&masked_neg)?;

    let mut final_shape = batch_dims;
    final_shape.push(chunk_size);
    final_shape.push(chunk_size);
    result.reshape(final_shape.as_slice())
}

/// Gated RMS Normalization (same as Mamba2)
#[derive(Debug, Clone)]
pub struct MambaRMSNormGated {
    weight: Tensor,
    eps: f64,
}

impl MambaRMSNormGated {
    pub fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(hidden_size, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, hidden_states: &Tensor, gate: Option<&Tensor>) -> Result<Tensor> {
        let dtype = hidden_states.dtype();
        let mut hs = hidden_states.to_dtype(DType::F32)?;

        if let Some(g) = gate {
            let g_f32 = g.to_dtype(DType::F32)?;
            let g_silu = ops::silu(&g_f32)?;
            hs = hs.broadcast_mul(&g_silu)?;
        }

        let variance = hs.sqr()?.mean_keepdim(D::Minus1)?;
        let eps_tensor = Tensor::from_slice(&[self.eps as f32], 1, hs.device())?;
        let var_eps = variance.broadcast_add(&eps_tensor)?;
        let hs_normed = hs.broadcast_div(&var_eps.sqrt()?)?;

        let result = hs_normed.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?.to_dtype(dtype)?;
        Ok(result)
    }
}

/// RMS Normalization for QK-style normalization on B and C
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;

        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let eps_tensor = Tensor::from_slice(&[self.eps as f32], 1, x.device())?;
        let var_eps = variance.broadcast_add(&eps_tensor)?;
        let x_normed = x_f32.broadcast_div(&var_eps.sqrt()?)?;

        let result = x_normed.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?.to_dtype(dtype)?;
        Ok(result)
    }
}

/// Mamba3 Mixer - Core implementation with trapezoidal discretization, complex RoPE, and MIMO
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Mamba3Mixer {
    // Configuration
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
    ssm_state_size: usize,
    chunk_size: usize,
    n_groups: usize,
    time_step_min: f64,
    time_step_max: f64,
    use_complex_rope: bool,
    mimo_rank: usize,
    use_conv: bool,

    // Learnable parameters - Projections
    in_proj: Linear,
    out_proj: Linear,

    // Trapezoidal discretization: λ projection
    lambda_proj: Linear,

    // Complex RoPE: θ projection (optional)
    theta_proj: Option<Linear>,

    // Learnable biases for B and C (Mamba-3 innovation, initialized to 1.0)
    b_bias: Tensor,
    c_bias: Tensor,

    // Standard SSM components
    dt_bias: Tensor,
    a_log: Tensor,
    d: Tensor,

    // QK-style normalization for B and C (repositioned in Mamba-3)
    bc_norm: RmsNorm,

    // Gated output norm
    norm: MambaRMSNormGated,

    // Optional Conv1D (disabled by default in Mamba-3)
    conv1d: Option<Conv1d>,
    conv_dim: Option<usize>,

    // MIMO projections (optional)
    mimo_x_up: Option<Linear>,
    mimo_x_down: Option<Linear>,
}

impl Mamba3Mixer {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        state_size: usize,
        chunk_size: usize,
        n_groups: usize,
        expand: usize,
        use_complex_rope: bool,
        mimo_rank: usize,
        use_conv: bool,
        conv_kernel: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Validate state_size is even when using complex RoPE (required for real/imag pairs)
        if use_complex_rope && state_size % 2 != 0 {
            return Err(Error::Msg(format!(
                "Mamba3 with complex RoPE requires even state_size, got {}. \
                 RoPE operates on (real, imag) pairs which requires N to be divisible by 2.",
                state_size
            )));
        }

        let intermediate_size = hidden_size * expand;

        // Calculate projection sizes
        // Without conv: we don't need conv_dim in projection
        // Project to: gate, hidden_states, B, C, dt
        let bc_size = n_groups * state_size;
        let projection_size = intermediate_size +  // gate
                              intermediate_size +  // hidden_states
                              bc_size +            // B
                              bc_size +            // C
                              num_heads;           // dt

        // Input projection
        let in_proj = linear(hidden_size, projection_size, vb.pp("in_proj"))?;

        // Output projection
        let out_proj = linear(intermediate_size, hidden_size, vb.pp("out_proj"))?;

        // Trapezoidal discretization: λ projection
        // Projects input to per-head mixing parameter
        let lambda_proj = linear(hidden_size, num_heads, vb.pp("lambda_proj"))?;

        // Complex RoPE: θ projection (projects to state_size/2 frequencies per head)
        let theta_proj = if use_complex_rope {
            Some(linear(hidden_size, num_heads * (state_size / 2), vb.pp("theta_proj"))?)
        } else {
            None
        };

        // Learnable BC biases (Mamba-3 innovation)
        // Initialized to 1.0 as per paper ablations
        // Load from checkpoint or initialize with ones
        let b_bias = vb.get_with_shape("b_bias", &[num_heads, state_size])?;
        let c_bias = vb.get_with_shape("c_bias", &[num_heads, state_size])?;

        // Time step bias
        let dt_bias = vb.get_with_shape("dt_bias", &[num_heads])?;

        // S4D initialization for A: A_log initialized to log(1..num_heads)
        let a_log = vb.get_with_shape("A_log", &[num_heads])?;

        // D skip connection
        let d = vb.get_with_shape("D", &[num_heads])?;

        // QK-style normalization for B and C (repositioned in Mamba-3)
        let bc_norm = RmsNorm::new(state_size, EPS, vb.pp("bc_norm"))?;

        // Gated RMS norm
        let norm = MambaRMSNormGated::new(intermediate_size, EPS, vb.pp("norm"))?;

        // Optional Conv1D (disabled by default in Mamba-3)
        let (conv1d, conv_dim) = if use_conv {
            let conv_dim = intermediate_size + 2 * bc_size;
            let conv_cfg = Conv1dConfig {
                groups: conv_dim,
                padding: conv_kernel - 1,
                ..Default::default()
            };
            let conv = conv1d(conv_dim, conv_dim, conv_kernel, conv_cfg, vb.pp("conv1d"))?;
            (Some(conv), Some(conv_dim))
        } else {
            (None, None)
        };

        // MIMO projections (optional)
        let (mimo_x_up, mimo_x_down) = if mimo_rank > 0 {
            let up = linear(head_dim, head_dim * mimo_rank, vb.pp("mimo_x_up"))?;
            let down = linear(head_dim * mimo_rank, head_dim, vb.pp("mimo_x_down"))?;
            (Some(up), Some(down))
        } else {
            (None, None)
        };

        Ok(Self {
            num_heads,
            head_dim,
            hidden_size,
            intermediate_size,
            ssm_state_size: state_size,
            chunk_size,
            n_groups,
            time_step_min: 0.001,
            time_step_max: 4.0,  // Critical: must be 2.0-15.0, NOT 0.1 (causes gradient explosion)
            use_complex_rope,
            mimo_rank,
            use_conv,
            in_proj,
            out_proj,
            lambda_proj,
            theta_proj,
            b_bias,
            c_bias,
            dt_bias,
            a_log,
            d,
            bc_norm,
            norm,
            conv1d,
            conv_dim,
            mimo_x_up,
            mimo_x_down,
        })
    }

    /// Forward pass
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let dtype = hidden_states.dtype();

        // 1. Input projection
        let projected = self.in_proj.forward(hidden_states)?;

        // Split projections: gate, x, B, C, dt
        let bc_size = self.n_groups * self.ssm_state_size;
        let splits = vec![
            self.intermediate_size,  // gate
            self.intermediate_size,  // x (hidden_states)
            bc_size,                 // B
            bc_size,                 // C
            self.num_heads,          // dt
        ];
        let parts = self.split_projection(&projected, &splits)?;
        let (gate, x, mut b, mut c, dt) = (
            &parts[0], &parts[1], parts[2].clone(), parts[3].clone(), &parts[4]
        );

        // 2. Optional Conv1D (if enabled)
        let x = if let Some(ref conv) = self.conv1d {
            // Apply conv to concatenated [x, B, C]
            let x_b_c = Tensor::cat(&[x, &b, &c], D::Minus1)?;
            let x_b_c_t = x_b_c.transpose(1, 2)?;
            let conv_out = conv.forward(&x_b_c_t)?;
            let conv_out = conv_out.narrow(2, 0, seq_len)?;
            let conv_out = conv_out.transpose(1, 2)?;
            let conv_out = ops::silu(&conv_out)?.to_dtype(dtype)?;

            // Split back
            let x_conv = conv_out.narrow(D::Minus1, 0, self.intermediate_size)?;
            b = conv_out.narrow(D::Minus1, self.intermediate_size, bc_size)?;
            c = conv_out.narrow(D::Minus1, self.intermediate_size + bc_size, bc_size)?;
            x_conv
        } else {
            // No conv - apply SiLU directly to x
            ops::silu(x)?.to_dtype(dtype)?
        };

        // 3. Compute trapezoidal mixing parameter λ
        let lambda = ops::sigmoid(&self.lambda_proj.forward(hidden_states)?)?;
        // λ: [B, L, num_heads]

        // 4. Reshape B and C for multi-head processing
        let mut b_reshaped = b.reshape(&[batch_size, seq_len, self.n_groups, self.ssm_state_size])?;
        let mut c_reshaped = c.reshape(&[batch_size, seq_len, self.n_groups, self.ssm_state_size])?;

        // Repeat B and C for all heads
        let repeat_factor = self.num_heads / self.n_groups;
        b_reshaped = self.repeat_interleave(&b_reshaped, repeat_factor, 2)?;
        c_reshaped = self.repeat_interleave(&c_reshaped, repeat_factor, 2)?;
        // B, C: [B, L, num_heads, state_size]

        // 5. QK-Norm on B and C (repositioned in Mamba-3)
        b_reshaped = self.apply_bc_norm(&b_reshaped)?;
        c_reshaped = self.apply_bc_norm(&c_reshaped)?;

        // 6. Add learnable biases (Mamba-3 innovation)
        b_reshaped = b_reshaped.broadcast_add(&self.b_bias.reshape(&[1, 1, self.num_heads, self.ssm_state_size])?)?;
        c_reshaped = c_reshaped.broadcast_add(&self.c_bias.reshape(&[1, 1, self.num_heads, self.ssm_state_size])?)?;

        // 7. Compute dt and discretization
        let dt = dt.broadcast_add(&self.dt_bias.reshape(&[1, 1, self.num_heads])?)?;
        let dt = softplus(&dt, 1.0)?.to_dtype(dtype)?;
        let dt = dt.clamp(self.time_step_min as f32, self.time_step_max as f32)?;

        // 8. Complex RoPE (if enabled)
        if self.use_complex_rope {
            if let Some(ref theta_proj) = self.theta_proj {
                // Compute rotation angles
                let theta = theta_proj.forward(hidden_states)?;
                // theta: [B, L, num_heads * (state_size/2)]
                let theta = theta.reshape(&[batch_size, seq_len, self.num_heads, self.ssm_state_size / 2])?;

                // Scale by dt and compute cumulative angles
                let theta_scaled = theta.broadcast_mul(&dt.unsqueeze(D::Minus1)?)?;
                let angles = theta_scaled.cumsum(1)?;
                // angles: [B, L, num_heads, state_size/2]

                // Apply RoPE to B and C
                b_reshaped = self.apply_rope(&b_reshaped, &angles)?;
                c_reshaped = self.apply_rope(&c_reshaped, &angles)?;
            }
        }

        // 9. Reshape x for multi-head processing
        let x_reshaped = x.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?;

        // 10. MIMO processing (if enabled)
        let x_for_ssm = if self.mimo_rank > 0 {
            if let Some(ref up_proj) = self.mimo_x_up {
                // Expand: [B, L, H, D] -> [B, L, H, D*R]
                let x_flat = x_reshaped.reshape(&[batch_size * seq_len * self.num_heads, self.head_dim])?;
                let x_up = up_proj.forward(&x_flat)?;
                x_up.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim * self.mimo_rank])?
            } else {
                x_reshaped.clone()
            }
        } else {
            x_reshaped.clone()
        };

        // 11. SSM transformation with trapezoidal discretization
        let y = self.ssd_forward_trapezoidal(
            &x_for_ssm,
            &b_reshaped,
            &c_reshaped,
            &dt,
            &lambda,
        )?.to_dtype(dtype)?;

        // 12. MIMO down-projection (if enabled)
        let y = if self.mimo_rank > 0 {
            if let Some(ref down_proj) = self.mimo_x_down {
                let y_flat = y.reshape(&[batch_size * seq_len * self.num_heads, self.head_dim * self.mimo_rank])?;
                let y_down = down_proj.forward(&y_flat)?;
                y_down.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            } else {
                y
            }
        } else {
            y
        };

        // 13. Reshape output
        let y = y.reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        // 14. Gated normalization
        let scan_output = self.norm.forward(&y, Some(gate))?.to_dtype(dtype)?;

        // 15. Output projection
        self.out_proj.forward(&scan_output)
    }

    /// Apply QK-style RMS normalization to B or C
    fn apply_bc_norm(&self, tensor: &Tensor) -> Result<Tensor> {
        // tensor: [B, L, num_heads, state_size]
        let shape = tensor.dims();
        let (b, l, h, n) = (shape[0], shape[1], shape[2], shape[3]);

        // Reshape to apply norm per position
        let flat = tensor.reshape(&[b * l * h, n])?;
        let normed = self.bc_norm.forward(&flat)?;
        normed.reshape(&[b, l, h, n])
    }

    /// Apply RoPE (rotary position embedding) to tensor (Mamba-3 Propositions 2-4)
    ///
    /// # Complex-Valued SSM via RoPE Trick
    ///
    /// Instead of explicit block-diagonal rotation matrices, Mamba-3 absorbs rotations
    /// into the B and C matrices using cumulative position-dependent angles:
    /// ```text
    /// B̄_t = (∏_{i=0}^{t} R_i^T) · B_t
    /// C̄_t = (∏_{i=0}^{t} R_i^T) · C_t
    ///
    /// where R(θ) = [[cos(θ), -sin(θ)],
    ///               [sin(θ),  cos(θ)]]
    /// ```
    ///
    /// This enables the SSM to track state across positions (critical for tasks like
    /// parity checking and modular arithmetic that Mamba-2 struggles with).
    ///
    /// # Implementation
    ///
    /// Treats state_size N as N/2 pairs of (real, imag) coordinates and applies
    /// 2D rotation to each pair using the cumulative angles.
    fn apply_rope(&self, tensor: &Tensor, angles: &Tensor) -> Result<Tensor> {
        // tensor: [B, L, num_heads, state_size]
        // angles: [B, L, num_heads, state_size/2]

        let shape = tensor.dims();
        let (b, l, h, n) = (shape[0], shape[1], shape[2], shape[3]);
        let half_n = n / 2;

        // Split tensor into pairs (real, imag)
        // Reshape to [B, L, H, N/2, 2]
        let tensor_pairs = tensor.reshape(&[b, l, h, half_n, 2])?;
        let real = tensor_pairs.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let imag = tensor_pairs.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;

        // Compute cos and sin
        let cos = angles.cos()?;
        let sin = angles.sin()?;

        // Apply rotation:
        // real' = real * cos - imag * sin
        // imag' = real * sin + imag * cos
        let real_new = real.broadcast_mul(&cos)?.broadcast_sub(&imag.broadcast_mul(&sin)?)?;
        let imag_new = real.broadcast_mul(&sin)?.broadcast_add(&imag.broadcast_mul(&cos)?)?;

        // Stack back
        let real_unsqueezed = real_new.unsqueeze(D::Minus1)?;
        let imag_unsqueezed = imag_new.unsqueeze(D::Minus1)?;
        let rotated = Tensor::cat(&[&real_unsqueezed, &imag_unsqueezed], D::Minus1)?;

        // Reshape back to [B, L, H, N]
        rotated.reshape(&[b, l, h, n])
    }

    /// Core SSD algorithm with trapezoidal discretization (Mamba-3 Proposition 1)
    ///
    /// # Trapezoidal Discretization (Generalized)
    ///
    /// Standard trapezoidal rule approximates integrals as: (Δt/2) * (f(t-1) + f(t))
    ///
    /// Mamba-3 generalizes this with a learned, data-dependent mixing parameter λ:
    /// ```text
    /// h_t = α_t·h_{t-1} + β_t·B_{t-1}·x_{t-1} + γ_t·B_t·x_t
    ///
    /// where:
    ///   α_t = exp(Δ_t · A)           -- decay factor
    ///   β_t = (1-λ_t) · Δ_t · α_t    -- weight for previous contribution
    ///   γ_t = λ_t · Δ_t              -- weight for current contribution
    ///   λ_t = sigmoid(proj(x_t))     -- data-dependent mixing (0=previous, 1=current)
    /// ```
    ///
    /// This allows the model to adaptively weight current vs previous input contributions,
    /// replacing the fixed short convolution from Mamba-2 with a more expressive recurrence.
    ///
    /// # MIMO Support
    ///
    /// When mimo_rank > 0, hidden_states has shape [B, L, H, D*R] where R is the MIMO rank.
    /// This increases arithmetic intensity, converting memory-bound inference to compute-bound.
    fn ssd_forward_trapezoidal(
        &self,
        hidden_states: &Tensor,  // [B, L, H, D] or [B, L, H, D*R] for MIMO
        b: &Tensor,              // [B, L, H, N]
        c: &Tensor,              // [B, L, H, N]
        dt: &Tensor,             // [B, L, H]
        lambda: &Tensor,         // [B, L, H] - trapezoidal mixing parameter
    ) -> Result<Tensor> {
        let shape = hidden_states.dims();
        let (batch_size, seq_len, num_heads, dim) = (shape[0], shape[1], shape[2], shape[3]);
        let input_dtype = hidden_states.dtype();

        // Compute A (always negative for stability)
        let a_exp = self.a_log.to_dtype(DType::F32)?.exp()?;
        let a_clamped = a_exp.clamp(1e-6_f32, 1e6_f32)?;
        let a = a_clamped.affine(-1.0, 0.0)?.to_dtype(input_dtype)?;

        // Compute discretization coefficients
        // α_t = exp(Δ_t · A)
        let a_discrete = a.reshape(&[1, 1, num_heads])?.broadcast_mul(dt)?;
        // [B, L, H]

        // β_t = (1 - λ_t) · Δ_t · exp(Δ_t · A)
        let one_minus_lambda = lambda.affine(-1.0, 1.0)?;
        let beta = one_minus_lambda.broadcast_mul(dt)?.broadcast_mul(&a_discrete.clamp(MIN_EXP_INPUT, MAX_EXP_INPUT)?.exp()?)?;

        // γ_t = λ_t · Δ_t
        let gamma = lambda.broadcast_mul(dt)?;

        // Scale hidden_states by γ for current contribution
        let x_gamma = hidden_states.broadcast_mul(&gamma.unsqueeze(D::Minus1)?)?;

        // For trapezoidal, we also need x_{t-1} scaled by β
        // Shift x right by 1 position (pad with zeros at start)
        let x_prev = self.shift_right(hidden_states)?;
        let x_beta = x_prev.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;

        // Similarly for B
        let b_prev = self.shift_right(b)?;

        // Calculate padding for chunks
        let pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size;

        // D residual
        let d_residual = self.d.reshape(&[1, 1, num_heads, 1])?
            .broadcast_mul(&pad_tensor_by_size(hidden_states, pad_size)?)?;

        // Reshape into chunks for parallel processing
        let x_gamma_chunks = reshape_into_chunks(&x_gamma, pad_size, self.chunk_size)?;
        let x_beta_chunks = reshape_into_chunks(&x_beta, pad_size, self.chunk_size)?;
        let a_chunks = reshape_into_chunks(&a_discrete, pad_size, self.chunk_size)?;
        let b_chunks = reshape_into_chunks(b, pad_size, self.chunk_size)?;
        let b_prev_chunks = reshape_into_chunks(&b_prev, pad_size, self.chunk_size)?;
        let c_chunks = reshape_into_chunks(c, pad_size, self.chunk_size)?;

        // Permute A for processing
        let a_chunks = a_chunks.permute((0, 3, 1, 2))?.contiguous_tracked()?;
        let a_cumsum = a_chunks.cumsum(D::Minus1)?;

        // 1. Intra-chunk computation with trapezoidal mask
        // The mask L = L_decay * L_conv (Eq. 5 in paper)
        let l_decay_pre_exp = segment_sum(&a_chunks)?.clamp(MIN_EXP_INPUT, MAX_EXP_INPUT)?;
        let l_decay = l_decay_pre_exp.exp()?.contiguous_tracked()?;

        // Compute G (attention-like weights)
        let g = self.compute_g(&c_chunks, &b_chunks)?.contiguous_tracked()?;

        // Compute M: G * L_decay
        let l_perm = l_decay.permute((0, 2, 3, 4, 1))?.contiguous_tracked()?;
        let m = g.broadcast_mul(&l_perm)?.contiguous_tracked()?;

        // Compute Y_diag for current contribution (γ · B · x)
        let y_diag_gamma = self.compute_y_diag(&m, &x_gamma_chunks)?;

        // Compute Y_diag for previous contribution (β · B_{t-1} · x_{t-1})
        // Use shifted masks/B for the trapezoidal term
        let g_prev = self.compute_g(&c_chunks, &b_prev_chunks)?.contiguous_tracked()?;
        let m_prev = g_prev.broadcast_mul(&l_perm)?.contiguous_tracked()?;
        let y_diag_beta = self.compute_y_diag(&m_prev, &x_beta_chunks)?;

        // Combined diagonal: γ contribution + β contribution
        let y_diag = y_diag_gamma.broadcast_add(&y_diag_beta)?;

        // 2. Compute states for inter-chunk recurrence
        // For true trapezoidal rule, chunk states must incorporate BOTH gamma and beta contributions
        // This ensures proper state accumulation across chunk boundaries
        let x_combined_chunks = x_gamma_chunks.broadcast_add(&x_beta_chunks)?;
        let states = self.compute_chunk_states(&a_cumsum, &b_chunks, &x_combined_chunks)?;

        // 3. Inter-chunk recurrence
        let states_with_initial = self.add_initial_state(&states)?;
        let decay_chunk = self.compute_decay_chunk(&a_cumsum)?;
        let new_states = self.apply_inter_chunk_recurrence(&decay_chunk, &states_with_initial)?;
        let states = new_states.narrow(1, 0, new_states.dim(1)? - 1)?;

        // 4. State-to-output conversion
        let y_off = self.compute_y_off(&c_chunks, &states, &a_cumsum)?;

        // Combine
        let mut y = y_diag.broadcast_add(&y_off)?;

        // Reshape back
        let total_len = seq_len + pad_size;
        y = y.reshape(&[batch_size, total_len, num_heads, dim])?;

        // Add D residual
        y = y.broadcast_add(&d_residual)?;

        // Crop padding
        if pad_size > 0 {
            y = y.narrow(1, 0, seq_len)?;
        }

        Ok(y)
    }

    /// Shift tensor right by 1 position along sequence dimension
    /// Pads with zeros at the start
    fn shift_right(&self, tensor: &Tensor) -> Result<Tensor> {
        let shape = tensor.dims();
        let seq_dim = 1;
        let seq_len = shape[seq_dim];

        if seq_len <= 1 {
            return Tensor::zeros(shape, tensor.dtype(), tensor.device());
        }

        // Take all but the last position
        let shifted = tensor.narrow(seq_dim, 0, seq_len - 1)?;

        // Pad with zeros at the start
        pad_1d(&shifted, (1, 0), seq_dim, 0.0)
    }

    // Helper: Split projected tensor
    fn split_projection(&self, tensor: &Tensor, sizes: &[usize]) -> Result<Vec<Tensor>> {
        let mut result = Vec::new();
        let mut offset = 0;
        for &size in sizes {
            result.push(tensor.narrow(D::Minus1, offset, size)?);
            offset += size;
        }
        Ok(result)
    }

    // Helper: Repeat interleave
    fn repeat_interleave(&self, tensor: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
        if repeats == 1 {
            return Ok(tensor.clone());
        }
        let shape = tensor.dims();
        let indices: Vec<u32> = (0..shape[dim])
            .flat_map(|i| std::iter::repeat(i as u32).take(repeats))
            .collect();
        let indices_tensor = Tensor::from_vec(indices, shape[dim] * repeats, tensor.device())?;
        tensor.index_select(&indices_tensor, dim)
    }

    // Helper: Compute G matrix
    fn compute_g(&self, c_chunks: &Tensor, b_chunks: &Tensor) -> Result<Tensor> {
        let shape = c_chunks.dims();
        let (b, n_chunks, chunk_size, num_heads, state_size) = (
            shape[0], shape[1], shape[2], shape[3], shape[4]
        );

        let c_flat = c_chunks.reshape(&[b * n_chunks * chunk_size, num_heads, state_size, 1])?;
        let b_flat = b_chunks.reshape(&[b * n_chunks * chunk_size, num_heads, 1, state_size])?;

        let g_intermediate = c_flat.broadcast_mul(&b_flat)?;
        let g = g_intermediate.sum(D::Minus1)?;
        let g = g.sum(D::Minus1)?;

        g.reshape(&[b, n_chunks, chunk_size, 1, num_heads])?
            .broadcast_as(&[b, n_chunks, chunk_size, chunk_size, num_heads])
    }

    // Helper: Compute Y_diag
    fn compute_y_diag(&self, m: &Tensor, hidden_states_chunks: &Tensor) -> Result<Tensor> {
        let shape = m.dims();
        let (b, n_chunks, l, s, num_heads) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let head_dim = hidden_states_chunks.dim(D::Minus1)?;

        let m_reshaped = m.permute((0, 1, 4, 2, 3))?.contiguous_tracked()?
            .reshape(&[b * n_chunks * num_heads, l, s])?;

        let hs_reshaped = hidden_states_chunks.permute((0, 1, 3, 2, 4))?.contiguous_tracked()?
            .reshape(&[b * n_chunks * num_heads, s, head_dim])?;

        let result = m_reshaped.matmul(&hs_reshaped)?;

        result.reshape(&[b, n_chunks, num_heads, l, head_dim])?
            .permute((0, 1, 3, 2, 4))
    }

    // Helper: Compute chunk states
    fn compute_chunk_states(
        &self,
        a_cumsum: &Tensor,
        b_chunks: &Tensor,
        hidden_states_chunks: &Tensor,
    ) -> Result<Tensor> {
        let a_last = a_cumsum.narrow(D::Minus1, self.chunk_size - 1, 1)?;
        let decay_states_temp = a_last.broadcast_sub(a_cumsum)?;
        let decay_states_clamped = decay_states_temp.clamp(MIN_EXP_INPUT, MAX_EXP_INPUT)?;
        let decay_states = decay_states_clamped.exp()?;

        let decay_perm = decay_states.permute((0, 2, 3, 1))?.contiguous_tracked()?;
        let b_decay = b_chunks.broadcast_mul(&decay_perm.unsqueeze(D::Minus1)?)?;

        let shape = b_decay.dims();
        let (b, n_chunks, chunk_size, num_heads, state_size) = (
            shape[0], shape[1], shape[2], shape[3], shape[4]
        );
        let head_dim = hidden_states_chunks.dim(D::Minus1)?;

        let b_reshaped = b_decay.permute((0, 1, 3, 2, 4))?.contiguous_tracked()?
            .reshape(&[b * n_chunks * num_heads, chunk_size, state_size])?;

        let hs_reshaped = hidden_states_chunks.permute((0, 1, 3, 2, 4))?.contiguous_tracked()?
            .reshape(&[b * n_chunks * num_heads, chunk_size, head_dim])?;

        let b_t = b_reshaped.transpose(1, 2)?;
        let states_flat = b_t.matmul(&hs_reshaped)?;

        states_flat.reshape(&[b, n_chunks, num_heads, state_size, head_dim])?
            .transpose(D::Minus2, D::Minus1)
    }

    // Helper: Add initial zero state
    fn add_initial_state(&self, states: &Tensor) -> Result<Tensor> {
        let shape = states.dims();
        let zero_state = Tensor::zeros(&[shape[0], 1, shape[2], shape[3], shape[4]], states.dtype(), states.device())?;
        Tensor::cat(&[&zero_state, states], 1)
    }

    // Helper: Compute decay between chunks
    fn compute_decay_chunk(&self, a_cumsum: &Tensor) -> Result<Tensor> {
        let a_last = a_cumsum.narrow(D::Minus1, self.chunk_size - 1, 1)?;
        let a_last_squeezed = a_last.squeeze(D::Minus1)?;
        let a_last_padded = pad_1d(&a_last_squeezed, (1, 0), 2, 0.0)?;

        let decay_pre_exp = segment_sum(&a_last_padded)?.clamp(MIN_EXP_INPUT, MAX_EXP_INPUT)?;
        let decay_chunk = decay_pre_exp.exp()?;
        decay_chunk.permute((0, 2, 1, 3))?.contiguous_tracked()
    }

    // Helper: Apply inter-chunk recurrence
    fn apply_inter_chunk_recurrence(&self, decay_chunk: &Tensor, states: &Tensor) -> Result<Tensor> {
        let states_shape = states.dims();
        let (b, n_chunks_plus_1, num_heads, head_dim, state_size) = (
            states_shape[0], states_shape[1], states_shape[2], states_shape[3], states_shape[4]
        );

        let decay_for_matmul = decay_chunk.contiguous_tracked()?
            .reshape(&[b * num_heads, n_chunks_plus_1, n_chunks_plus_1])?;

        let states_for_matmul = states.permute((0, 2, 1, 3, 4))?.contiguous_tracked()?
            .reshape(&[b * num_heads, n_chunks_plus_1, head_dim * state_size])?;

        let result = decay_for_matmul.matmul(&states_for_matmul)?;

        result.reshape(&[b, num_heads, n_chunks_plus_1, head_dim, state_size])?
            .permute((0, 2, 1, 3, 4))
    }

    // Helper: Compute Y_off
    fn compute_y_off(
        &self,
        c_chunks: &Tensor,
        states: &Tensor,
        a_cumsum: &Tensor,
    ) -> Result<Tensor> {
        let c_shape = c_chunks.dims();
        let (b, n_chunks, chunk_size, num_heads, state_size) = (
            c_shape[0], c_shape[1], c_shape[2], c_shape[3], c_shape[4]
        );
        let head_dim = states.dim(3)?;

        let a_cumsum_clamped = a_cumsum.clamp(MIN_EXP_INPUT, MAX_EXP_INPUT)?;
        let state_decay = a_cumsum_clamped.exp()?.contiguous_tracked()?;

        let c_reshaped = c_chunks.permute((0, 1, 3, 2, 4))?.contiguous_tracked()?
            .reshape(&[b * n_chunks * num_heads, chunk_size, state_size])?;

        let states_reshaped = states.permute((0, 1, 2, 4, 3))?.contiguous_tracked()?
            .reshape(&[b * n_chunks * num_heads, state_size, head_dim])?;

        let c_times_states = c_reshaped.matmul(&states_reshaped)?;

        let c_times_states = c_times_states.reshape(&[b, n_chunks, num_heads, chunk_size, head_dim])?
            .permute((0, 1, 3, 2, 4))?.contiguous_tracked()?;

        let decay_perm = state_decay.permute((0, 2, 3, 1))?.contiguous_tracked()?;
        let decay_expanded = decay_perm.unsqueeze(D::Minus1)?.contiguous_tracked()?;

        c_times_states.broadcast_mul(&decay_expanded)
    }
}

/// Mamba3 Block (combines norm + mixer + residual)
#[derive(Debug, Clone)]
pub struct Mamba3Block {
    norm: LayerNorm,
    mixer: Mamba3Mixer,
}

impl Mamba3Block {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        state_size: usize,
        chunk_size: usize,
        n_groups: usize,
        expand: usize,
        use_complex_rope: bool,
        mimo_rank: usize,
        use_conv: bool,
        conv_kernel: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm = layer_norm(hidden_size, EPS, vb.pp("norm"))?;
        let mixer = Mamba3Mixer::new(
            hidden_size,
            num_heads,
            head_dim,
            state_size,
            chunk_size,
            n_groups,
            expand,
            use_complex_rope,
            mimo_rank,
            use_conv,
            conv_kernel,
            vb.pp("mixer"),
        )?;

        Ok(Self { norm, mixer })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let normalized = self.norm.forward(x)?;
        let output = self.mixer.forward(&normalized)?;
        residual.broadcast_add(&output)
    }
}
