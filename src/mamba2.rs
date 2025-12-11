// Mamba2 (State Space Duality) Implementation using boostr
// Based on "Transformers are SSMs" (Dao & Gu, 2024)
//
// This implementation uses boostr's fused SSD kernels for GPU acceleration.
// The SSD algorithm is implemented in boostr/src/mamba2/ssd/ with full autograd support.

use crate::prelude::{
    Conv1d, Conv1dConfig, Linear, Result, RmsNorm, Tensor, VarBuilder,
    conv1d, linear, ops, rms_norm,
};
use boostr::mamba2::{SSDConfig, ssd_forward};

// Numerical stability constants
const EPS: f64 = 1e-6;

/// Mamba2 Mixer - Core SSD (State Space Duality) implementation
/// Uses boostr's fused GPU kernels for the SSD algorithm
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Mamba2Mixer {
    // Configuration
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
    ssm_state_size: usize,
    chunk_size: usize,
    n_groups: usize,
    conv_kernel_size: usize,
    conv_dim: usize,
    time_step_min: f32,
    time_step_max: f32,

    // Learnable parameters
    in_proj: Linear,
    conv1d: Conv1d,
    dt_bias: Tensor,
    a_log: Tensor,
    d: Tensor,
    norm: RmsNorm,  // Use boostr's RmsNorm with autograd support
    out_proj: Linear,

    // SSD config for boostr kernel
    ssd_config: SSDConfig,
}

impl Mamba2Mixer {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        state_size: usize,
        chunk_size: usize,
        n_groups: usize,
        conv_kernel: usize,
        expand: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let intermediate_size = hidden_size * expand;
        let conv_dim = intermediate_size + 2 * n_groups * state_size;
        let projection_size = intermediate_size + conv_dim + num_heads;

        // Input projection
        let in_proj = linear(hidden_size, projection_size, vb.pp("in_proj"))?;

        // Depthwise Conv1D (groups = conv_dim for depthwise)
        let conv_cfg = Conv1dConfig {
            groups: conv_dim,
            padding: conv_kernel - 1,
            ..Default::default()
        };
        let conv1d = conv1d(conv_dim, conv_dim, conv_kernel, conv_cfg, vb.pp("conv1d"))?;

        // Time step bias (dt_bias)
        let dt_bias = vb.get(num_heads, "dt_bias")?;

        // A_log initialization (Official Mamba2: zeros, model learns decay rates)
        // Use get_with_init to force zeros regardless of VarBuilder's default init
        let a_log = vb.get_with_init(num_heads, "A_log", boostr::nn::var_builder::Init::Zeros)?;

        // D skip connection (residual multiplier per head)
        // Official Mamba2: initialized to ones (torch.ones(nheads))
        let d = vb.get_with_init(num_heads, "D", boostr::nn::var_builder::Init::Ones)?;

        // RMS norm (using boostr's implementation with autograd support)
        let norm = rms_norm(intermediate_size, EPS, vb.pp("norm"))?;

        // Output projection
        let out_proj = linear(intermediate_size, hidden_size, vb.pp("out_proj"))?;

        // DEBUG: Log weight initialization scales
        if std::env::var("DEBUG_INIT").is_ok() {
            if let Ok(w) = in_proj.weight().to_vec::<f32>() {
                let rms = (w.iter().map(|v| v * v).sum::<f32>() / w.len() as f32).sqrt();
                let max_abs = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                eprintln!("[Mamba2 INIT] in_proj: rms={:.6e}, max={:.6e}", rms, max_abs);
            }
            if let Ok(w) = out_proj.weight().to_vec::<f32>() {
                let rms = (w.iter().map(|v| v * v).sum::<f32>() / w.len() as f32).sqrt();
                let max_abs = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                eprintln!("[Mamba2 INIT] out_proj: rms={:.6e}, max={:.6e}", rms, max_abs);
            }
        }

        // SSD config for boostr kernel
        // CRITICAL: Pass n_groups so backward can reduce B/C gradients correctly
        let ssd_config = SSDConfig::with_groups(num_heads, head_dim, state_size, chunk_size, n_groups);

        Ok(Self {
            num_heads,
            head_dim,
            hidden_size,
            intermediate_size,
            ssm_state_size: state_size,
            chunk_size,
            n_groups,
            conv_kernel_size: conv_kernel,
            conv_dim,
            time_step_min: 0.0,  // Official Mamba2 default: dt_limit=(0.0, float("inf"))
            time_step_max: f32::INFINITY,  // No upper bound (Official uses float("inf"))
            in_proj,
            conv1d,
            dt_bias,
            a_log,
            d,
            norm,
            out_proj,
            ssd_config,
        })
    }

    /// Forward pass using boostr's fused SSD kernels
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (_batch_size, seq_len, _) = hidden_states.dims3()?;
        let dtype = hidden_states.dtype();

        // 1. Input projection
        let projected = self.in_proj.forward(hidden_states)?;

        // Split projections - use narrow_tracked for gradient flow
        let d_mlp = (projected.dim(-1)? - 2 * self.intermediate_size
                    - 2 * self.n_groups * self.ssm_state_size - self.num_heads) / 2;

        let splits = vec![d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads];
        let parts = self.split_projection_tracked(&projected, &splits)?;
        let (_skip1, _skip2, gate, hidden_states_b_c, dt) = (
            &parts[0], &parts[1], &parts[2], &parts[3], &parts[4]
        );

        // 2. Conv1D (depthwise)
        // Transpose for conv: [B, L, C] -> [B, C, L] - use transpose_tracked
        let hidden_states_b_c_t = hidden_states_b_c.transpose_tracked(1, 2)?;
        let conv_out = self.conv1d.forward(&hidden_states_b_c_t)?;

        // Crop to seq_len and transpose back - use tracked versions
        let conv_out = conv_out.narrow_tracked(2, 0, seq_len)?;
        let hidden_states_b_c = conv_out.transpose_tracked(1, 2)?;

        // Apply SiLU activation - use to_dtype_tracked for gradient flow
        let hidden_states_b_c = ops::silu(&hidden_states_b_c)?.to_dtype_tracked(dtype)?;

        // Split into hidden_states, B, C - use narrow_tracked
        let hs = hidden_states_b_c.narrow_tracked(-1, 0, self.intermediate_size)?;
        let b = hidden_states_b_c.narrow_tracked(
            -1,
            self.intermediate_size,
            self.n_groups * self.ssm_state_size
        )?;
        let c = hidden_states_b_c.narrow_tracked(
            -1,
            self.intermediate_size + self.n_groups * self.ssm_state_size,
            self.n_groups * self.ssm_state_size
        )?;

        // 3. Reshape for SSD - use reshape_tracked
        let (batch_size, seq_len, _) = hs.dims3()?;
        let hs_reshaped = hs.reshape_tracked(vec![batch_size, seq_len, self.num_heads, self.head_dim])?;

        // Reshape and repeat B, C for all heads - use reshape_tracked
        let mut b_reshaped = b.reshape_tracked(vec![batch_size, seq_len, self.n_groups, self.ssm_state_size])?;
        let mut c_reshaped = c.reshape_tracked(vec![batch_size, seq_len, self.n_groups, self.ssm_state_size])?;

        // DEBUG: Check B/C norms BEFORE repeat_interleave
        let debug_enabled = std::env::var("DEBUG_MAMBA2_FWD").is_ok();
        if debug_enabled {
            let b_norm_before = b_reshaped.to_vec::<f32>().ok()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt());
            let c_norm_before = c_reshaped.to_vec::<f32>().ok()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt());
            eprintln!("[SSD_IN] BEFORE repeat_interleave: b_shape={:?}, c_shape={:?}",
                b_reshaped.dims(), c_reshaped.dims());
            eprintln!("[SSD_IN] BEFORE repeat_interleave: b_norm={:?}, c_norm={:?}",
                b_norm_before, c_norm_before);
        }

        let repeat_factor = self.num_heads / self.n_groups;
        if repeat_factor > 1 {
            b_reshaped = self.repeat_interleave(&b_reshaped, repeat_factor, 2)?;
            c_reshaped = self.repeat_interleave(&c_reshaped, repeat_factor, 2)?;
        }

        // DEBUG: Check B/C norms AFTER repeat_interleave
        if debug_enabled {
            let b_norm_after = b_reshaped.to_vec::<f32>().ok()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt());
            let c_norm_after = c_reshaped.to_vec::<f32>().ok()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt());
            eprintln!("[SSD_IN] AFTER repeat_interleave: b_shape={:?}, c_shape={:?}",
                b_reshaped.dims(), c_reshaped.dims());
            eprintln!("[SSD_IN] AFTER repeat_interleave: b_norm={:?}, c_norm={:?}",
                b_norm_after, c_norm_after);
            eprintln!("[SSD_IN] Expected norm ratio: sqrt({}) = {:.2}",
                repeat_factor, (repeat_factor as f32).sqrt());
        }

        // DEBUG: Check SSD inputs
        if debug_enabled {
            let hs_norm = hs_reshaped.to_vec::<f32>().ok()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt());
            let b_norm = b_reshaped.to_vec::<f32>().ok()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt());
            let c_norm = c_reshaped.to_vec::<f32>().ok()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt());
            let dt_norm = dt.to_vec::<f32>().ok()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt());
            let a_log_vals = self.a_log.to_vec::<f32>().ok();
            let d_vals = self.d.to_vec::<f32>().ok();
            eprintln!("[SSD_IN] x_norm={:?}, b_norm={:?}, c_norm={:?}, dt_norm={:?}",
                hs_norm, b_norm, c_norm, dt_norm);
            eprintln!("[SSD_IN] a_log={:?}, D={:?}", a_log_vals, d_vals);
        }

        // 4. Call boostr's fused SSD kernel
        let y = ssd_forward(
            &hs_reshaped,
            &b_reshaped,
            &c_reshaped,
            dt,
            &self.dt_bias,
            &self.a_log,
            &self.d,
            &self.ssd_config,
            self.time_step_min,
            self.time_step_max,
        )?;

        // Reshape output: [B, seq_len, num_heads, head_dim] -> [B, seq_len, intermediate_size]
        // Use reshape_tracked for gradient flow
        let y = y.reshape_tracked(vec![batch_size, seq_len, self.num_heads * self.head_dim])?;

        // 5. Gating: y * silu(gate)
        let gate_silu = ops::silu(gate)?;
        let y_gated = y.mul(&gate_silu)?;

        // DEBUG: Check forward values
        if std::env::var("DEBUG_MAMBA2_FWD").is_ok() {
            let y_norm = y.to_vec::<f32>().ok()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt());
            let gate_silu_norm = gate_silu.to_vec::<f32>().ok()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt());
            let y_gated_data = y_gated.to_vec::<f32>().ok();
            let (y_gated_norm, mean_rms) = if let Some(ref v) = y_gated_data {
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                // Compute mean RMS (outer_size = batch * seq, hidden = intermediate)
                let outer = batch_size * seq_len;
                let hidden = self.intermediate_size;
                let mut rms_sum = 0.0f32;
                for row in 0..outer {
                    let start = row * hidden;
                    let sum_sq: f32 = v[start..start+hidden].iter().map(|x| x * x).sum();
                    let rms = (sum_sq / hidden as f32 + 1e-5).sqrt();
                    rms_sum += rms;
                }
                (norm, rms_sum / outer as f32)
            } else {
                (0.0, 0.0)
            };
            eprintln!("[MAMBA2_FWD] y_norm={:?}, gate_silu_norm={:?}, y_gated_norm={:.4}, mean_rms={:.6}",
                y_norm, gate_silu_norm, y_gated_norm, mean_rms);
        }

        // 6. RMS normalization (boostr's implementation with autograd)
        let scan_output = self.norm.forward(&y_gated)?;

        // 7. Output projection
        let output = self.out_proj.forward(&scan_output)?;
        output.to_dtype_tracked(dtype)
    }

    // Helper: Split projected tensor with gradient tracking
    fn split_projection_tracked(&self, tensor: &Tensor, sizes: &[usize]) -> Result<Vec<Tensor>> {
        let mut result = Vec::new();
        let mut offset = 0;
        for &size in sizes {
            result.push(tensor.narrow_tracked(-1, offset, size)?);
            offset += size;
        }
        Ok(result)
    }

    // Helper: Repeat interleave (like PyTorch's repeat_interleave)
    // Uses index_select_tracked for proper gradient flow - backward sums gradients
    // from repeated elements back to the original positions.
    fn repeat_interleave(&self, tensor: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
        let shape = tensor.dims();

        // Create indices for gathering
        let indices: Vec<u32> = (0..shape[dim])
            .flat_map(|i| std::iter::repeat(i as u32).take(repeats))
            .collect();
        let indices_tensor = Tensor::from_vec(indices, shape[dim] * repeats, tensor.device())?;

        // Use tracked version for autograd support
        tensor.index_select_tracked(&indices_tensor, dim as i32)
    }
}

/// Mamba2 Block (combines norm + mixer + residual)
#[derive(Debug, Clone)]
pub struct Mamba2Block {
    norm: RmsNorm,
    mixer: Mamba2Mixer,
}

impl Mamba2Block {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        state_size: usize,
        chunk_size: usize,
        n_groups: usize,
        conv_kernel: usize,
        expand: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm = rms_norm(hidden_size, EPS, vb.pp("norm"))?;
        let mixer = Mamba2Mixer::new(
            hidden_size,
            num_heads,
            head_dim,
            state_size,
            chunk_size,
            n_groups,
            conv_kernel,
            expand,
            vb.pp("mixer"),
        )?;

        Ok(Self { norm, mixer })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let normalized = self.norm.forward(x)?;
        let output = self.mixer.forward(&normalized)?;
        residual.add(&output)
    }
}
