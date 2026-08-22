use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::Path;

// Default value functions for serde
fn default_max_checkpoints() -> usize {
    10
}
fn default_model_name() -> String {
    "model".to_string()
}
fn default_max_grad_norm() -> f64 {
    1.0
}
fn default_target_grad_norm() -> Option<f64> {
    None
} // None = no normalization
fn default_max_grad_value() -> Option<f64> {
    None
} // None = no per-element clipping

// AdamW optimizer hyperparameters (from optimization.md "Golden Recipe")
fn default_adamw_beta1() -> f64 {
    0.9
} // Standard momentum coefficient
fn default_adamw_beta2() -> f64 {
    0.95
} // Lower β₂ stabilizes MoE sparse updates
fn default_adamw_eps() -> f64 {
    1e-6
} // Higher ε for BF16 numerical safety
fn default_weight_decay() -> f64 {
    0.1
} // Industry standard L2 regularization

// Router z-loss coefficient (prevents MoE logit drift/explosion in BF16)
fn default_z_loss_alpha() -> f64 {
    1e-3
} // SOTA: 1e-3 to 1e-4 range

// Learning rate warmup (critical for Mamba SSM parameter alignment)
fn default_warmup_steps() -> Option<usize> {
    None
} // None = disabled
fn default_warmup_ratio() -> Option<f64> {
    Some(0.1)
} // 10% of total steps if warmup_steps not set

// Re-export distributed config types
pub use crate::distributed::sync::DistributedConfig;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LitConfig {
    pub model: OxidizrModelConfig,
    pub trainer: TrainerConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OxidizrModelConfig {
    /// Model name (e.g., "nano", "small", "base") - used in checkpoint filenames
    #[serde(default = "default_model_name")]
    pub name: String,

    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,

    // Legacy GQA config (for old nano)
    pub kv_heads: Option<usize>,

    // MLA (Multi-Head Latent Attention) config
    pub kv_latent_dim: Option<usize>,
    pub q_latent_dim: Option<usize>,
    pub d_rope: Option<usize>, // Decoupled RoPE dimension

    // MoE (Mixture of Experts) config
    pub num_experts: Option<usize>,
    pub experts_per_tok: Option<usize>,
    #[allow(dead_code)] // Phase 2: Implement sparse MoE with shared expert routing
    pub shared_expert_enabled: Option<bool>,
    pub intermediate_size: Option<usize>,

    // Hybrid architecture config
    pub mamba_layers: Option<Vec<usize>>, // Which layers are Mamba (rest are Attention)

    // Mamba2 (State Space Duality) config
    pub mamba2_num_heads: Option<usize>, // Number of SSM heads (e.g., 24 for nano)
    pub mamba2_head_dim: Option<usize>,  // Dimension per head (e.g., 64)
    pub mamba2_state_size: Option<usize>, // SSM state dimension N (e.g., 64)
    pub mamba2_chunk_size: Option<usize>, // Chunk size for SSD decomposition (e.g., 64)
    pub mamba2_n_groups: Option<usize>,  // Groups for efficient computation (e.g., 1)
    pub mamba2_conv_kernel: Option<usize>, // Conv1D kernel size (e.g., 4)
    pub mamba2_expand: Option<usize>,    // Expansion factor (e.g., 2)

    // Mamba3 config (extends Mamba2 with trapezoidal discretization, complex RoPE, MIMO)
    pub mamba3_enabled: Option<bool>, // Enable Mamba-3 (uses mamba2_* params as base)
    pub mamba3_complex_rope: Option<bool>, // Enable complex-valued RoPE for state tracking
    pub mamba3_mimo_rank: Option<usize>, // MIMO rank (0 = SISO, >0 = MIMO)
    pub mamba3_use_conv: Option<bool>, // Optional conv (default: false, trapezoidal replaces it)

    pub vocab_size: usize,
    /// Grow a smaller checkpoint's `embed_tokens` / `lm_head` tables to
    /// `vocab_size` on load, mean-initializing the added rows. Off by default:
    /// without it, a row-count shortfall is a hard error rather than a silent
    /// reinterpretation of the checkpoint.
    #[serde(default)]
    pub grow_vocab: bool,
    /// Share one weight matrix between `embed_tokens` and `lm_head`. Must match
    /// the checkpoint: a tied checkpoint has no separate `lm_head.weight`.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Tokenizer vocabulary name (cl100k_base, o200k_base, llama3, deepseek_v3, custom)
    /// REQUIRED: Must match the vocabulary used to tokenize your training data
    /// Use "custom" if you have a custom tokenizer not listed above
    pub tokenizer_vocab: String,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    /// RoPE frequency scaling (linear / llama3 / yarn). Only honored on the
    /// dense-attention and hybrid model paths; see `RopeScalingSpec`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_scaling: Option<RopeScalingSpec>,
    /// Sliding-window attention span (Mistral-style). Only honored on the
    /// dense-attention path; boostr's hybrid path never reads it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window: Option<usize>,

    /// Model precision (f32/f16/bf16) - default F32
    /// Mixed precision: SSM (Mamba/Mamba2) and RMSNorm always use F32 internally
    #[serde(default)]
    pub dtype: DTypePrecision,
}

/// RoPE scaling parameters, mirroring `boostr::model::config::RopeScalingConfig`
/// field-for-field so an HF `rope_scaling` block pastes in unchanged.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RopeScalingSpec {
    #[serde(rename = "type")]
    pub scaling_type: String,
    pub factor: f32,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub low_freq_factor: Option<f32>,
    #[serde(default)]
    pub high_freq_factor: Option<f32>,
    #[serde(default)]
    pub attention_factor: Option<f32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
}

/// Target device for training/inference
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TargetDevice {
    #[default]
    Gpu,
    Cpu,
}

/// Model precision for training (F32/F16/BF16)
/// Mixed precision: SSM (Mamba/Mamba2) and RMSNorm always use F32 for numerical stability
#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DTypePrecision {
    #[default]
    F32,
    F16,
    BF16,
}

impl DTypePrecision {
    pub fn to_dtype(self) -> crate::prelude::DType {
        match self {
            DTypePrecision::F32 => crate::prelude::DType::F32,
            DTypePrecision::F16 => crate::prelude::DType::F16,
            DTypePrecision::BF16 => crate::prelude::DType::BF16,
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "f32" | "fp32" | "float32" => Some(Self::F32),
            "f16" | "fp16" | "float16" => Some(Self::F16),
            "bf16" | "bfloat16" => Some(Self::BF16),
            _ => None,
        }
    }
}

impl std::fmt::Display for DTypePrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DTypePrecision::F32 => write!(f, "f32"),
            DTypePrecision::F16 => write!(f, "f16"),
            DTypePrecision::BF16 => write!(f, "bf16"),
        }
    }
}

impl OxidizrModelConfig {
    /// Generate checkpoint filename prefix: {name}_{seq_len}_{params}_{dtype}
    /// Example: nano_512_117m_bf16
    pub fn checkpoint_prefix(&self) -> String {
        let params = self.estimate_params_short();
        format!(
            "{}_{}_{}_{}",
            self.name.to_lowercase(),
            self.max_seq_len,
            params,
            self.dtype.to_string().to_lowercase()
        )
    }

    /// Estimate parameter count in short format (e.g., "117m", "1.5b")
    fn estimate_params_short(&self) -> String {
        let head_dim = self.hidden_size / self.num_heads;
        let kv_heads = self.kv_heads.unwrap_or(self.num_heads);
        let kv_size = kv_heads * head_dim;
        let intermediate_size = self.intermediate_size.unwrap_or(self.hidden_size * 4);

        // Embedding
        let embed = self.vocab_size * self.hidden_size;

        // Attention per layer
        let attn = self.hidden_size * self.hidden_size +  // Q
                   self.hidden_size * kv_size +           // K
                   self.hidden_size * kv_size +           // V
                   self.hidden_size * self.hidden_size; // O

        // MLP per layer
        let mlp = self.hidden_size * intermediate_size * 3;

        // Per layer total
        let layer = attn + mlp + 2 * self.hidden_size;

        // LM head + final norm. A tied head is the embedding matrix, already
        // counted above, so counting it again would overstate the size by a
        // whole vocab-sized matrix — and this figure names the checkpoint file.
        let lm_head = if self.tie_word_embeddings {
            0
        } else {
            self.hidden_size * self.vocab_size
        };
        let head = lm_head + self.hidden_size;

        let total = embed + self.num_layers * layer + head;

        if total >= 1_000_000_000 {
            format!("{:.1}b", total as f64 / 1_000_000_000.0)
        } else {
            format!("{}m", total / 1_000_000)
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainerConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub max_steps: usize,
    pub num_epochs: usize,            // Number of epochs to train
    pub gradient_accumulation: usize, // Number of gradient accumulation steps
    pub checkpoint_dir: String,
    pub log_interval: usize,
    pub save_interval: usize,
    #[serde(default = "default_max_checkpoints")]
    pub max_checkpoints: usize, // Max checkpoints to keep (0 = unlimited)
    pub load_balance_alpha: f64, // For MoE auxiliary loss

    // Gradient control parameters (from "Keep the Gradients Flowing" research)
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f64, // Global gradient clipping threshold (default: 1.0)
    // If ||grad|| > max_grad_norm, scale down to max_grad_norm
    #[serde(default = "default_target_grad_norm")]
    pub target_grad_norm: Option<f64>, // Target gradient normalization (None = disabled)
    // If set, normalize gradients to this target norm
    // Good values: 1.0 - 5.0
    #[serde(default = "default_max_grad_value")]
    pub max_grad_value: Option<f64>, // Per-element gradient clipping (None = disabled)
    // Clips individual gradient values to [-max, +max]
    // Good values: 1.0 - 10.0

    // -------------------------------------------------------------------------
    // AdamW Optimizer Hyperparameters (from optimization.md "Golden Recipe")
    // -------------------------------------------------------------------------
    // These settings are critical for stable training of Hybrid Mamba2-MoE models.
    // The defaults follow SOTA recommendations for BF16/mixed-precision training.
    #[serde(default = "default_adamw_beta1")]
    pub adamw_beta1: f64, // Momentum coefficient (default: 0.9)

    #[serde(default = "default_adamw_beta2")]
    pub adamw_beta2: f64, // Variance coefficient (default: 0.95)
    // Lower β₂ stabilizes sparse MoE expert updates
    #[serde(default = "default_adamw_eps")]
    pub adamw_eps: f64, // Numerical stability (default: 1e-6)
    // Higher ε prevents NaN in BF16 when variance is small
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64, // L2 regularization (default: 0.1)
    // Decoupled from LR, applied to weights only

    // -------------------------------------------------------------------------
    // Router Z-Loss (prevents MoE routing collapse)
    // -------------------------------------------------------------------------
    // Z-loss penalizes large router logits that cause softmax saturation.
    // Essential for BF16 training stability.
    #[serde(default = "default_z_loss_alpha")]
    pub z_loss_alpha: f64, // Router z-loss coefficient (default: 1e-3)
    // L_total = L_MLE + α*L_aux + γ*L_z where γ = z_loss_alpha

    // -------------------------------------------------------------------------
    // Learning Rate Warmup (critical for Mamba SSM alignment)
    // -------------------------------------------------------------------------
    // Linear warmup from 0 to learning_rate over warmup period.
    // Allows SSM parameters (A, B, C, dt) to align before full-speed training.
    // Either warmup_steps OR warmup_ratio should be set (not both).
    #[serde(default = "default_warmup_steps")]
    pub warmup_steps: Option<usize>, // Explicit number of warmup steps (None = use ratio)

    #[serde(default = "default_warmup_ratio")]
    pub warmup_ratio: Option<f64>, // Fraction of total steps for warmup (default: 0.1 = 10%)
    // Only used if warmup_steps is None
    /// Target device: "gpu" (default) or "cpu"
    #[serde(default)]
    pub target_device: TargetDevice,

    /// Distributed training configuration
    #[serde(default)]
    pub distributed: DistributedConfig,

    // GRPO (Group Relative Policy Optimization) config - Phase 2
    #[allow(dead_code)] // Phase 2: GRPO implementation
    pub group_size: usize, // Number of rollouts per prompt (e.g., 4 or 8)
    #[allow(dead_code)] // Phase 2: GRPO implementation
    pub beta_kl: f64, // KL penalty coefficient (e.g., 0.04)
}

impl LitConfig {
    /// Load configuration from a YAML file
    ///
    /// # Arguments
    /// * `path` - Path to YAML configuration file (e.g., "models/nano.yaml")
    ///
    /// # Returns
    /// Parsed configuration or error if file doesn't exist or YAML is invalid
    ///
    /// # Example
    /// ```no_run
    /// # use oxidizr::config::LitConfig;
    /// let config = LitConfig::from_yaml("models/nano.yaml")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// - File not found
    /// - Invalid YAML syntax
    /// - Missing required fields
    /// - Type mismatches in YAML values
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: LitConfig = serde_yaml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration constraints
    fn validate(&self) -> anyhow::Result<()> {
        // Mamba2 constraint: hidden_size * expand == num_heads * head_dim
        if let (Some(num_heads), Some(head_dim), Some(expand)) = (
            self.model.mamba2_num_heads,
            self.model.mamba2_head_dim,
            self.model.mamba2_expand,
        ) {
            let expected = num_heads * head_dim;
            let actual = self.model.hidden_size * expand;
            if actual != expected {
                anyhow::bail!(
                    "Mamba2 config invalid: hidden_size * expand ({}) != num_heads * head_dim ({})",
                    actual,
                    expected
                );
            }
        }

        // Warn if using Mamba-1 (sequential SSM) instead of Mamba-2 (parallel SSD)
        let has_mamba_layers = self
            .model
            .mamba_layers
            .as_ref()
            .is_some_and(|v| !v.is_empty());
        let has_mamba2_config = self.model.mamba2_num_heads.is_some();
        if has_mamba_layers && !has_mamba2_config {
            eprintln!(
                "⚠️  WARNING: Using Mamba-1 (sequential SSM) which is 5-10x slower than Mamba-2."
            );
            eprintln!("   Mamba-1 uses a sequential for-loop over sequence length, bottlenecking GPU utilization.");
            eprintln!(
                "   To enable Mamba-2 (State Space Duality), add these fields to your config:"
            );
            eprintln!("     mamba2_num_heads: 48");
            eprintln!("     mamba2_head_dim: 16");
            eprintln!("     mamba2_state_size: 64");
            eprintln!("     mamba2_chunk_size: 64");
            eprintln!("     mamba2_n_groups: 1");
            eprintln!("     mamba2_conv_kernel: 4");
            eprintln!("     mamba2_expand: 2");
            eprintln!("   See models/nano.yaml for the default Mamba-2 configuration.");
            eprintln!();
        }

        Ok(())
    }

    // Original "Nano" config for local testing (~83M parameters - Llama-style baseline)
    #[allow(dead_code)] // Baseline config for comparison - use models/nano.yaml instead
    pub fn nano_baseline() -> Self {
        Self {
            model: OxidizrModelConfig {
                name: "nano-baseline".to_string(),
                hidden_size: 512,
                num_layers: 8,
                num_heads: 8,
                kv_heads: Some(4), // Grouped Query Attention
                kv_latent_dim: None,
                q_latent_dim: None,
                d_rope: None,
                num_experts: None,
                experts_per_tok: None,
                shared_expert_enabled: None,
                intermediate_size: Some(2048), // 4x hidden
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
                vocab_size: 128354, // Llama 3 + splintr agent tokens
                grow_vocab: false,
                tie_word_embeddings: false,
                tokenizer_vocab: "llama3".to_string(),
                max_seq_len: 512,
                rope_theta: 10000.0,
                rope_scaling: None,
                sliding_window: None,
                dtype: DTypePrecision::F32,
            },
            trainer: TrainerConfig {
                learning_rate: 3e-4,
                batch_size: 2, // Micro-batch size
                max_steps: 5000,
                num_epochs: 2,            // Train for 2 epochs
                gradient_accumulation: 1, // Effective batch = 32
                checkpoint_dir: "./checkpoints".to_string(),
                log_interval: 10,
                save_interval: 500,
                max_checkpoints: 10,     // Keep last 10 checkpoints
                load_balance_alpha: 0.0, // No MoE
                max_grad_norm: 1.0,      // Gradient clipping threshold
                target_grad_norm: None,  // No normalization
                max_grad_value: None,    // No per-element clipping
                // AdamW hyperparameters (SOTA defaults from optimization.md)
                adamw_beta1: 0.9,
                adamw_beta2: 0.95, // Lower β₂ for MoE stability
                adamw_eps: 1e-6,   // Higher ε for BF16 safety
                weight_decay: 0.1, // Industry standard
                z_loss_alpha: 0.0, // No MoE, no z-loss needed
                // Learning rate warmup
                warmup_steps: None,      // Use ratio instead
                warmup_ratio: Some(0.1), // 10% warmup
                target_device: TargetDevice::Gpu,
                distributed: DistributedConfig::default(),
                group_size: 4, // GRPO: 4 rollouts per prompt
                beta_kl: 0.04, // GRPO: KL penalty coefficient
            },
        }
    }

    // Nano POC config (~60M total params, ~25M active via MoE)
    // Hybrid Mamba2 + MLA + Fine-Grained MoE
    #[allow(dead_code)] // Hardcoded config for reference - use models/nano.yaml instead
    pub fn nano() -> Self {
        Self {
            model: OxidizrModelConfig {
                name: "nano".to_string(),
                hidden_size: 384,
                num_layers: 8,
                num_heads: 6,
                kv_heads: None, // Not using GQA, using MLA instead

                // MLA configuration
                kv_latent_dim: Some(192), // Compress KV to half
                q_latent_dim: Some(192),  // Compress Q to half
                d_rope: Some(16),         // Small RoPE dimension for decoupled heads

                // MoE configuration (Top-2 routing to prevent collapse)
                num_experts: Some(4), // 4 experts for nano (reduced from 8 to save memory)
                experts_per_tok: Some(2), // CRITICAL: Top-2 not Top-1
                shared_expert_enabled: Some(true),
                intermediate_size: Some(1536), // 4x hidden per expert

                // Hybrid architecture: 6 Mamba2 + 2 MLA in 3:1 pattern
                // Layers 0,1,2 = Mamba2, Layer 3 = MLA, Layers 4,5,6 = Mamba2, Layer 7 = MLA
                mamba_layers: Some(vec![0, 1, 2, 4, 5, 6]),

                // Mamba2 (State Space Duality) config - enables parallel chunk-based training
                // CONSTRAINT: hidden_size * expand == num_heads * head_dim
                // Validation: 384 * 2 = 768 == 48 * 16 ✓
                mamba2_num_heads: Some(48),
                mamba2_head_dim: Some(16),
                mamba2_state_size: Some(64),
                mamba2_chunk_size: Some(64),
                mamba2_n_groups: Some(1),
                mamba2_conv_kernel: Some(4),
                mamba2_expand: Some(2),
                mamba3_enabled: None,
                mamba3_complex_rope: None,
                mamba3_mimo_rank: None,
                mamba3_use_conv: None,

                vocab_size: 128354, // Llama 3 + splintr agent tokens
                grow_vocab: false,
                tie_word_embeddings: false,
                tokenizer_vocab: "llama3".to_string(),
                max_seq_len: 512, // Can handle longer context with MLA
                rope_theta: 10000.0,
                rope_scaling: None,
                sliding_window: None,
                dtype: DTypePrecision::F32,
            },
            trainer: TrainerConfig {
                learning_rate: 3e-4,
                batch_size: 1,
                max_steps: 5000,
                num_epochs: 2, // Train for 2 epochs
                gradient_accumulation: 1,
                checkpoint_dir: "./checkpoints".to_string(),
                log_interval: 10,
                save_interval: 500,
                max_checkpoints: 10,      // Keep last 10 checkpoints
                load_balance_alpha: 0.01, // For MoE load balancing
                max_grad_norm: 1.0,       // Gradient clipping threshold
                target_grad_norm: None,   // No normalization
                max_grad_value: None,     // No per-element clipping
                // AdamW hyperparameters (SOTA defaults from optimization.md)
                adamw_beta1: 0.9,
                adamw_beta2: 0.95,  // Lower β₂ for MoE stability
                adamw_eps: 1e-6,    // Higher ε for BF16 safety
                weight_decay: 0.1,  // Industry standard
                z_loss_alpha: 1e-3, // Router z-loss for MoE stability
                // Learning rate warmup
                warmup_steps: None,      // Use ratio instead
                warmup_ratio: Some(0.1), // 10% warmup
                target_device: TargetDevice::Gpu,
                distributed: DistributedConfig::default(),
                group_size: 4, // GRPO: 4 rollouts per prompt
                beta_kl: 0.04, // GRPO: KL penalty coefficient
            },
        }
    }
}

/// Checkpoint metadata for validation during resume
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub step: usize,
    pub timestamp: String,
    pub model_config_hash: String,
    pub data_file_path: String,
    pub data_file_hash: String,
    pub checkpoint_file: String,
    /// Model precision used when saving checkpoint (for resume validation)
    #[serde(default = "default_dtype_string")]
    pub dtype: String,
}

fn default_dtype_string() -> String {
    "f32".to_string()
}

impl CheckpointMeta {
    /// Create new checkpoint metadata
    pub fn new(
        step: usize,
        model_config: &OxidizrModelConfig,
        data_file_path: &str,
        checkpoint_file: &str,
    ) -> anyhow::Result<Self> {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let model_config_hash = compute_model_config_hash(model_config);
        let data_file_hash = compute_file_hash(data_file_path)?;

        Ok(Self {
            step,
            timestamp,
            model_config_hash,
            data_file_path: data_file_path.to_string(),
            data_file_hash,
            checkpoint_file: checkpoint_file.to_string(),
            dtype: model_config.dtype.to_string(),
        })
    }

    /// Save metadata to JSON file alongside checkpoint
    pub fn save(&self, checkpoint_dir: &str) -> anyhow::Result<()> {
        let meta_path = format!("{}/checkpoint_step_{}.meta.json", checkpoint_dir, self.step);
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&meta_path, json)?;
        Ok(())
    }

    /// Load metadata from JSON file
    pub fn load(meta_path: &str) -> anyhow::Result<Self> {
        let content = fs::read_to_string(meta_path)?;
        let meta: CheckpointMeta = serde_json::from_str(&content)?;
        Ok(meta)
    }

    /// Validate that this checkpoint matches the current config and data
    pub fn validate(
        &self,
        model_config: &OxidizrModelConfig,
        data_file_path: &str,
    ) -> anyhow::Result<CheckpointValidation> {
        let current_model_hash = compute_model_config_hash(model_config);
        let current_data_hash = compute_file_hash(data_file_path)?;
        let current_dtype = model_config.dtype.to_string();

        let model_matches = self.model_config_hash == current_model_hash;
        let data_matches = self.data_file_hash == current_data_hash;
        let dtype_matches = self.dtype == current_dtype;

        Ok(CheckpointValidation {
            model_matches,
            data_matches,
            dtype_matches,
            checkpoint_model_hash: self.model_config_hash.clone(),
            current_model_hash,
            checkpoint_data_hash: self.data_file_hash.clone(),
            current_data_hash,
            checkpoint_dtype: self.dtype.clone(),
            current_dtype,
        })
    }
}

/// Result of checkpoint validation
#[derive(Debug)]
pub struct CheckpointValidation {
    pub model_matches: bool,
    pub data_matches: bool,
    pub dtype_matches: bool,
    pub checkpoint_model_hash: String,
    pub current_model_hash: String,
    pub checkpoint_data_hash: String,
    pub current_data_hash: String,
    pub checkpoint_dtype: String,
    pub current_dtype: String,
}

impl CheckpointValidation {
    /// Check if checkpoint is valid for resume (model and data must match)
    /// dtype mismatch is allowed (weights will be converted)
    pub fn is_valid(&self) -> bool {
        self.model_matches && self.data_matches
    }
}

/// Compute hash of OxidizrModelConfig for validation
///
/// Only hashes architecture-affecting fields, excluding metadata like:
/// - name (cosmetic)
/// - tokenizer_vocab (inference metadata, doesn't affect weights)
/// - grow_vocab (a load-time policy, not architecture: two otherwise-identical
///   models are the same model whether or not growth was permitted, and
///   `vocab_size` — which does describe the weights — is hashed already)
///
/// `tie_word_embeddings` is deliberately NOT stripped: unlike `grow_vocab` it
/// changes the parameter set (a tied model has no independent
/// `lm_head.weight`), so checkpoints across the two settings are genuinely
/// incompatible and the hash must reflect that.
///
/// Anything added to `OxidizrModelConfig` that is not architecture MUST be stripped
/// here. A new field that reaches the hash changes it for every existing
/// checkpoint, so resume validation reports a model mismatch on weights that
/// are in fact compatible.
pub fn compute_model_config_hash(config: &OxidizrModelConfig) -> String {
    // Create a JSON Value and remove non-architecture fields before hashing
    let mut value = serde_json::to_value(config).unwrap_or(serde_json::Value::Null);
    if let serde_json::Value::Object(ref mut map) = value {
        map.remove("name");
        map.remove("tokenizer_vocab");
        map.remove("grow_vocab");
    }
    let json = serde_json::to_string(&value).unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    json.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Compute hash of a file (first 1MB + file size for speed)
pub fn compute_file_hash(path: &str) -> anyhow::Result<String> {
    let mut file = fs::File::open(path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();

    // Read first 1MB for hashing (fast for large files)
    let mut buffer = vec![0u8; 1024 * 1024.min(file_size as usize)];
    file.read_exact(&mut buffer)?;

    let mut hasher = DefaultHasher::new();
    buffer.hash(&mut hasher);
    file_size.hash(&mut hasher);

    Ok(format!("{:016x}", hasher.finish()))
}

/// Blazr-compatible inference configuration
///
/// This struct matches the config.json format expected by the blazr inference server.
/// It's generated once at training start and saved to the checkpoint directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlazrConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    /// Whether `lm_head` shares the embedding matrix. Must be recorded here:
    /// a tied checkpoint has no `lm_head.weight`, so an inference server that
    /// assumed untied would look for a tensor the file does not contain.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Tokenizer vocabulary name (cl100k_base, o200k_base, llama3, deepseek_v3)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_vocab: Option<String>,

    // Mamba2 parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mamba2_num_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mamba2_head_dim: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mamba2_state_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mamba2_chunk_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mamba2_expand: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mamba2_conv_kernel: Option<usize>,

    // Mamba3 parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mamba3_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mamba3_complex_rope: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mamba3_mimo_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mamba3_use_conv: Option<bool>,

    // MLA (Multi-Head Latent Attention) parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_attention_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_latent_dim: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_latent_dim: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub d_rope: Option<usize>,

    // MoE (Mixture of Experts) parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_experts: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experts_per_tok: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shared_expert_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intermediate_size: Option<usize>,

    // Layer configuration
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mamba_layers: Vec<usize>,

    // RMSNorm
    pub rms_norm_eps: f64,

    // Inference settings
    pub max_seq_len: usize,
}

impl BlazrConfig {
    /// Create a BlazrConfig from an OxidizrModelConfig
    pub fn from_model_config(cfg: &OxidizrModelConfig) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            num_layers: cfg.num_layers,
            vocab_size: cfg.vocab_size,
            tie_word_embeddings: cfg.tie_word_embeddings,
            tokenizer_vocab: Some(cfg.tokenizer_vocab.clone()),

            // Mamba2 parameters
            mamba2_num_heads: cfg.mamba2_num_heads,
            mamba2_head_dim: cfg.mamba2_head_dim,
            mamba2_state_size: cfg.mamba2_state_size,
            mamba2_chunk_size: cfg.mamba2_chunk_size,
            mamba2_expand: cfg.mamba2_expand,
            mamba2_conv_kernel: cfg.mamba2_conv_kernel,

            // Mamba3 parameters
            mamba3_enabled: cfg.mamba3_enabled,
            mamba3_complex_rope: cfg.mamba3_complex_rope,
            mamba3_mimo_rank: cfg.mamba3_mimo_rank,
            mamba3_use_conv: cfg.mamba3_use_conv,

            // MLA parameters
            num_attention_heads: Some(cfg.num_heads),
            kv_latent_dim: cfg.kv_latent_dim,
            q_latent_dim: cfg.q_latent_dim,
            d_rope: cfg.d_rope,

            // MoE parameters
            num_experts: cfg.num_experts,
            experts_per_tok: cfg.experts_per_tok,
            shared_expert_enabled: cfg.shared_expert_enabled,
            intermediate_size: cfg.intermediate_size,

            // Layer configuration
            mamba_layers: cfg.mamba_layers.clone().unwrap_or_default(),

            // RMSNorm (hardcoded default, same as blazr)
            rms_norm_eps: 1e-5,

            // Inference settings
            max_seq_len: cfg.max_seq_len,
        }
    }

    /// Save config to checkpoint directory as config.json
    pub fn save(&self, checkpoint_dir: &str) -> anyhow::Result<()> {
        let config_path = format!("{}/config.json", checkpoint_dir);
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&config_path, json)?;
        Ok(())
    }

    /// Check if config.json already exists in checkpoint directory
    pub fn exists(checkpoint_dir: &str) -> bool {
        Path::new(&format!("{}/config.json", checkpoint_dir)).exists()
    }
}

/// Find the latest checkpoint in a directory
/// Format: {name}_{seq}_{params}_{dtype}_step_{N}.safetensors
pub fn find_latest_checkpoint(
    checkpoint_dir: &str,
) -> anyhow::Result<Option<(String, String, usize)>> {
    let dir = Path::new(checkpoint_dir);
    if !dir.exists() {
        return Ok(None);
    }

    let mut latest_step = 0usize;
    let mut latest_checkpoint: Option<String> = None;
    let mut latest_meta: Option<String> = None;

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
            // Look for *_step_N.safetensors (e.g., nano_512_117m_bf16_step_100.safetensors)
            if filename.contains("_step_") && filename.ends_with(".safetensors") {
                // Extract step number after "_step_"
                if let Some(step_part) = filename.split("_step_").last() {
                    if let Some(step_str) = step_part.strip_suffix(".safetensors") {
                        if let Ok(step) = step_str.parse::<usize>() {
                            if step > latest_step {
                                // Check if corresponding meta file exists
                                let meta_path = format!(
                                    "{}/checkpoint_step_{}.meta.json",
                                    checkpoint_dir, step
                                );
                                if Path::new(&meta_path).exists() {
                                    latest_step = step;
                                    latest_checkpoint = Some(path.to_string_lossy().to_string());
                                    latest_meta = Some(meta_path);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    match (latest_checkpoint, latest_meta) {
        (Some(ckpt), Some(meta)) => Ok(Some((ckpt, meta, latest_step))),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The config hash identifies the *architecture*, so a load-time policy
    /// flag must not enter it. If it did, every checkpoint written before that
    /// field existed would fail `CheckpointValidation::model_matches` and
    /// resume would refuse weights that are perfectly compatible.
    #[test]
    fn grow_vocab_does_not_change_the_config_hash() {
        let mut cfg = LitConfig::nano().model;
        cfg.grow_vocab = false;
        let without = compute_model_config_hash(&cfg);
        cfg.grow_vocab = true;
        let with = compute_model_config_hash(&cfg);
        assert_eq!(
            without, with,
            "grow_vocab is a load-time policy, not architecture — it must be \
             stripped in compute_model_config_hash"
        );
    }

    /// The complement of the above: a field that genuinely describes the
    /// weights must still move the hash, so this test cannot pass by the hash
    /// having become insensitive to everything.
    #[test]
    fn vocab_size_does_change_the_config_hash() {
        let mut cfg = LitConfig::nano().model;
        let before = compute_model_config_hash(&cfg);
        cfg.vocab_size += 1;
        assert_ne!(before, compute_model_config_hash(&cfg));
    }

    /// `rope_scaling`/`sliding_window` are new fields on an existing struct.
    /// Every config written before they existed has them unset, so the JSON
    /// must omit the keys entirely (not serialize as `null`) or every
    /// pre-existing checkpoint fails resume validation with a false "model
    /// mismatch".
    #[test]
    fn new_attention_extras_unset_do_not_change_the_config_hash() {
        let cfg = LitConfig::nano_baseline().model;
        assert!(cfg.rope_scaling.is_none());
        assert!(cfg.sliding_window.is_none());

        let value = serde_json::to_value(&cfg).unwrap();
        let map = value.as_object().unwrap();
        assert!(
            !map.contains_key("rope_scaling"),
            "rope_scaling must be omitted from JSON when unset"
        );
        assert!(
            !map.contains_key("sliding_window"),
            "sliding_window must be omitted from JSON when unset"
        );
    }

    #[test]
    fn sliding_window_set_changes_the_config_hash() {
        let mut cfg = LitConfig::nano_baseline().model;
        let before = compute_model_config_hash(&cfg);
        cfg.sliding_window = Some(4096);
        assert_ne!(before, compute_model_config_hash(&cfg));
    }

    #[test]
    fn rope_scaling_set_changes_the_config_hash() {
        let mut cfg = LitConfig::nano_baseline().model;
        let before = compute_model_config_hash(&cfg);
        cfg.rope_scaling = Some(RopeScalingSpec {
            scaling_type: "yarn".to_string(),
            factor: 2.0,
            original_max_position_embeddings: None,
            low_freq_factor: None,
            high_freq_factor: None,
            attention_factor: None,
            beta_fast: None,
            beta_slow: None,
        });
        assert_ne!(before, compute_model_config_hash(&cfg));
    }
}
