//! The console text of a training run: the parameter estimate, the banner, the OOM report.

use crate::commands::train_plan::RunPlan;
use crate::config::{LitConfig, OxidizrModelConfig};
use crate::data::RowLayout;

/// Estimate the number of parameters in the model
pub fn estimate_params(cfg: &OxidizrModelConfig) -> String {
    // Embedding: vocab_size * hidden_size
    let embed_params = cfg.vocab_size * cfg.hidden_size;

    // Each layer: Q, K, V, O projections (K and V at the GQA width), the
    // SwiGLU gate, up and down projections, and 2 RMSNorms.
    let head_dim = cfg.hidden_size / cfg.num_heads;
    let kv_heads = cfg.kv_heads.unwrap_or(cfg.num_heads);
    let kv_size = kv_heads * head_dim;
    let intermediate_size = cfg.intermediate_size.unwrap_or(cfg.hidden_size * 4);

    let attention_params = cfg.hidden_size * cfg.hidden_size + // Q proj
                          cfg.hidden_size * kv_size +          // K proj
                          cfg.hidden_size * kv_size +          // V proj
                          cfg.hidden_size * cfg.hidden_size; // O proj

    let mlp_params = cfg.hidden_size * intermediate_size +  // gate_proj
                    cfg.hidden_size * intermediate_size +   // up_proj
                    intermediate_size * cfg.hidden_size; // down_proj

    let layernorm_params = 2 * cfg.hidden_size; // 2 RMSNorms per layer

    let layer_params = cfg.num_layers * (attention_params + mlp_params + layernorm_params);

    // LM Head: hidden_size * vocab_size, but zero when tied — a tied model
    // reuses the embedding matrix rather than holding a second one, so
    // counting it again overstates the model by a whole vocab-sized matrix.
    let lm_head_params = if cfg.tie_word_embeddings {
        0
    } else {
        cfg.hidden_size * cfg.vocab_size
    };

    // Final norm
    let final_norm_params = cfg.hidden_size;

    let total = embed_params + layer_params + lm_head_params + final_norm_params;

    if total > 1_000_000_000 {
        format!("{:.2}B", total as f64 / 1_000_000_000.0)
    } else if total > 1_000_000 {
        format!("{:.2}M", total as f64 / 1_000_000.0)
    } else {
        format!("{}K", total / 1000)
    }
}

/// GPU memory of the first GPU, from `nvidia-smi`. `None` when it does not answer.
pub fn cuda_memory_info() -> Option<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let stats = String::from_utf8_lossy(&output.stdout);
    let line = stats.lines().next()?;
    let parts: Vec<f32> = line
        .trim()
        .split(',')
        .map(|p| p.trim().parse().unwrap_or(0.0))
        .collect();
    let [total_mb, used_mb, free_mb] = parts.as_slice() else {
        return None;
    };
    Some(format!(
        "Total: {:.2} GB, Used: {:.2} GB, Free: {:.2} GB",
        total_mb / 1024.0,
        used_mb / 1024.0,
        free_mb / 1024.0
    ))
}

/// What the banner reports about one run.
pub struct BannerFacts<'a> {
    pub config: &'a LitConfig,
    pub dataset_size: usize,
    pub trainable_params: usize,
    pub row_layout: &'a RowLayout,
    pub steps_per_epoch: f64,
    pub total_steps: usize,
    /// Rows per optimizer step over every rank.
    pub effective_batch_size: usize,
}

/// Prints the configuration and training plan before the first step.
pub fn print_banner(facts: &BannerFacts<'_>) {
    let config = facts.config;
    let trainer = &config.trainer;
    let layout = facts.row_layout;
    println!("\n============================================================");
    println!("🦥 OXIDIZR TRAINING");
    println!("============================================================\n");

    println!("📋 Configuration:");
    let num_mamba = config.model.mamba_layers.as_ref().map_or(0, Vec::len);
    let num_attention = config.model.num_layers - num_mamba;
    let mamba_type = if config.model.mamba2_num_heads.is_some() {
        "Mamba2"
    } else {
        "Mamba"
    };
    println!(
        "  Model: {} parameters ({} {} + {} MLA+MoE layers)",
        estimate_params(&config.model),
        num_mamba,
        mamba_type,
        num_attention
    );
    println!("  Dataset: {} tokens", facts.dataset_size);
    println!("  Checkpoint Dir: {}", trainer.checkpoint_dir);
    println!();
    println!("  Max Seq Length: {}", config.model.max_seq_len);
    if let Some(num_experts) = config.model.num_experts {
        println!(
            "  MoE: {} experts, Top-{} routing, Load balance α={:.3}",
            num_experts,
            config.model.experts_per_tok.unwrap_or(1),
            trainer.load_balance_alpha
        );
    }
    println!(
        "  Batch Size: {}, Gradient Accumulation: {}",
        trainer.batch_size, trainer.gradient_accumulation
    );
    println!("  Learning Rate: {:.6}", trainer.learning_rate);
    println!("  Epochs: {}", trainer.num_epochs);
    println!();

    println!("📊 Training Plan:");
    println!(
        "  Trainable params: {} || all params: {} || trainable%: 100.00",
        facts.trainable_params, facts.trainable_params
    );
    println!(
        "  Rows = {} | Steps per epoch = {:.2} | Num Epochs = {} | Total steps = {}",
        layout.num_rows(),
        facts.steps_per_epoch,
        trainer.num_epochs,
        facts.total_steps
    );
    println!(
        "  Batch size per device = {} | Devices = {} | Gradient accumulation = {}",
        layout.rank_batch_size(),
        layout.world_size(),
        trainer.gradient_accumulation
    );
    println!(
        "  Total batch size ({} per device x {} devices x {} accumulation) = {}",
        layout.rank_batch_size(),
        layout.world_size(),
        trainer.gradient_accumulation,
        facts.effective_batch_size
    );
    println!();

    println!("============================================================");
    println!("🚀 STARTING TRAINING");
    println!("============================================================");
    println!();
    println!("Training in progress... (progress bar should appear below)");
    println!("If no progress bar appears, use --headless for text output.\n");
}

/// Prints the device memory, the run's sizes and what to reduce after an OOM.
fn print_oom_report(config: &LitConfig) {
    let (batch_size, seq_len) = (config.trainer.batch_size, config.model.max_seq_len);
    println!("\n❌ CUDA OUT OF MEMORY ERROR ❌");
    if let Some(mem_info) = cuda_memory_info() {
        println!("GPU Memory at failure: {}", mem_info);
    }
    println!("\nCurrent configuration:");
    println!("  - Model: {} parameters", estimate_params(&config.model));
    println!("  - Batch size: {}", batch_size);
    println!("  - Sequence length: {}", seq_len);
    println!("  - Hidden size: {}", config.model.hidden_size);
    println!("  - Number of layers: {}", config.model.num_layers);
    println!(
        "  - Number of experts: {}",
        config.model.num_experts.unwrap_or(0)
    );
    println!("\nSuggestions:");
    println!("  1. Reduce batch_size (currently {})", batch_size);
    println!("  2. Reduce seq_len (currently {})", seq_len);
    println!("  3. Reduce model size (hidden_size, num_layers, or num_experts)");
    println!("  4. Use --dtype bf16 or --dtype f16 for lower memory usage");
    println!();
}

/// The result of a rank's fit loop, with an out-of-memory error marked.
///
/// When `say` is true, a completed run prints where its checkpoints are, and
/// an OOM prints the run's sizes and what to reduce.
pub fn finish_rank(
    result: crate::prelude::Result<()>,
    config: &LitConfig,
    say: bool,
) -> anyhow::Result<()> {
    match result {
        Ok(()) => {
            if say {
                println!(
                    "\nTraining complete! Checkpoints saved to: {}",
                    config.trainer.checkpoint_dir
                );
            }
            Ok(())
        }
        Err(e) => {
            let err_str = format!("{:?}", e);
            if err_str.contains("out of memory") || err_str.contains("OUT_OF_MEMORY") {
                if say {
                    print_oom_report(config);
                }
                // Keep the device error under the summary, so run.json records both.
                Err(anyhow::Error::from(e).context("CUDA out of memory"))
            } else {
                Err(e.into())
            }
        }
    }
}

/// The `config` object of `run.json`: the model, and the trainer over every rank.
pub fn run_config_json(
    plan: &RunPlan,
    layout: &RowLayout,
    total_steps: usize,
    effective_batch_size: usize,
) -> serde_json::Value {
    let config = &plan.config;
    serde_json::json!({
        "model": {
            "name": config.model.name,
            "hidden_size": config.model.hidden_size,
            "num_layers": config.model.num_layers,
            "num_heads": config.model.num_heads,
            "num_experts": config.model.num_experts,
            "experts_per_tok": config.model.experts_per_tok,
            "vocab_size": config.model.vocab_size,
            "max_seq_len": config.model.max_seq_len,
            "dtype": config.model.dtype.to_string(),
        },
        "trainer": {
            "learning_rate": config.trainer.learning_rate,
            "batch_size": layout.rank_batch_size(),
            "world_size": layout.world_size(),
            "gpu_ids": plan.gpu_ids,
            "seq_len": config.model.max_seq_len,
            "gradient_accumulation": config.trainer.gradient_accumulation,
            "max_steps": config.trainer.max_steps,
            "num_epochs": config.trainer.num_epochs,
            "effective_batch_size": effective_batch_size,
            "total_steps": total_steps,
        }
    })
}
