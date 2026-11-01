//! Pack command implementation
//!
//! Packages a trained model for distribution to HuggingFace Hub.

use anyhow::{Context, Result};
use clap::Args;
use dialoguer::{theme::ColorfulTheme, Select};
use std::fs;
use std::path::{Path, PathBuf};

use boostr::model::UniversalConfig;

/// Pack command arguments
#[derive(Args, Debug)]
pub struct PackArgs {
    /// Checkpoint directory to scan
    #[arg(long, default_value = "./checkpoints")]
    pub checkpoint_dir: String,

    /// Specific checkpoint to pack ("latest", "final", or step number)
    #[arg(short = 'c', long)]
    pub checkpoint: Option<String>,

    /// Model name for the package (default: derived from checkpoint)
    #[arg(short = 'n', long)]
    pub name: Option<String>,

    /// Output directory (default: hf/{username}/{model})
    #[arg(short = 'o', long)]
    pub output: Option<String>,

    /// HuggingFace username (overrides .env HF_USERNAME)
    #[arg(short = 'u', long)]
    pub username: Option<String>,
}

/// Information about a checkpoint file
#[derive(Debug, Clone)]
struct CheckpointInfo {
    path: PathBuf,
    name: String,
    parent_dir: Option<String>,
    step: Option<usize>,
    is_final: bool,
    size_mb: f64,
}

impl CheckpointInfo {
    fn display_name(&self) -> String {
        let prefix = self
            .parent_dir
            .as_ref()
            .map(|p| format!("{}/", p))
            .unwrap_or_default();
        format!("{}{} - {:.1} MB", prefix, self.name, self.size_mb)
    }
}

/// Scan checkpoint directory for _final.safetensors files (recursively searches subdirectories)
fn scan_checkpoints(checkpoint_dir: &Path) -> Result<Vec<CheckpointInfo>> {
    let mut checkpoints = Vec::new();

    if !checkpoint_dir.exists() {
        return Err(anyhow::anyhow!(
            "Checkpoint directory does not exist: {}",
            checkpoint_dir.display()
        ));
    }

    scan_checkpoints_recursive(checkpoint_dir, &mut checkpoints)?;

    // Only keep final checkpoints for packing
    checkpoints.retain(|c| c.is_final);

    // Sort alphabetically by name
    checkpoints.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(checkpoints)
}

/// Recursively scan for .safetensors files
fn scan_checkpoints_recursive(dir: &Path, checkpoints: &mut Vec<CheckpointInfo>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Recurse into subdirectories
            scan_checkpoints_recursive(&path, checkpoints)?;
        } else if path
            .extension()
            .map(|e| e == "safetensors")
            .unwrap_or(false)
        {
            let filename = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let size_bytes = entry.metadata()?.len();
            let size_mb = size_bytes as f64 / (1024.0 * 1024.0);

            // Get parent directory name (e.g., "nano" from "./checkpoints/nano/file.safetensors")
            let parent_dir = path
                .parent()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().to_string());

            let is_final = filename.ends_with("_final");
            let step = if is_final {
                None
            } else if filename.contains("_step_") {
                filename
                    .split("_step_")
                    .last()
                    .and_then(|s| s.parse::<usize>().ok())
            } else {
                None
            };

            checkpoints.push(CheckpointInfo {
                path,
                name: filename,
                parent_dir,
                step,
                is_final,
                size_mb,
            });
        }
    }
    Ok(())
}

/// Select a checkpoint interactively or by CLI argument
fn select_checkpoint<'a>(
    checkpoints: &'a [CheckpointInfo],
    selection: Option<&str>,
) -> Result<&'a CheckpointInfo> {
    if checkpoints.is_empty() {
        return Err(anyhow::anyhow!("No checkpoints found in directory"));
    }

    match selection {
        Some("latest") => {
            // Find the checkpoint with the highest step number (not final)
            checkpoints
                .iter()
                .filter(|c| !c.is_final && c.step.is_some())
                .max_by_key(|c| c.step)
                .or_else(|| checkpoints.first())
                .ok_or_else(|| anyhow::anyhow!("No checkpoints found"))
        }
        Some("final") => {
            // Find the final checkpoint
            checkpoints
                .iter()
                .find(|c| c.is_final)
                .ok_or_else(|| anyhow::anyhow!("No final checkpoint found"))
        }
        Some(step_str) => {
            // Try to parse as step number
            let step: usize = step_str
                .parse()
                .context("Invalid checkpoint selector. Use 'latest', 'final', or a step number")?;
            checkpoints
                .iter()
                .find(|c| c.step == Some(step))
                .ok_or_else(|| anyhow::anyhow!("Checkpoint for step {} not found", step))
        }
        None => {
            // Interactive selection
            let items: Vec<String> = checkpoints.iter().map(|c| c.display_name()).collect();
            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Select a checkpoint to pack")
                .items(&items)
                .default(0)
                .interact()
                .context("Failed to get user selection")?;
            Ok(&checkpoints[selection])
        }
    }
}

/// Load training stats from steps.json or metadata
fn load_training_stats(checkpoint_dir: &Path) -> Option<TrainingStats> {
    // Try to load from the most recent run directory
    let runs_dir = Path::new("./runs");
    if runs_dir.exists() {
        // Find the most recent run directory
        let mut run_dirs: Vec<_> = fs::read_dir(runs_dir)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();
        run_dirs.sort_by_key(|e| e.path());

        if let Some(latest_run) = run_dirs.last() {
            let steps_json = latest_run.path().join("steps.json");
            if steps_json.exists() {
                if let Ok(content) = fs::read_to_string(&steps_json) {
                    // Parse last line for final stats
                    if let Some(last_line) = content.lines().last() {
                        if let Ok(stats) = serde_json::from_str::<serde_json::Value>(last_line) {
                            return Some(TrainingStats {
                                final_loss: stats.get("loss").and_then(|v| v.as_f64()),
                                final_step: stats
                                    .get("step")
                                    .and_then(|v| v.as_u64())
                                    .map(|s| s as usize),
                                learning_rate: stats.get("learning_rate").and_then(|v| v.as_f64()),
                            });
                        }
                    }
                }
            }
        }
    }

    // Try checkpoint directory steps.json
    let steps_json = checkpoint_dir.join("steps.json");
    if steps_json.exists() {
        if let Ok(content) = fs::read_to_string(&steps_json) {
            if let Some(last_line) = content.lines().last() {
                if let Ok(stats) = serde_json::from_str::<serde_json::Value>(last_line) {
                    return Some(TrainingStats {
                        final_loss: stats.get("loss").and_then(|v| v.as_f64()),
                        final_step: stats
                            .get("step")
                            .and_then(|v| v.as_u64())
                            .map(|s| s as usize),
                        learning_rate: stats.get("learning_rate").and_then(|v| v.as_f64()),
                    });
                }
            }
        }
    }

    None
}

#[derive(Debug, Default)]
struct TrainingStats {
    final_loss: Option<f64>,
    final_step: Option<usize>,
    learning_rate: Option<f64>,
}

/// Mamba (SSM) layer count: every layer under a pure-SSM config, the
/// `hybrid_layers.ssm_layers` count under a hybrid one, 0 for dense attention.
fn mamba_layer_count(config: &UniversalConfig) -> usize {
    if let Some(hybrid) = &config.hybrid_layers {
        hybrid.ssm_layers.len()
    } else if config.ssm.is_some() {
        config.num_layers
    } else {
        0
    }
}

/// True when the SSM config's variant is Mamba3, not Mamba2/Mamba1.
fn is_mamba3(config: &UniversalConfig) -> bool {
    config.ssm.as_ref().is_some_and(|s| s.variant == "mamba3")
}

/// True for MLA (Multi-Head Latent Attention): the attention block compresses
/// its KV cache into `kv_latent_dim`.
fn is_mla(config: &UniversalConfig) -> bool {
    config
        .attention
        .as_ref()
        .is_some_and(|a| a.kv_latent_dim.is_some())
}

/// Generate model card README.md
fn generate_model_card(
    model_name: &str,
    username: &str,
    config: &UniversalConfig,
    _checkpoint: &CheckpointInfo,
    stats: Option<&TrainingStats>,
) -> String {
    let arch_desc = describe_architecture(config);
    let param_count = estimate_params_from_config(config);
    let repo_id = format!("{}/{}", username, model_name);

    // Build tags
    let mut tags = vec!["oxidizr", "llm"];
    if config.ssm.is_some() {
        tags.push("mamba");
    }
    if is_mla(config) {
        tags.push("mla");
    }
    if config.moe.is_some() {
        tags.push("moe");
    }
    let tags_str = tags
        .iter()
        .map(|t| format!("- {}", t))
        .collect::<Vec<_>>()
        .join("\n");

    // Training stats
    let training_info = if let Some(stats) = stats {
        let mut info = String::new();
        if let Some(loss) = stats.final_loss {
            info.push_str(&format!("- **Final Loss:** {:.4}\n", loss));
        }
        if let Some(step) = stats.final_step {
            info.push_str(&format!("- **Training Steps:** {}\n", step));
        }
        if let Some(lr) = stats.learning_rate {
            info.push_str(&format!("- **Learning Rate:** {:.2e}\n", lr));
        }
        info
    } else {
        String::new()
    };

    format!(
        r#"---
license: apache-2.0
library_name: oxidizr
tags:
{tags}
pipeline_tag: text-generation
---

# {model_name}

> {arch_desc} model with {params} parameters

Trained with [oxidizr](https://github.com/ml-rust/oxidizr), a Rust-based LLM training framework.

## Overview

{arch_details}

**Key Specifications:**
- **Parameters:** {params}
- **Context Length:** {max_seq} tokens
- **Vocabulary:** {vocab_size} tokens ([splintr](https://github.com/ml-rust/splintr) tokenizer)
{training_info}
## Quick Start

```bash
# Install blazr (recommended inference server)
cargo install blazr

# Generate text
blazr generate --model {repo_id} --prompt "Hello, world!"

# Start API server
blazr serve --model {repo_id} --port 8080
```

## Usage

### Command Line

```bash
# Basic generation
blazr generate --model {repo_id} --prompt "Your prompt here" --max-tokens 100

# With sampling parameters
blazr generate --model {repo_id} \
  --prompt "Once upon a time" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-p 0.9
```

### API Server

```bash
# Start the server
blazr serve --model {repo_id} --port 8080

# The server provides OpenAI-compatible endpoints:
# - POST /v1/completions
# - POST /v1/chat/completions
# - GET  /v1/models
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

# Chat completion
response = client.chat.completions.create(
    model="{repo_id}",
    messages=[{{"role": "user", "content": "Hello!"}}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

### Manual Download

```bash
# Using huggingface-cli
huggingface-cli download {repo_id} --local-dir ./model

# Then run locally
blazr generate --model ./model --prompt "Hello!"
```

## Important Notes

> **This model requires [blazr](https://github.com/ml-rust/blazr) for inference.**

Standard inference tools (llama.cpp, vLLM, Transformers, etc.) do not support this architecture. The model uses:

- **Custom architecture:** Hybrid Mamba2/MLA/MoE layers trained with [oxidizr](https://github.com/ml-rust/oxidizr)
- **Custom tokenizer:** [splintr](https://github.com/ml-rust/splintr) BPE tokenizer with specialized tokens

## Model Card

| Property | Value |
|----------|-------|
| Architecture | {arch_desc} |
| Parameters | {params} |
| Hidden Size | {hidden} |
| Layers | {layers} |
| Vocab Size | {vocab_size} |
| Max Sequence Length | {max_seq} |
| Precision | FP32 |
| License | Apache-2.0 |

## Links

- **Inference:** [blazr](https://github.com/ml-rust/blazr)
- **Training:** [oxidizr](https://github.com/ml-rust/oxidizr)
- **Tokenizer:** [splintr](https://github.com/ml-rust/splintr)
"#,
        tags = tags_str,
        model_name = model_name,
        arch_desc = arch_desc,
        params = param_count,
        arch_details = describe_architecture_details(config),
        max_seq = config.max_seq_len,
        vocab_size = config.vocab_size,
        training_info = training_info,
        repo_id = repo_id,
        hidden = config.hidden_size,
        layers = config.num_layers,
    )
}

/// Generate detailed architecture description for README
fn describe_architecture_details(config: &UniversalConfig) -> String {
    let mut lines = Vec::new();

    let mamba_count = mamba_layer_count(config);
    let attn_count = config.num_layers.saturating_sub(mamba_count);

    // Determine if hybrid or pure architecture
    if mamba_count > 0 && attn_count > 0 {
        lines.push("This model uses a hybrid architecture with:".to_string());
    } else if mamba_count > 0 {
        lines.push("This model uses a pure Mamba architecture with:".to_string());
    } else {
        lines.push("This model architecture includes:".to_string());
    }

    // Mamba layers
    if mamba_count > 0 {
        let mamba_type = if is_mamba3(config) {
            "Mamba3"
        } else {
            "Mamba2"
        };
        lines.push(format!(
            "- **{} {} layers** - State Space Model (SSM) for efficient sequence modeling",
            mamba_count, mamba_type
        ));
    }

    // Attention layers
    if attn_count > 0 {
        if is_mla(config) {
            lines.push(format!(
                "- **{} MLA (Multi-Head Latent Attention) layers** - Compressed KV cache attention",
                attn_count
            ));
        } else {
            lines.push(format!(
                "- **{} Attention layers** - Standard multi-head attention",
                attn_count
            ));
        }
    }

    // MoE
    if let Some(moe) = &config.moe {
        let shared = if moe.has_shared_expert() {
            " + shared expert"
        } else {
            ""
        };
        lines.push(format!(
            "- **MoE (Mixture of Experts)** - {} experts{}, top-{} routing",
            moe.num_experts, shared, moe.experts_per_tok
        ));
    }

    lines.join("\n")
}

/// Describe the model architecture
fn describe_architecture(config: &UniversalConfig) -> String {
    let mut parts = Vec::new();

    // Mamba layers
    let mamba_count = mamba_layer_count(config);
    if mamba_count > 0 {
        if is_mamba3(config) {
            parts.push(format!("{} Mamba3", mamba_count));
        } else {
            parts.push(format!("{} Mamba2", mamba_count));
        }
    }

    // Attention layers (saturating_sub prevents underflow)
    let attn_count = config.num_layers.saturating_sub(mamba_count);
    if attn_count > 0 {
        if is_mla(config) {
            parts.push(format!("{} MLA", attn_count));
        } else {
            parts.push(format!("{} Attention", attn_count));
        }
    }

    // MoE
    if let Some(moe) = &config.moe {
        let num_experts = moe.num_experts;
        let experts_per_tok = moe.experts_per_tok;
        parts.push(format!(
            "MoE ({} experts, top-{})",
            num_experts, experts_per_tok
        ));
    }

    if parts.is_empty() {
        "Transformer".to_string()
    } else {
        parts.join(" + ")
    }
}

/// Estimate parameters from UniversalConfig
fn estimate_params_from_config(config: &UniversalConfig) -> String {
    // Rough estimation based on config
    let embed_params = config.vocab_size * config.hidden_size;
    let intermediate = config.intermediate_size.unwrap_or(config.hidden_size * 4);

    let layer_params = config.num_layers
        * (
            // Attention/Mamba + MLP
            4 * config.hidden_size * config.hidden_size + // Q, K, V, O projections (rough)
        3 * config.hidden_size * intermediate
            // gate, up, down
        );

    let lm_head_params = config.hidden_size * config.vocab_size;
    let total = embed_params + layer_params + lm_head_params;

    if total > 1_000_000_000 {
        format!("{:.2}B", total as f64 / 1_000_000_000.0)
    } else if total > 1_000_000 {
        format!("{:.2}M", total as f64 / 1_000_000.0)
    } else {
        format!("{}K", total / 1000)
    }
}

/// Run the pack command
pub fn run_pack(args: PackArgs) -> Result<()> {
    // Load .env file if it exists
    let _ = dotenvy::dotenv();

    // Get HuggingFace username
    let username = args
        .username
        .or_else(|| std::env::var("HF_USERNAME").ok())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "HuggingFace username not specified. Use --username or set HF_USERNAME in .env"
            )
        })?;

    println!("📦 Oxidizr Model Packer");
    println!("========================\n");

    // Scan checkpoints
    let checkpoint_dir = Path::new(&args.checkpoint_dir);
    println!("Scanning checkpoints in: {}", checkpoint_dir.display());

    let checkpoints = scan_checkpoints(checkpoint_dir)?;
    println!("Found {} checkpoint(s)\n", checkpoints.len());

    // Select checkpoint
    let checkpoint = select_checkpoint(&checkpoints, args.checkpoint.as_deref())?;
    println!("Selected: {}\n", checkpoint.display_name());

    // Load config.json from checkpoint's parent directory (e.g., ./checkpoints/nano-start/)
    let checkpoint_parent = checkpoint
        .path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine checkpoint parent directory"))?;
    let config_path = checkpoint_parent.join("config.json");
    if !config_path.exists() {
        return Err(anyhow::anyhow!(
            "config.json not found in {}. Was this model trained with oxidizr?",
            checkpoint_parent.display()
        ));
    }

    let config_content = fs::read_to_string(&config_path)?;
    let mut config: UniversalConfig =
        serde_json::from_str(&config_content).context("Failed to parse config.json")?;
    let repaired = crate::checkpoint_config::repair_legacy_shared_expert(&mut config);

    // Determine model name
    let model_name = args.name.unwrap_or_else(|| {
        // Extract base name from checkpoint (e.g., "nano_512_117m_f32" from "nano_512_117m_f32_step_1000")
        let filename = checkpoint.name.clone();
        if let Some(pos) = filename.find("_step_") {
            filename[..pos].to_string()
        } else if filename.ends_with("_final") {
            filename.trim_end_matches("_final").to_string()
        } else {
            filename
        }
    });

    // Determine output directory
    let output_dir = args
        .output
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("hf").join(&username).join(&model_name));

    println!("Packing model: {}", model_name);
    println!("Output directory: {}\n", output_dir.display());

    // Create output directory
    fs::create_dir_all(&output_dir)?;

    // Copy model weights
    let model_dest = output_dir.join("model.safetensors");
    println!("  Copying model weights...");
    fs::copy(&checkpoint.path, &model_dest)?;

    // Copy config.json
    let config_dest = output_dir.join("config.json");
    if repaired {
        println!("  Writing config.json with moe.shared_expert: true (older files hold null)...");
        let json =
            serde_json::to_string_pretty(&config).context("Failed to serialize config.json")?;
        fs::write(&config_dest, json)
            .with_context(|| format!("Failed to write {}", config_dest.display()))?;
    } else {
        println!("  Copying config.json...");
        fs::copy(&config_path, &config_dest)?;
    }

    // Try to find and copy training config.yaml
    // Look for run.json or the original config
    let runs_dir = Path::new("./runs");
    if runs_dir.exists() {
        let mut run_dirs: Vec<_> = fs::read_dir(runs_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();
        run_dirs.sort_by_key(|e| e.path());

        if let Some(latest_run) = run_dirs.last() {
            let run_json = latest_run.path().join("run.json");
            if run_json.exists() {
                println!("  Copying run.json as training_config.json...");
                fs::copy(&run_json, output_dir.join("training_config.json"))?;
            }
        }
    }

    // Load training stats
    let stats = load_training_stats(checkpoint_dir);

    // Generate model card README.md
    println!("  Generating README.md (model card)...");
    let model_card =
        generate_model_card(&model_name, &username, &config, checkpoint, stats.as_ref());
    fs::write(output_dir.join("README.md"), model_card)?;

    println!("\n✅ Model packed successfully!");
    println!("\nOutput files:");
    println!("  - {}/model.safetensors", output_dir.display());
    println!("  - {}/config.json", output_dir.display());
    println!("  - {}/README.md", output_dir.display());

    println!("\nNext steps:");
    println!("  1. Review the generated README.md");
    println!(
        "  2. Push to HuggingFace: oxidizr push --model {}",
        output_dir.display()
    );
    println!("     (requires --features huggingface)");

    Ok(())
}
