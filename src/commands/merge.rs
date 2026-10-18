//! Merge command implementation
//!
//! Folds trained LoRA adapters into the base weights they wrap and writes a
//! plain checkpoint. A LoRA checkpoint only loads under the config that
//! created its adapters; the merged one loads under that same config with the
//! `lora` block removed, which is what blazr and every other non-LoRA path
//! expect.

use anyhow::{Context, Result};
use clap::Args;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::checkpoint_config::{blazr_export_refusal, write_checkpoint_config};
use crate::checkpoint_sidecars::copy_checkpoint_sidecars;
use crate::config::{LitConfig, LoraConfig};
use crate::model::{load_checkpoint_into_model, save_checkpoint_from_params, LitGPT};
use crate::prelude::{DType, Tensor, VarBuilder, VarMap};
use boostr::runtime::{Runtime, RuntimeClient};
use boostr::tensor::TensorId;

/// Merge command arguments
#[derive(Args, Debug)]
pub struct MergeArgs {
    /// Path to the YAML config the LoRA checkpoint was trained with
    #[arg(short = 'f', long = "config")]
    pub config_file: String,

    /// LoRA checkpoint to merge (.safetensors)
    #[arg(short = 'c', long = "checkpoint")]
    pub checkpoint: String,

    /// Output checkpoint path (.safetensors)
    #[arg(short = 'o', long = "output")]
    pub output: String,

    /// Override target device: "gpu" or "cpu" (default from config)
    #[arg(long = "target-device")]
    pub target_device: Option<String>,
}

/// A config with no `lora` block builds a model with no adapter, so there is
/// nothing to fold into anything.
fn ensure_lora_config(lora: Option<&LoraConfig>, config_file: &str) -> Result<()> {
    if lora.is_some() {
        return Ok(());
    }
    Err(anyhow::anyhow!(
        "config '{config_file}' has no 'lora' block, so a checkpoint trained with it carries \
         no adapter to merge. Pass the config the LoRA run used, or use the checkpoint as it is."
    ))
}

pub fn run_merge(args: MergeArgs) -> Result<()> {
    let config = LitConfig::from_yaml(&args.config_file)
        .with_context(|| format!("Failed to load config '{}'", args.config_file))?;
    ensure_lora_config(config.model.lora.as_ref(), &args.config_file)?;

    #[cfg(feature = "cuda")]
    {
        let use_cpu = args
            .target_device
            .as_deref()
            .map(|dev| dev.eq_ignore_ascii_case("cpu"))
            .unwrap_or(config.trainer.target_device == crate::config::TargetDevice::Cpu);
        if !use_cpu {
            return run_merge_with_runtime::<boostr::CudaRuntime>(args, config);
        }
    }

    run_merge_with_runtime::<boostr::CpuRuntime>(args, config)
}

fn run_merge_with_runtime<R>(args: MergeArgs, config: LitConfig) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client: RuntimeClient<R>
        + boostr::TensorOps<R>
        + boostr::RandomOps<R>
        + boostr::ScalarOps<R>
        + boostr::BinaryOps<R>
        + boostr::CompareOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::ReduceOps<R>
        + boostr::ShapeOps<R>
        + boostr::DequantOps<R>,
{
    println!("🔗 Oxidizr LoRA Merge");
    println!("=====================\n");

    let device = R::default_device();
    let client = R::default_client(&device);
    println!("Using runtime: {}", R::name());

    if !Path::new(&args.checkpoint).exists() {
        return Err(anyhow::anyhow!(
            "LoRA checkpoint '{}' does not exist. Point --checkpoint at the .safetensors file \
             the LoRA run wrote.",
            args.checkpoint
        ));
    }

    println!("Config:     {}", args.config_file);
    println!("Checkpoint: {}", args.checkpoint);
    println!("Output:     {}\n", args.output);

    let mut varmap = VarMap::<R>::new();
    let mut vb = VarBuilder::new(&mut varmap, &device);
    let mut model = LitGPT::<R>::new(&config.model, &mut vb, &client)?;

    // A checkpoint whose `lora_a`/`lora_b` tensors are absent is rejected here:
    // the loader checks the file against the model's FULL parameter set, which
    // for this config includes every adapter factor.
    println!("  Loading LoRA checkpoint...");
    load_checkpoint_into_model(&mut model, &args.checkpoint)?;

    println!("  Merging adapters into base weights...");
    let merged = model.merge_lora_adapters(&client)?;
    if merged == 0 {
        return Err(anyhow::anyhow!(
            "config '{}' has a 'lora' block but the model it builds carries no adapted \
             projection, so nothing was merged. Point --config at the config the LoRA run used.",
            args.config_file
        ));
    }
    println!("  Merged {} projection(s)", merged);

    // The merged model's own `Var`s hold every value to write: the optimizer
    // is not in play here, so each parameter is its own source.
    let params: HashMap<TensorId, Tensor<R>> = model
        .named_parameters()?
        .into_iter()
        .map(|(_, id, var)| (id, var.tensor().clone()))
        .collect();

    let output = PathBuf::from(&args.output);
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory '{}'", parent.display()))?;
    }
    println!("  Writing merged checkpoint...");
    save_checkpoint_from_params(&args.output, &model, &params)?;

    // The blazr sidecar carries no `lora` key in any case — a merged model has
    // no adapter left to describe — but the config it is derived from is
    // stripped too, so what is written describes the merged model exactly.
    let mut merged_config = config.model.clone();
    merged_config.lora = None;
    let sidecar_dir = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let sidecar_dir = sidecar_dir.to_string_lossy().into_owned();
    let refusal = blazr_export_refusal(&merged_config);
    match &refusal {
        Some(reason) => println!("  Skipping config.json: {reason}"),
        None => {
            println!("  Writing config.json for blazr inference compatibility...");
            write_checkpoint_config(&merged_config, &sidecar_dir)
                .with_context(|| format!("Failed to write config.json in '{}'", sidecar_dir))?;
        }
    }
    copy_checkpoint_sidecars(&config.trainer.checkpoint_sidecars, &sidecar_dir).with_context(
        || {
            format!(
                "Failed to copy checkpoint sidecar files into '{}'",
                sidecar_dir
            )
        },
    )?;

    println!("\n✅ Merge complete!");
    println!("\nOutput files:");
    println!("  - {}", args.output);
    if refusal.is_none() {
        println!("  - {}/config.json", sidecar_dir);
    }
    println!("\nNext steps:");
    println!("  Load it with the same config minus its 'lora' block, or serve it with blazr.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lora_cfg() -> LoraConfig {
        LoraConfig {
            rank: 2,
            alpha: 4.0,
            target_modules: vec!["q_proj".to_string()],
            freeze_base: false,
            train_modules: Vec::new(),
        }
    }

    #[test]
    fn a_config_without_a_lora_block_is_rejected() {
        let message = ensure_lora_config(None, "models/plain.yaml")
            .expect_err("a config with no adapter must not be mergeable")
            .to_string();
        assert!(message.contains("models/plain.yaml"), "{message}");
        assert!(message.contains("lora"), "{message}");
    }

    #[test]
    fn a_config_with_a_lora_block_is_accepted() {
        ensure_lora_config(Some(&lora_cfg()), "models/lora.yaml")
            .expect("a config with an adapter must be mergeable");
    }
}
