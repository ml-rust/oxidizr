//! Import command implementation
//!
//! Turns a HuggingFace checkpoint directory into an oxidizr checkpoint, so a
//! run starts from pretrained weights instead of random init.
//!
//! The only naming difference between the two conventions is HuggingFace's
//! `model.` prefix on everything below the LM head, so the mapping is a prefix
//! strip. It is applied against the model's OWN parameter-name set rather than
//! blindly: an HF tensor whose stripped name matches nothing is a hard error,
//! not a silent drop, and the loader's missing-tensor check then catches the
//! reverse case (an oxidizr parameter no HF tensor filled). Between them a
//! "successful" import cannot leave a randomly initialized weight behind.
//!
//! LoRA adapters are the one exemption, and it is narrow: a config with a
//! `lora:` block adds `lora_a`/`lora_b` to each targeted projection, and a
//! pretrained base checkpoint has neither, because adapters are created at
//! fine-tune time. Those names — and only those — are initialized here rather
//! than demanded of the source, with `lora_b` zeroed so the imported model is
//! numerically the base model at step 0. Every other absent parameter still
//! fails the import, and the written checkpoint carries the adapters, so the
//! later `--resume` of it meets the unrelaxed check with nothing missing.

use anyhow::{Context, Result};
use clap::Args;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::config::{CheckpointMeta, LitConfig};
use crate::model::{load_tensors_initializing_lora, save_checkpoint_from_params, LitGPT};
use crate::prelude::{DType, Tensor, VarBuilder, VarMap};
use boostr::runtime::{Runtime, RuntimeClient};
use boostr::tensor::TensorId;
use boostr::TypeConversionOps;

/// The step an imported checkpoint is stamped with: nothing has been trained
/// from it yet.
const IMPORT_STEP: usize = 0;

/// HuggingFace's prefix on every decoder parameter. `lm_head.weight` is the
/// one tensor outside it.
const HF_DECODER_PREFIX: &str = "model.";

/// How many freshly initialized adapter names are printed in full. A rank-8
/// adapter on seven projections of a 32-layer model is 448 names; the count
/// and a sample are what an operator reads.
const FRESH_ADAPTERS_LISTED: usize = 8;

/// Import command arguments
#[derive(Args, Debug)]
pub struct ImportArgs {
    /// Path to the oxidizr YAML config describing the imported architecture
    #[arg(short = 'f', long = "config")]
    pub config_file: String,

    /// HuggingFace model directory (config.json + *.safetensors)
    #[arg(short = 'm', long = "model-dir")]
    pub model_dir: String,

    /// Tokenized training data the imported checkpoint will be trained on.
    /// Recorded in the checkpoint metadata so `--resume auto` validates.
    #[arg(short = 'd', long = "data")]
    pub data_file: String,

    /// Output checkpoint path (.safetensors). Defaults to
    /// `<checkpoint_dir>/<prefix>_step_0.safetensors` from the config.
    #[arg(short = 'o', long = "output")]
    pub output: Option<String>,

    /// Override target device: "gpu" or "cpu" (default from config)
    #[arg(long = "target-device")]
    pub target_device: Option<String>,
}

/// One HuggingFace tensor that carries no oxidizr parameter, and why that is
/// expected rather than a mapping failure.
#[derive(Debug)]
struct SkippedTensor {
    hf_name: String,
    reason: &'static str,
}

/// The full name mapping for one import, decided before a single tensor is
/// read so a failure costs no IO.
#[derive(Debug)]
struct MappingPlan {
    /// HuggingFace name -> oxidizr parameter name.
    mapped: BTreeMap<String, String>,
    skipped: Vec<SkippedTensor>,
}

/// The oxidizr name a HuggingFace tensor belongs under.
fn oxidizr_name(hf_name: &str) -> &str {
    hf_name.strip_prefix(HF_DECODER_PREFIX).unwrap_or(hf_name)
}

/// Decide where every HuggingFace tensor goes.
///
/// `expected` is the model's own parameter-name set, so the mapping is checked
/// against the architecture that was actually built rather than a hardcoded
/// table. Anything unaccounted for fails the import: dropping it would train
/// from a weight the checkpoint had and the model never saw.
fn plan_mapping(
    hf_names: &[String],
    expected: &HashSet<String>,
    tie_word_embeddings: bool,
) -> Result<MappingPlan> {
    let mut mapped: BTreeMap<String, String> = BTreeMap::new();
    let mut claimed: HashMap<String, String> = HashMap::new();
    let mut skipped = Vec::new();
    let mut unrecognized = Vec::new();

    for hf_name in hf_names {
        let target = oxidizr_name(hf_name);
        if expected.contains(target) {
            if let Some(previous) = claimed.get(target) {
                return Err(anyhow::anyhow!(
                    "HuggingFace tensors '{previous}' and '{hf_name}' both map to oxidizr \
                     parameter '{target}'. One of them would silently overwrite the other."
                ));
            }
            claimed.insert(target.to_string(), hf_name.clone());
            mapped.insert(hf_name.clone(), target.to_string());
        } else if target == "lm_head.weight" && tie_word_embeddings {
            skipped.push(SkippedTensor {
                hf_name: hf_name.clone(),
                reason: "tied to embed_tokens.weight, which the config already shares",
            });
        } else if target.ends_with(".inv_freq") {
            skipped.push(SkippedTensor {
                hf_name: hf_name.clone(),
                reason: "RoPE frequency buffer, recomputed from rope_theta",
            });
        } else {
            unrecognized.push(hf_name.clone());
        }
    }

    if !unrecognized.is_empty() {
        unrecognized.sort();
        return Err(anyhow::anyhow!(
            "{} HuggingFace tensor(s) have no oxidizr parameter under the given config: {}. \
             Importing would discard them. Fix the config so the architecture matches, or \
             teach the importer what they map to.",
            unrecognized.len(),
            unrecognized.join(", ")
        ));
    }

    Ok(MappingPlan { mapped, skipped })
}

/// The safetensors shards to read, in a stable order.
///
/// A sharded checkpoint lists its files in `model.safetensors.index.json`; a
/// single-file one has no index, so the directory is scanned instead.
fn shard_files(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let content = fs::read_to_string(&index_path)
            .with_context(|| format!("Failed to read '{}'", index_path.display()))?;
        let index: serde_json::Value = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse '{}'", index_path.display()))?;
        let weight_map = index
            .get("weight_map")
            .and_then(|value| value.as_object())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "'{}' has no 'weight_map' object; it is not a safetensors index.",
                    index_path.display()
                )
            })?;
        let mut names: Vec<String> = weight_map
            .values()
            .filter_map(|value| value.as_str())
            .map(|s| s.to_string())
            .collect();
        names.sort();
        names.dedup();
        if names.is_empty() {
            return Err(anyhow::anyhow!(
                "'{}' lists no shard files.",
                index_path.display()
            ));
        }
        return Ok(names.into_iter().map(|name| model_dir.join(name)).collect());
    }

    let mut shards: Vec<PathBuf> = fs::read_dir(model_dir)
        .with_context(|| format!("Failed to read directory '{}'", model_dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(anyhow::anyhow!(
            "'{}' contains no .safetensors file and no model.safetensors.index.json.",
            model_dir.display()
        ));
    }
    Ok(shards)
}

pub fn run_import(args: ImportArgs) -> Result<()> {
    let config = LitConfig::from_yaml(&args.config_file)
        .with_context(|| format!("Failed to load config '{}'", args.config_file))?;

    #[cfg(feature = "cuda")]
    {
        let use_cpu = args
            .target_device
            .as_deref()
            .map(|dev| dev.eq_ignore_ascii_case("cpu"))
            .unwrap_or(config.trainer.target_device == crate::config::TargetDevice::Cpu);
        if !use_cpu {
            return run_import_with_runtime::<boostr::CudaRuntime>(args, config);
        }
    }

    run_import_with_runtime::<boostr::CpuRuntime>(args, config)
}

fn run_import_with_runtime<R>(args: ImportArgs, config: LitConfig) -> Result<()>
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
        + boostr::ShapeOps<R>,
{
    println!("📥 Oxidizr HuggingFace Import");
    println!("=============================\n");

    let device = R::default_device();
    let client = R::default_client(&device);
    println!("Using runtime: {}", R::name());

    let model_dir = PathBuf::from(&args.model_dir);
    if !model_dir.is_dir() {
        return Err(anyhow::anyhow!(
            "model directory '{}' does not exist. Point --model-dir at the directory \
             holding config.json and the .safetensors shards.",
            model_dir.display()
        ));
    }
    if !Path::new(&args.data_file).exists() {
        return Err(anyhow::anyhow!(
            "data file '{}' does not exist. The imported checkpoint's metadata records \
             the data it will be trained on, so --data must name the real file.",
            args.data_file
        ));
    }

    let output = match args.output {
        Some(path) => PathBuf::from(path),
        None => PathBuf::from(&config.trainer.checkpoint_dir).join(format!(
            "{}_step_{}.safetensors",
            config.model.checkpoint_prefix(),
            IMPORT_STEP
        )),
    };

    println!("Config:     {}", args.config_file);
    println!("Model dir:  {}", model_dir.display());
    println!("Data:       {}", args.data_file);
    println!("Output:     {}", output.display());
    println!("dtype:      {}\n", config.model.dtype);

    // Random init first: the model decides which parameters exist, which is
    // what the mapping is checked against.
    let mut varmap = VarMap::<R>::new();
    let mut vb = VarBuilder::new(&mut varmap, &device);
    let mut model = LitGPT::<R>::new(&config.model, &mut vb, &client)?;

    let expected: HashSet<String> = model
        .named_parameters()
        .into_iter()
        .map(|(name, _, _)| name)
        .collect();

    let shards = shard_files(&model_dir)?;
    println!("  Reading {} shard header(s)...", shards.len());

    // Headers first, so the whole mapping is decided — and can fail — before
    // a single gigabyte of weights is read.
    let mut hf_names: Vec<String> = Vec::new();
    for shard in &shards {
        let path = shard.to_string_lossy().into_owned();
        let safetensors = boostr::format::SafeTensors::open(&path)?;
        let names: Vec<String> = safetensors
            .tensor_names()
            .map(|name| name.to_string())
            .collect();
        println!("    {} ({} tensor(s))", path, names.len());
        hf_names.extend(names);
    }

    let plan = plan_mapping(&hf_names, &expected, config.model.tie_word_embeddings)?;

    // Only the mapped tensors are read: a skipped one never occupies memory.
    println!("\n  Reading {} mapped tensor(s)...", plan.mapped.len());
    let mut loaded: HashMap<String, Tensor<R>> = HashMap::with_capacity(plan.mapped.len());
    for shard in &shards {
        let path = shard.to_string_lossy().into_owned();
        let mut safetensors = boostr::format::SafeTensors::open(&path)?;
        let shard_names: Vec<String> = safetensors
            .tensor_names()
            .map(|name| name.to_string())
            .collect();
        for hf_name in shard_names {
            if let Some(target) = plan.mapped.get(&hf_name) {
                let tensor = safetensors.load_tensor::<R>(&hf_name, &device)?;
                loaded.insert(target.clone(), tensor);
            }
        }
    }

    println!("\n  Mapped {} tensor(s)", plan.mapped.len());
    if plan.skipped.is_empty() {
        println!("  Skipped 0 tensor(s)");
    } else {
        println!("  Skipped {} tensor(s):", plan.skipped.len());
        for skipped in &plan.skipped {
            println!("    {} — {}", skipped.hf_name, skipped.reason);
        }
    }

    // Every oxidizr parameter the mapping did not fill is named here and the
    // import fails. This is the same check `--resume` runs, deliberately
    // reused rather than reimplemented — with ONE exemption: the LoRA adapter
    // factors of the targeted projections, which a pretrained base checkpoint
    // cannot contain.
    println!("\n  Loading weights into the model...");
    let fresh_adapters = load_tensors_initializing_lora(&mut model, loaded, &args.model_dir)?;
    report_fresh_adapters(&fresh_adapters);

    if config.model.qk_norm {
        report_qk_norm(&model, &client)?;
    }

    // Nothing is frozen and no optimizer is in play, so each parameter's own
    // `Var` is its value.
    let params: HashMap<TensorId, Tensor<R>> = model
        .named_parameters()
        .into_iter()
        .map(|(_, id, var)| (id, var.tensor().clone()))
        .collect();

    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory '{}'", parent.display()))?;
    }
    let output_path = output.to_string_lossy().into_owned();
    println!("  Writing checkpoint...");
    save_checkpoint_from_params(&output_path, &model, &params)?;

    // The SAME `OxidizrModelConfig` the training run will use, so
    // `compute_model_config_hash` matches and `--resume auto` accepts this
    // checkpoint without editing it.
    let meta_dir = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let meta = CheckpointMeta::new(IMPORT_STEP, &config.model, &args.data_file, &output_path)?;
    meta.save(&meta_dir.to_string_lossy())
        .with_context(|| format!("Failed to write metadata in '{}'", meta_dir.display()))?;

    println!("\n✅ Import complete!");
    if !fresh_adapters.is_empty() {
        println!(
            "   {} LoRA adapter tensor(s) are FRESH, not from '{}'.",
            fresh_adapters.len(),
            model_dir.display()
        );
    }
    println!("\nOutput files:");
    println!("  - {}", output_path);
    println!(
        "  - {}/checkpoint_step_{}.meta.json",
        meta_dir.display(),
        IMPORT_STEP
    );
    println!("\nNext steps:");
    println!(
        "  oxidizr train -f {} -d {} --resume {} --dtype {}",
        args.config_file, args.data_file, output_path, config.model.dtype
    );

    Ok(())
}

/// Say exactly what the source did NOT provide.
///
/// An operator reading "Import complete" must never believe the adapters came
/// from the pretrained model. They are freshly initialized here, with
/// `lora_b` all zeros, so the imported model computes the base model's
/// function until the first optimizer step.
fn report_fresh_adapters(names: &[String]) {
    if names.is_empty() {
        return;
    }
    println!(
        "  Initialized {} LoRA adapter tensor(s) fresh — the source has none, as a \
         pretrained base checkpoint never does.",
        names.len()
    );
    println!("    lora_a: random init   lora_b: zeros (the adapter is a no-op at step 0)");
    for name in names.iter().take(FRESH_ADAPTERS_LISTED) {
        println!("    {name}");
    }
    if names.len() > FRESH_ADAPTERS_LISTED {
        println!("    ... and {} more", names.len() - FRESH_ADAPTERS_LISTED);
    }
}

/// Confirm the per-head QK norms carry pretrained values.
///
/// A loss check cannot tell "QK-norm applied" from "QK-norm dropped": the
/// effect is a rescale, not a collapse. `RmsNorm` initializes its weight to
/// all-ones, so an all-ones `q_norm` after import means the tensor never
/// arrived and the model is running the identity in its place.
fn report_qk_norm<R>(model: &LitGPT<R>, client: &R::Client) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client: RuntimeClient<R> + boostr::TensorOps<R> + boostr::TypeConversionOps<R>,
{
    let named = model.named_parameters();
    let norms: Vec<(String, Tensor<R>)> = named
        .iter()
        .filter(|(name, _, _)| {
            name.ends_with(".self_attn.q_norm.weight") || name.ends_with(".self_attn.k_norm.weight")
        })
        .map(|(name, _, var)| (name.clone(), var.tensor().clone()))
        .collect();

    if norms.is_empty() {
        return Err(anyhow::anyhow!(
            "config sets qk_norm: true but the built model exposes no \
             'self_attn.q_norm.weight'/'k_norm.weight' parameter. The flag did not reach \
             the attention block."
        ));
    }

    let mut all_ones = Vec::new();
    for (name, tensor) in &norms {
        if is_all_ones(tensor, client)? {
            all_ones.push(name.clone());
        }
    }

    if !all_ones.is_empty() {
        all_ones.sort();
        return Err(anyhow::anyhow!(
            "{} QK-norm tensor(s) still hold the all-ones RMSNorm init after import: {}. \
             The pretrained values did not land, and the model would run an identity \
             normalization instead.",
            all_ones.len(),
            all_ones.join(", ")
        ));
    }

    println!(
        "  QK-norm: {} tensor(s) carry pretrained (non all-ones) values",
        norms.len()
    );
    Ok(())
}

fn is_all_ones<R>(tensor: &Tensor<R>, client: &R::Client) -> Result<bool>
where
    R: Runtime<DType = DType>,
    R::Client: RuntimeClient<R> + boostr::TensorOps<R> + boostr::TypeConversionOps<R>,
{
    let contiguous = tensor.contiguous()?;
    let as_f32 = if contiguous.dtype() == DType::F32 {
        contiguous
    } else {
        client.cast(&contiguous, DType::F32)?
    };
    let values = as_f32.contiguous()?.to_vec::<f32>();
    Ok(values.iter().all(|value| (value - 1.0).abs() < 1e-6))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_set(names: &[&str]) -> HashSet<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    fn owned(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn the_decoder_prefix_is_the_whole_mapping() {
        let expected = expected_set(&[
            "embed_tokens.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.q_norm.weight",
            "norm.weight",
        ]);
        let hf = owned(&[
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.norm.weight",
        ]);

        let plan = plan_mapping(&hf, &expected, true).expect("the prefix strip must map cleanly");

        assert_eq!(plan.mapped.len(), 4);
        assert_eq!(
            plan.mapped.get("model.layers.0.self_attn.q_norm.weight"),
            Some(&"layers.0.self_attn.q_norm.weight".to_string())
        );
        assert!(plan.skipped.is_empty());
    }

    #[test]
    fn a_tied_lm_head_is_skipped_with_a_reason() {
        let expected = expected_set(&["embed_tokens.weight"]);
        let hf = owned(&["model.embed_tokens.weight", "lm_head.weight"]);

        let plan = plan_mapping(&hf, &expected, true).expect("a tied lm_head must be skippable");

        assert_eq!(plan.mapped.len(), 1);
        assert_eq!(plan.skipped.len(), 1);
        assert_eq!(plan.skipped[0].hf_name, "lm_head.weight");
    }

    #[test]
    fn an_untied_lm_head_is_not_skipped() {
        // Untied, so `lm_head.weight` is a real parameter and mapping it is
        // the only correct outcome.
        let expected = expected_set(&["embed_tokens.weight", "lm_head.weight"]);
        let hf = owned(&["model.embed_tokens.weight", "lm_head.weight"]);

        let plan = plan_mapping(&hf, &expected, false).expect("an untied lm_head must map");

        assert_eq!(plan.mapped.len(), 2);
        assert!(plan.skipped.is_empty());
    }

    #[test]
    fn an_untied_config_must_not_silently_drop_the_lm_head() {
        // The model has no `lm_head.weight` (it is tied) but the caller said
        // it is not: the tensor has nowhere to go and must fail loudly.
        let expected = expected_set(&["embed_tokens.weight"]);
        let hf = owned(&["model.embed_tokens.weight", "lm_head.weight"]);

        let message = plan_mapping(&hf, &expected, false)
            .expect_err("an unplaceable lm_head must not be dropped")
            .to_string();

        assert!(message.contains("lm_head.weight"), "{message}");
    }

    #[test]
    fn an_unrecognized_tensor_fails_the_import() {
        let expected = expected_set(&["embed_tokens.weight"]);
        let hf = owned(&["model.embed_tokens.weight", "model.layers.0.mystery.weight"]);

        let message = plan_mapping(&hf, &expected, true)
            .expect_err("a tensor with no counterpart must fail the import")
            .to_string();

        assert!(
            message.contains("model.layers.0.mystery.weight"),
            "{message}"
        );
    }

    #[test]
    fn two_tensors_claiming_one_parameter_fail() {
        let expected = expected_set(&["embed_tokens.weight"]);
        let hf = owned(&["model.embed_tokens.weight", "embed_tokens.weight"]);

        let message = plan_mapping(&hf, &expected, true)
            .expect_err("a collision must not let one tensor overwrite the other")
            .to_string();

        assert!(message.contains("embed_tokens.weight"), "{message}");
    }

    #[test]
    fn a_rope_frequency_buffer_is_skipped() {
        let expected = expected_set(&["embed_tokens.weight"]);
        let hf = owned(&[
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.rotary_emb.inv_freq",
        ]);

        let plan = plan_mapping(&hf, &expected, true).expect("a RoPE buffer must be skippable");

        assert_eq!(plan.skipped.len(), 1);
        assert!(plan.skipped[0].reason.contains("RoPE"));
    }
}
