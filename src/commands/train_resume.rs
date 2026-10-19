//! Which checkpoint a training run resumes from, and where its data stream starts.
//!
//! The run resolves its resume point once, before any rank starts. Each rank
//! then loads the same weights and seeks to the same global data position.

use crate::checkpoint_set::manual_resume_target;
use crate::config::{find_latest_checkpoint, CheckpointMeta, LitConfig};
use crate::data::{check_meta_sources, DataPosition, RowLayout};
use anyhow::Result;
use std::path::Path;

/// The checkpoint a run resumes from. All `None` and step 0 for a fresh run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResumePoint {
    /// Completed optimizer steps of the checkpoint.
    pub start_step: usize,
    /// The checkpoint meta, when one exists. It holds the data position and
    /// names the optimizer state file.
    pub meta_path: Option<String>,
    /// The weights file every rank loads.
    pub weights: Option<String>,
}

/// Resolves `--resume`: `auto` finds the latest checkpoint, a path names one.
///
/// `auto` checks the checkpoint against the model config and `data_files`.
/// On a mismatch an interactive run asks to clear the checkpoints and start
/// fresh. A headless run fails instead.
///
/// A path resumes that checkpoint. When its meta records data sources, the
/// data files must match them in count, order, hash and rows.
///
/// # Errors
/// Returns an error when the checkpoint is unusable, the user declines, a
/// headless run finds a mismatch, or a path resume names other data.
pub fn resolve_resume(
    resume: Option<&str>,
    config: &LitConfig,
    data_files: &[String],
    headless: bool,
) -> Result<ResumePoint> {
    match resume {
        None => Ok(ResumePoint::default()),
        Some("auto") => resolve_auto(config, data_files, headless),
        Some(path) => {
            if !headless {
                println!("Loading checkpoint from: {path}");
            }
            // `*_step_N` resumes at step N, `*_final` at the step its meta records.
            let (step, meta_path) = manual_resume_target(path)?;
            if let Some(meta) = &meta_path {
                check_resume_data(meta, config, data_files)?;
            }
            if !headless && step > 0 {
                println!("Resuming from step {step}");
            }
            Ok(ResumePoint {
                start_step: step,
                meta_path,
                weights: Some(path.to_string()),
            })
        }
    }
}

/// Refuses a resume whose data files differ from the sources `meta_path` records.
///
/// A run on generated tokens, or a meta without recorded data, is not checked.
fn check_resume_data(meta_path: &str, config: &LitConfig, data_files: &[String]) -> Result<()> {
    if data_files.is_empty() {
        return Ok(());
    }
    let meta = CheckpointMeta::load(meta_path).map_err(|e| {
        anyhow::anyhow!(
            "failed to read checkpoint metadata {meta_path}: {e}. Restore the file, or pass \
             the checkpoint's .safetensors path to --resume without it."
        )
    })?;
    if meta.data_sources.is_empty() && meta.data_file_hash.is_empty() {
        return Ok(());
    }
    let check = check_meta_sources(&meta, data_files, config.model.max_seq_len)?;
    match check.mismatch {
        Some(mismatch) => anyhow::bail!("checkpoint {meta_path}: {mismatch}"),
        None => Ok(()),
    }
}

fn resolve_auto(config: &LitConfig, data_files: &[String], headless: bool) -> Result<ResumePoint> {
    let checkpoint_dir = &config.trainer.checkpoint_dir;
    if !headless {
        println!("Auto-resume: searching for latest checkpoint in {checkpoint_dir}");
    }
    let Some((checkpoint_path, meta_path, step)) = find_latest_checkpoint(checkpoint_dir)? else {
        // stderr, and not gated on `headless`. `--resume auto` asks to continue
        // from existing weights. A silent fallback to random init looks like a
        // working run in the JSON metrics until the loss refuses to drop.
        eprintln!(
            "⚠️  --resume auto found no checkpoint in '{checkpoint_dir}' — STARTING FROM RANDOM \
             WEIGHTS. A checkpoint is only eligible if it is named `*_step_N.safetensors` AND \
             has a matching `checkpoint_step_N.meta.json` beside it."
        );
        return Ok(ResumePoint::default());
    };
    if !headless {
        println!("Found checkpoint: {checkpoint_path} (step {step})");
    }
    let resumed = ResumePoint {
        start_step: step,
        meta_path: Some(meta_path.clone()),
        weights: Some(checkpoint_path),
    };
    if data_files.is_empty() {
        if !headless {
            println!("⚠️  No data file specified, skipping validation");
            println!("Resuming from step {step}");
        }
        return Ok(resumed);
    }
    let validation =
        CheckpointMeta::load(&meta_path)?.validate_sources(&config.model, data_files)?;
    if validation.is_valid() {
        if let Some(warning) = validation.data_check_warning() {
            // stderr in both modes: an unchecked data file is worth a line even in headless output.
            eprintln!("{warning}");
        } else if !headless {
            println!("✅ Checkpoint validated: model and data match");
        }
        if !validation.dtype_matches && !headless {
            eprintln!(
                "⚠️  dtype mismatch: checkpoint={}, current={}. Weights will be converted.",
                validation.checkpoint_dtype, validation.current_dtype
            );
        }
        if !headless {
            println!("Resuming from step {step}");
        }
        return Ok(resumed);
    }

    println!("\n⚠️  Checkpoint mismatch detected!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    if !validation.model_matches {
        println!("❌ Model config mismatch:");
        println!("   Checkpoint hash: {}", validation.checkpoint_model_hash);
        println!("   Current hash:    {}", validation.current_model_hash);
    }
    if !validation.data_matches {
        println!("❌ Data file mismatch:");
        println!("   Checkpoint hash: {}", validation.checkpoint_data_hash);
        println!("   Current hash:    {}", validation.current_data_hash);
        if let Some(mismatch) = &validation.data_mismatch {
            println!("   {mismatch}");
        }
    }
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\nThe current configuration does not match the existing checkpoint.");
    println!("Starting fresh training will REMOVE all existing checkpoints in:");
    println!("  {checkpoint_dir}\n");

    if headless {
        let data = validation
            .data_mismatch
            .as_deref()
            .map(|m| format!(" Data: {m}"))
            .unwrap_or_default();
        anyhow::bail!(
            "Checkpoint validation failed in headless mode. Clear checkpoints manually or fix \
             config mismatch.{data}"
        );
    }
    if !prompt_confirmation("Do you want to clear old checkpoints and start fresh?") {
        println!(
            "Aborted. Please resolve the configuration mismatch or use a different checkpoint \
             directory."
        );
        anyhow::bail!("Training aborted by user");
    }
    let count = clear_checkpoint_directory(checkpoint_dir).map_err(|e| {
        anyhow::anyhow!("Failed to clear checkpoint directory '{checkpoint_dir}': {e}")
    })?;
    println!("🗑️  Removed {count} checkpoint file(s)");
    println!("Starting fresh training from step 0\n");
    Ok(ResumePoint::default())
}

/// Prompt user for yes/no confirmation. Returns true if user confirms (y/Y).
fn prompt_confirmation(message: &str) -> bool {
    use std::io::{self, Write};

    print!("{} [y/N]: ", message);
    // A flush error leaves the prompt unshown. The answer is still read, so it is ignored.
    let _ = io::stdout().flush();

    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        return false;
    }
    matches!(input.trim().to_lowercase().as_str(), "y" | "yes")
}

/// Clear all checkpoint files from a directory.
///
/// Removes weights and optimizer state (`.safetensors`), metas (`.meta.json`),
/// `config.json`, and temp files that an interrupted atomic write left behind.
fn clear_checkpoint_directory(checkpoint_dir: &str) -> Result<usize> {
    let dir = Path::new(checkpoint_dir);
    if !dir.exists() {
        return Ok(0);
    }
    let mut removed_count = 0;
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // `.safetensors` also matches `.optimizer.safetensors`. A temp file is
        // `.<file_name>.tmp-<pid>-<n>`.
        let is_temp = filename.starts_with('.') && filename.contains(".tmp-");
        if filename.ends_with(".safetensors")
            || filename.ends_with(".meta.json")
            || filename == "config.json"
            || is_temp
        {
            match std::fs::remove_file(&path) {
                Ok(()) => removed_count += 1,
                Err(e) => eprintln!("Warning: Failed to remove {}: {}", path.display(), e),
            }
        }
    }
    Ok(removed_count)
}

/// Global loader position for a run that resumes at `start_step`.
///
/// Reads `data_position` from the checkpoint metadata at `meta_path`. Every
/// rank seeks to it. Metadata written before row loading carries none, and
/// an explicit `--resume <path>` can have no metadata at all. The position
/// then follows from `start_step` under the current global row layout and
/// accumulation. Unless `quiet`, a stderr line says so.
pub fn resume_data_position(
    meta_path: Option<&str>,
    start_step: usize,
    layout: &RowLayout,
    grad_accum: usize,
    quiet: bool,
) -> Result<DataPosition> {
    if let Some(path) = meta_path {
        let meta = CheckpointMeta::load(path).map_err(|e| {
            anyhow::anyhow!(
                "failed to read checkpoint metadata {path}: {e}. Restore the file, or pass \
                 the checkpoint's .safetensors path to --resume without it."
            )
        })?;
        if let Some(position) = meta.data_position {
            return Ok(position);
        }
    }
    let position = layout.position_after_steps(start_step, grad_accum);
    if start_step > 0 && !quiet {
        eprintln!(
            "Checkpoint metadata ({}) records no data position: resuming the data stream at \
             epoch {} row {}, derived from step {start_step} x {grad_accum} batches per step \
             x {} rows per batch.",
            meta_path.unwrap_or("none found"),
            position.epoch,
            position.row,
            layout.batch_size()
        );
    }
    Ok(position)
}
