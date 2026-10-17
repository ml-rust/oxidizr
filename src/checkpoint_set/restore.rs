//! Decide how a resume restores the optimizer, then restore it.

use super::files::{optimizer_path_for, CheckpointLabel};
use super::names::NameMaps;
use super::optimizer_io::load_optimizer_file;
use crate::config::CheckpointMeta;
use crate::prelude::{DType, Error, FusedAdamW, Result};
use boostr::runtime::Runtime;
use std::path::{Path, PathBuf};

/// How a run restores its optimizer state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizerResume {
    /// No state to restore: a fresh run, or a resume at step 0.
    Fresh,
    /// Load the optimizer state file at this path. The resume is exact.
    Load(PathBuf),
    /// The checkpoint at `weights` has no optimizer state. The Adam moments restart.
    Inexact { weights: String },
}

fn read_meta(meta_path: &str) -> Result<CheckpointMeta> {
    CheckpointMeta::load(meta_path).map_err(|e| Error::TrainingError {
        reason: format!(
            "checkpoint meta '{meta_path}' was not read: {e}. Restore the file, or delete it \
             and pass the weights path to `--resume`."
        ),
    })
}

/// Plan the optimizer restore for a resume from `weights_path` at `start_step`.
///
/// `meta_path` is the meta of that checkpoint, when one exists.
/// - A meta that names an optimizer file gives `Load`. A missing file is an error.
/// - Without that entry, an optimizer file beside the weights gives `Load`.
/// - Else step 0 gives `Fresh`, and a later step gives `Inexact`.
///
/// # Errors
/// Returns an error when the meta is unreadable, or names a file that does not exist.
pub fn plan_optimizer_resume(
    weights_path: Option<&str>,
    meta_path: Option<&str>,
    start_step: usize,
) -> Result<OptimizerResume> {
    let Some(weights) = weights_path else {
        return Ok(OptimizerResume::Fresh);
    };
    if let Some(meta_path) = meta_path {
        if let Some(file) = read_meta(meta_path)?.optimizer_file {
            let dir = Path::new(meta_path).parent().unwrap_or(Path::new("."));
            let path = dir.join(&file);
            if !path.exists() {
                return Err(Error::TrainingError {
                    reason: format!(
                        "checkpoint meta '{meta_path}' names optimizer state '{}', which does \
                         not exist. Restore the file, or remove `optimizer_file` from the meta \
                         to resume with fresh Adam moments.",
                        path.display()
                    ),
                });
            }
            return Ok(OptimizerResume::Load(path));
        }
    }
    if let Some(path) = optimizer_path_for(Path::new(weights)).filter(|p| p.exists()) {
        return Ok(OptimizerResume::Load(path));
    }
    if start_step == 0 {
        Ok(OptimizerResume::Fresh)
    } else {
        Ok(OptimizerResume::Inexact {
            weights: weights.to_string(),
        })
    }
}

/// Apply `plan` to `optimizer`. `Inexact` prints a warning and leaves the optimizer fresh.
///
/// # Errors
/// Returns an error when the optimizer state file does not load. The optimizer stays unchanged.
pub fn apply_optimizer_resume<R: Runtime<DType = DType>>(
    optimizer: &mut FusedAdamW<R>,
    name_maps: &NameMaps,
    device: &R::Device,
    plan: &OptimizerResume,
) -> Result<()> {
    match plan {
        OptimizerResume::Fresh => Ok(()),
        OptimizerResume::Load(path) => load_optimizer_file(path, optimizer, name_maps, device),
        OptimizerResume::Inexact { weights } => {
            eprintln!(
                "WARNING: checkpoint '{weights}' has no optimizer state file. This resume \
                 restarts the Adam moments and the optimizer step count at zero. Training \
                 continues, but it does not match an uninterrupted run."
            );
            Ok(())
        }
    }
}

/// Start step and meta path of an explicit `--resume <path>`.
///
/// - `*_step_N.safetensors` resumes at step `N`, with its meta when present.
/// - `*_final.safetensors` resumes at the step its meta records.
/// - Any other file, or a final checkpoint without a meta, resumes at step 0.
///
/// # Errors
/// Returns an error when a final checkpoint's meta is unreadable.
pub fn manual_resume_target(checkpoint: &str) -> Result<(usize, Option<String>)> {
    let path = Path::new(checkpoint);
    let label = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(CheckpointLabel::of_weights_file);
    let Some(label) = label else {
        return Ok((0, None));
    };
    let dir = path.parent().unwrap_or(Path::new("."));
    let meta = dir.join(label.meta_file_name());
    let meta = meta.exists().then(|| meta.to_string_lossy().into_owned());
    match (label, meta) {
        (CheckpointLabel::Step(step), meta) => Ok((step, meta)),
        (CheckpointLabel::Final, Some(meta)) => Ok((read_meta(&meta)?.step, Some(meta))),
        (CheckpointLabel::Final, None) => {
            eprintln!(
                "WARNING: final checkpoint '{checkpoint}' has no checkpoint_final.meta.json. \
                 The run restarts at step 0 and at the start of the data."
            );
            Ok((0, None))
        }
    }
}
