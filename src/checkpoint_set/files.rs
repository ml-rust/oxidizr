//! File names of a checkpoint set, and the parser that reads a label back.
//!
//! One set holds three files in one directory:
//! - `{prefix}_step_{N}.safetensors`, `{prefix}_final.safetensors` or
//!   `{prefix}_best.safetensors`: the weights
//! - the same stem with `.optimizer.safetensors`: the optimizer state
//! - `checkpoint_step_{N}.meta.json`, `checkpoint_final.meta.json` or
//!   `checkpoint_best.meta.json`: the meta

use std::path::{Path, PathBuf};

/// Suffix of a weights file.
const WEIGHTS_SUFFIX: &str = ".safetensors";
/// Suffix of an optimizer state file. It also ends in `.safetensors`.
pub const OPTIMIZER_SUFFIX: &str = ".optimizer.safetensors";

/// Which checkpoint of a run a set is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointLabel {
    /// An interval checkpoint after `N` completed optimizer steps.
    Step(usize),
    /// The checkpoint written when the run ends.
    Final,
    /// The checkpoint at the lowest eval loss so far. Each new best overwrites it.
    ///
    /// Cleanup never removes it, and `--resume auto` never picks it.
    Best,
}

impl CheckpointLabel {
    fn stem(self, prefix: &str) -> String {
        match self {
            Self::Step(step) => format!("{prefix}_step_{step}"),
            Self::Final => format!("{prefix}_final"),
            Self::Best => format!("{prefix}_best"),
        }
    }

    /// The weights file name for a model whose checkpoint prefix is `prefix`.
    pub fn weights_file_name(self, prefix: &str) -> String {
        format!("{}{WEIGHTS_SUFFIX}", self.stem(prefix))
    }

    /// The optimizer state file name for a model whose checkpoint prefix is `prefix`.
    pub fn optimizer_file_name(self, prefix: &str) -> String {
        format!("{}{OPTIMIZER_SUFFIX}", self.stem(prefix))
    }

    /// The meta file name. It carries no prefix.
    pub fn meta_file_name(self) -> String {
        match self {
            Self::Step(step) => format!("checkpoint_step_{step}.meta.json"),
            Self::Final => "checkpoint_final.meta.json".to_string(),
            Self::Best => "checkpoint_best.meta.json".to_string(),
        }
    }

    /// The label of a weights file name, or `None` for any other file.
    ///
    /// An optimizer state file is never a weights file.
    pub fn of_weights_file(file_name: &str) -> Option<Self> {
        if file_name.ends_with(OPTIMIZER_SUFFIX) {
            return None;
        }
        let stem = file_name.strip_suffix(WEIGHTS_SUFFIX)?;
        if stem.ends_with("_final") {
            return Some(Self::Final);
        }
        if stem.ends_with("_best") {
            return Some(Self::Best);
        }
        let (_, step) = stem.rsplit_once("_step_")?;
        if !step.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        step.parse().ok().map(Self::Step)
    }
}

/// The step of a `*_step_N.safetensors` weights file name.
pub fn weights_step(file_name: &str) -> Option<usize> {
    match CheckpointLabel::of_weights_file(file_name)? {
        CheckpointLabel::Step(step) => Some(step),
        CheckpointLabel::Final | CheckpointLabel::Best => None,
    }
}

/// The optimizer state path beside the weights file at `weights_path`.
///
/// `None` when `weights_path` is not a weights file name.
pub fn optimizer_path_for(weights_path: &Path) -> Option<PathBuf> {
    let name = weights_path.file_name()?.to_str()?;
    if name.ends_with(OPTIMIZER_SUFFIX) {
        return None;
    }
    let stem = name.strip_suffix(WEIGHTS_SUFFIX)?;
    Some(weights_path.with_file_name(format!("{stem}{OPTIMIZER_SUFFIX}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREFIX: &str = "nano_512_117m_f32";

    #[test]
    fn step_names_round_trip() {
        let label = CheckpointLabel::Step(120);
        let weights = label.weights_file_name(PREFIX);
        assert_eq!(weights, "nano_512_117m_f32_step_120.safetensors");
        assert_eq!(CheckpointLabel::of_weights_file(&weights), Some(label));
        assert_eq!(weights_step(&weights), Some(120));
        assert_eq!(label.meta_file_name(), "checkpoint_step_120.meta.json");
        assert_eq!(
            label.optimizer_file_name(PREFIX),
            "nano_512_117m_f32_step_120.optimizer.safetensors"
        );
    }

    #[test]
    fn final_names_round_trip() {
        let label = CheckpointLabel::Final;
        let weights = label.weights_file_name(PREFIX);
        assert_eq!(weights, "nano_512_117m_f32_final.safetensors");
        assert_eq!(CheckpointLabel::of_weights_file(&weights), Some(label));
        assert_eq!(weights_step(&weights), None);
        assert_eq!(label.meta_file_name(), "checkpoint_final.meta.json");
    }

    #[test]
    fn best_names_round_trip_and_have_no_step() {
        let label = CheckpointLabel::Best;
        let weights = label.weights_file_name(PREFIX);
        assert_eq!(weights, "nano_512_117m_f32_best.safetensors");
        assert_eq!(CheckpointLabel::of_weights_file(&weights), Some(label));
        assert_eq!(weights_step(&weights), None);
        assert_eq!(label.meta_file_name(), "checkpoint_best.meta.json");
        assert_eq!(
            label.optimizer_file_name(PREFIX),
            "nano_512_117m_f32_best.optimizer.safetensors"
        );
    }

    #[test]
    fn optimizer_file_is_never_a_weights_checkpoint() {
        let labels = [
            CheckpointLabel::Step(5),
            CheckpointLabel::Final,
            CheckpointLabel::Best,
        ];
        for label in labels {
            let name = label.optimizer_file_name(PREFIX);
            assert_eq!(CheckpointLabel::of_weights_file(&name), None, "{name}");
            assert_eq!(weights_step(&name), None, "{name}");
            assert_eq!(optimizer_path_for(Path::new(&name)), None, "{name}");
        }
    }

    #[test]
    fn other_files_have_no_label() {
        for name in [
            "checkpoint_step_5.meta.json",
            "config.json",
            "model_step_x.safetensors",
            "model_step_+5.safetensors",
            "model.safetensors",
        ] {
            assert_eq!(CheckpointLabel::of_weights_file(name), None, "{name}");
        }
    }

    #[test]
    fn optimizer_path_sits_beside_the_weights() {
        let weights = Path::new("ckpt/run/m_step_7.safetensors");
        assert_eq!(
            optimizer_path_for(weights),
            Some(PathBuf::from("ckpt/run/m_step_7.optimizer.safetensors"))
        );
        assert_eq!(optimizer_path_for(Path::new("ckpt/run/m.json")), None);
    }
}
