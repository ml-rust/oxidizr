//! What a checkpoint meta records of a weighted mixture, and the resume check against it.
//!
//! The weights and `rows_per_epoch` decide the row at every slot, so the
//! saved data position means other rows under other values. A resume must
//! run the same mixture. A meta without a mixture resumes only a run without weights.

use crate::config::CheckpointMeta;
use crate::prelude::{Error, Result};
use serde::{Deserialize, Serialize};

/// The weighted mixture of a run, as a checkpoint meta records it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixtureRecord {
    /// `trainer.data.sources[i].weight`, in source order.
    pub weights: Vec<f64>,
    /// Rows per epoch: `trainer.data.rows_per_epoch`, or every training row of every source.
    pub rows_per_epoch: usize,
}

/// One source of a weighted mixture: its weight, its rows per epoch and its training rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixtureSource {
    /// `trainer.data.sources[i].weight`.
    pub weight: f64,
    /// Rows the source supplies to each epoch.
    pub rows_per_epoch: usize,
    /// Training rows of the source, after the holdout.
    pub train_rows: usize,
}

impl MixtureSource {
    /// Passes over the source's training rows per epoch. Above 1 the source repeats rows.
    pub fn repeats(&self) -> f64 {
        self.rows_per_epoch as f64 / self.train_rows.max(1) as f64
    }
}

const RESUME_HINT: &str = "Resume with the original config, or start a fresh run.";

/// Checks the mixture of the meta at `meta_path` against this run's `current` mixture.
///
/// A run at step 0 and a resume without a meta check nothing.
///
/// # Errors
/// Returns an error when the meta is unreadable, or when its mixture
/// differs from `current`. The error names the key that differs.
pub fn check_mixture_resume(
    meta_path: Option<&str>,
    start_step: usize,
    current: Option<&MixtureRecord>,
) -> Result<()> {
    let Some(path) = meta_path.filter(|_| start_step > 0) else {
        return Ok(());
    };
    let meta = CheckpointMeta::load(path).map_err(|e| Error::DataError {
        reason: format!(
            "checkpoint meta '{path}' was not read: {e}. Restore the file to resume with its \
             data mixture."
        ),
    })?;
    match mixture_mismatch(meta.mixture.as_ref(), current) {
        Some(reason) => Err(Error::DataError {
            reason: format!(
                "checkpoint meta '{path}': {reason} The mixture decides which rows the saved \
                 data position counts. {RESUME_HINT}"
            ),
        }),
        None => Ok(()),
    }
}

/// What differs between the `saved` and `current` mixtures. `None` when they match.
fn mixture_mismatch(
    saved: Option<&MixtureRecord>,
    current: Option<&MixtureRecord>,
) -> Option<String> {
    let (saved, current) = match (saved, current) {
        (None, None) => return None,
        (Some(saved), None) => {
            return Some(format!(
                "the checkpoint trained on trainer.data.sources weights {:?}, and this run sets \
                 no weights.",
                saved.weights
            ))
        }
        (None, Some(current)) => {
            return Some(format!(
                "the checkpoint trained without trainer.data.sources weights, and this run sets \
                 weights {:?}.",
                current.weights
            ))
        }
        (Some(saved), Some(current)) => (saved, current),
    };
    if saved.weights.len() != current.weights.len() {
        return Some(format!(
            "the checkpoint trained on {} weighted trainer.data.sources, and this run sets {}.",
            saved.weights.len(),
            current.weights.len()
        ));
    }
    let changed = saved
        .weights
        .iter()
        .zip(&current.weights)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    if let Some(i) = changed {
        return Some(format!(
            "trainer.data.sources[{i}].weight was {}, and this run sets {}.",
            saved.weights[i], current.weights[i]
        ));
    }
    (saved.rows_per_epoch != current.rows_per_epoch).then(|| {
        format!(
            "trainer.data.rows_per_epoch was {}, and this run reads {} rows per epoch.",
            saved.rows_per_epoch, current.rows_per_epoch
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LitConfig;

    fn record(weights: &[f64], rows: usize) -> MixtureRecord {
        MixtureRecord {
            weights: weights.to_vec(),
            rows_per_epoch: rows,
        }
    }

    #[test]
    fn equal_mixtures_and_two_plain_runs_match() {
        assert_eq!(mixture_mismatch(None, None), None);
        let a = record(&[0.7, 0.3], 100);
        assert_eq!(mixture_mismatch(Some(&a), Some(&a.clone())), None);
    }

    #[test]
    fn each_change_names_its_key() {
        let a = record(&[0.7, 0.3], 100);
        let text = mixture_mismatch(Some(&a), Some(&record(&[0.7, 0.4], 100))).expect("weight");
        assert!(
            text.contains("trainer.data.sources[1].weight was 0.3"),
            "{text}"
        );
        let text = mixture_mismatch(Some(&a), Some(&record(&[0.7, 0.3], 90))).expect("rows");
        assert!(
            text.contains("trainer.data.rows_per_epoch was 100"),
            "{text}"
        );
        let text = mixture_mismatch(Some(&a), None).expect("dropped");
        assert!(text.contains("sets no weights"), "{text}");
        let text = mixture_mismatch(None, Some(&a)).expect("added");
        assert!(
            text.contains("trained without trainer.data.sources weights"),
            "{text}"
        );
        let text = mixture_mismatch(Some(&a), Some(&record(&[1.0], 100))).expect("count");
        assert!(text.contains("2 weighted"), "{text}");
    }

    #[test]
    fn a_resume_reads_the_meta_and_skips_step_zero() {
        let dir = std::env::temp_dir().join(format!("oxidizr-mixture-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let cfg = LitConfig::nano();
        let mut meta =
            CheckpointMeta::new(3, &cfg.model, None, "w.safetensors", None).expect("meta");
        let saved = record(&[2.0, 1.0], 12);
        meta.mixture = Some(saved.clone());
        let path = dir.join("checkpoint_step_3.meta.json");
        meta.save_to(&path).expect("save meta");
        let path = path.to_str();
        assert!(check_mixture_resume(path, 3, Some(&saved)).is_ok());
        let err = check_mixture_resume(path, 3, None).expect_err("weights dropped");
        let text = err.to_string();
        assert!(text.contains("checkpoint_step_3.meta.json"), "{text}");
        assert!(text.contains(RESUME_HINT), "{text}");
        assert!(
            check_mixture_resume(path, 0, None).is_ok(),
            "step 0 starts fresh"
        );
        assert!(check_mixture_resume(None, 3, Some(&saved)).is_ok());
        let _ = std::fs::remove_dir_all(dir);
    }
}
