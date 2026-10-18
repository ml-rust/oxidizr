//! The data-parallel keys of a checkpoint meta on resume: the world size and the global batch.

use crate::config::CheckpointMeta;
use crate::prelude::{Error, Result};

/// Checks the meta at `meta_path` against a run of `world` ranks at `start_step`.
///
/// `global_batch_rows` is this run's rows per global batch: the per-rank
/// `batch_size` times `world`. A meta without `world_size` reads as 1 rank.
/// A run at step 0 and a resume without a meta check nothing.
///
/// Returns a warning when the meta records another global batch. The data
/// stream then resumes at the saved row, but not on the saved batch bounds.
///
/// # Errors
/// Returns an error when the meta is unreadable, or records another world
/// size. Every rank restores the same optimizer state and data position, so
/// the rank count must match.
pub fn check_resume_world(
    meta_path: Option<&str>,
    start_step: usize,
    world: usize,
    global_batch_rows: usize,
) -> Result<Option<String>> {
    let Some(path) = meta_path.filter(|_| start_step > 0) else {
        return Ok(None);
    };
    let meta = CheckpointMeta::load(path).map_err(|e| Error::TrainingError {
        reason: format!(
            "checkpoint meta '{path}' was not read: {e}. Restore the file to resume with its \
             world size."
        ),
    })?;
    let saved = meta.world_size.unwrap_or(1);
    if saved != world {
        return Err(Error::TrainingError {
            reason: format!(
                "checkpoint meta '{path}' records world size {saved}, and this run has world \
                 size {world}. Resume with {saved} GPU id(s) in `--gpus`, or start a fresh run."
            ),
        });
    }
    Ok(meta
        .global_batch_rows
        .filter(|&rows| rows != global_batch_rows)
        .map(|rows| {
            format!(
                "Warning: checkpoint meta '{path}' records global batches of {rows} rows, and \
                 this run's global batch is {global_batch_rows} rows. The data stream resumes \
                 at the saved row, but the resume does not match an uninterrupted run."
            )
        }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LitConfig;
    use std::path::PathBuf;

    /// A meta file with the given data-parallel keys, in its own temp directory.
    fn meta_file(tag: &str, world: Option<usize>, rows: Option<usize>) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("oxidizr-resume-world-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let cfg = LitConfig::nano();
        let mut meta =
            CheckpointMeta::new(3, &cfg.model, None, "w.safetensors", None).expect("meta");
        meta.world_size = world;
        meta.global_batch_rows = rows;
        let path = dir.join("checkpoint_step_3.meta.json");
        meta.save_to(&path).expect("save meta");
        path
    }

    #[test]
    fn a_matching_world_and_batch_pass_without_a_warning() {
        let path = meta_file("match", Some(2), Some(8));
        let got = check_resume_world(path.to_str(), 3, 2, 8).expect("same world");
        assert_eq!(got, None);
    }

    #[test]
    fn another_world_size_is_an_error_naming_both_values() {
        let path = meta_file("mismatch", Some(2), Some(8));
        let err = check_resume_world(path.to_str(), 3, 1, 4).expect_err("world 2 at 1");
        let text = err.to_string();
        assert!(text.contains("world size 2"), "{text}");
        assert!(text.contains("world size 1"), "{text}");
        assert!(text.contains("--gpus"), "{text}");
    }

    #[test]
    fn an_old_meta_reads_as_one_rank() {
        let path = meta_file("old", None, None);
        assert_eq!(
            check_resume_world(path.to_str(), 3, 1, 4).expect("world 1"),
            None
        );
        let err = check_resume_world(path.to_str(), 3, 2, 8).expect_err("world 1 at 2");
        assert!(err.to_string().contains("world size 1"), "{err}");
    }

    #[test]
    fn another_global_batch_warns() {
        let path = meta_file("batch", Some(1), Some(4));
        let warning = check_resume_world(path.to_str(), 3, 1, 6)
            .expect("same world")
            .expect("a warning");
        assert!(
            warning.contains("4 rows") && warning.contains("6 rows"),
            "{warning}"
        );
    }

    #[test]
    fn step_zero_and_no_meta_check_nothing() {
        let path = meta_file("step-zero", Some(2), Some(8));
        assert_eq!(
            check_resume_world(path.to_str(), 0, 1, 4).expect("step 0"),
            None
        );
        assert_eq!(check_resume_world(None, 5, 3, 12).expect("no meta"), None);
    }
}
