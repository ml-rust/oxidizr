//! Commit one checkpoint set in order: optimizer state, weights, meta.

use super::files::CheckpointLabel;
use super::optimizer_io::write_staged_optimizer_file;
use super::snapshot::SetSnapshot;
use crate::config::CheckpointMeta;
use crate::model::save_staged_checkpoint;
use crate::prelude::Result;
use std::path::PathBuf;

/// Write `snapshot` as the set labelled `label`: optimizer state, weights, then the meta.
///
/// Each file is written atomically. The meta is the commit marker. A crash
/// before it leaves a set that `--resume auto` skips. A meta write error is a
/// warning: the weights stay usable through an explicit `--resume <path>`.
/// A set written over an older one, such as `best`, loses the old meta first,
/// so no meta ever names a half-replaced set.
///
/// `release_optimizer` drops the snapshot's optimizer state once its file is
/// written, before the weights encode. Pass it for the last label of a snapshot.
///
/// # Errors
/// Returns an error when the directory, the old meta removal, the optimizer
/// state, or the weights write fails.
pub fn commit_set(
    snapshot: &SetSnapshot,
    label: CheckpointLabel,
    release_optimizer: bool,
) -> Result<()> {
    let files = write_set_files(snapshot, label, release_optimizer)?;
    write_set_meta(snapshot, label, &files);
    Ok(())
}

/// The files of a set whose optimizer state and weights are on disk.
pub(crate) struct SetFiles {
    meta_path: PathBuf,
    optimizer_name: String,
    weights: String,
}

/// Every step of [`commit_set`] before the meta.
pub(crate) fn write_set_files(
    snapshot: &SetSnapshot,
    label: CheckpointLabel,
    release_optimizer: bool,
) -> Result<SetFiles> {
    let dir = snapshot.dir();
    std::fs::create_dir_all(dir)?;
    let meta_path = dir.join(label.meta_file_name());
    if let Err(e) = std::fs::remove_file(&meta_path) {
        if e.kind() != std::io::ErrorKind::NotFound {
            return Err(e.into());
        }
    }
    let optimizer_name = label.optimizer_file_name(snapshot.prefix());
    let optimizer_path = dir.join(&optimizer_name);
    snapshot.with_optimizer(release_optimizer, |state| {
        write_staged_optimizer_file(&optimizer_path, state)
    })?;

    let weights_path = dir.join(label.weights_file_name(snapshot.prefix()));
    let weights = weights_path.to_string_lossy().into_owned();
    save_staged_checkpoint(&weights, snapshot.weights())?;
    Ok(SetFiles {
        meta_path,
        optimizer_name,
        weights,
    })
}

/// The last step of [`commit_set`]: the meta. An error is a warning.
pub(crate) fn write_set_meta(snapshot: &SetSnapshot, label: CheckpointLabel, files: &SetFiles) {
    let mut meta: CheckpointMeta = snapshot.meta().clone();
    meta.checkpoint_file = files.weights.clone();
    meta.optimizer_file = Some(files.optimizer_name.clone());
    meta.timestamp = CheckpointMeta::timestamp_now();
    if let Err(e) = meta.save_to(&files.meta_path) {
        eprintln!(
            "Warning: checkpoint meta '{}' of {label} was not written: {e}. `--resume auto` \
             skips this checkpoint. Pass '{}' to `--resume` to use it.",
            files.meta_path.display(),
            files.weights
        );
    }
}
