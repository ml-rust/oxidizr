//! Remove the oldest interval checkpoint sets beyond a keep count.

use super::files::{optimizer_path_for, weights_step, CheckpointLabel};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// Keep the `keep` newest `*_step_N` sets in `dir` and remove the rest.
///
/// Each removed set loses its meta first, so a crash mid-removal leaves no
/// meta that names a removed file. `_final` and `_best` sets stay. A removal error is a
/// warning, because the run itself is unaffected.
pub fn remove_old_checkpoints(dir: &Path, keep: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut sets: Vec<(usize, PathBuf)> = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let step = path.file_name()?.to_str().and_then(weights_step)?;
            Some((step, path))
        })
        .collect();
    sets.sort_by_key(|(step, _)| *step);

    let excess = sets.len().saturating_sub(keep);
    for (step, weights) in sets.into_iter().take(excess) {
        let meta = dir.join(CheckpointLabel::Step(step).meta_file_name());
        remove(&meta);
        remove(&weights);
        if let Some(optimizer) = optimizer_path_for(&weights) {
            remove(&optimizer);
        }
    }
}

/// Remove `path`. A missing file is not an error: older sets lack an optimizer file.
fn remove(path: &Path) {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => eprintln!(
            "Warning: old checkpoint file '{}' was not removed: {e}. Remove it by hand.",
            path.display()
        ),
    }
}
