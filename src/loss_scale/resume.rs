//! Read the loss scale state a checkpoint meta records.

use crate::config::CheckpointMeta;
use crate::prelude::{Error, Result};
use boostr::optimizer::GradScalerState;

/// The scaler state a run at `start_step` resumes from the meta at `meta_path`.
///
/// A run at step 0, a resume without a meta, and a meta without the key
/// all give `None`.
///
/// # Errors
/// Returns an error when the meta is unreadable.
pub fn read_loss_scale_state(
    meta_path: Option<&str>,
    start_step: usize,
) -> Result<Option<GradScalerState>> {
    let Some(path) = meta_path.filter(|_| start_step > 0) else {
        return Ok(None);
    };
    let meta = CheckpointMeta::load(path).map_err(|e| Error::TrainingError {
        reason: format!(
            "checkpoint meta '{path}' was not read: {e}. Restore the file to resume with its \
             loss scale."
        ),
    })?;
    Ok(meta.loss_scale)
}
