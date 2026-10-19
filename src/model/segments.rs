//! Which block arms take document segments.

use crate::model::{Block, BlockType};
use crate::prelude::{DType, Result};

impl<R> Block<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// Refuses document segments on a recurrent arm.
    ///
    /// # Errors
    /// Returns an error for a Mamba or Gated DeltaNet arm: its recurrent
    /// state carries across documents, so it cannot honour the segments.
    pub(in crate::model) fn check_segments_supported(&self) -> Result<()> {
        let kind = match &self.block_type {
            BlockType::Attention { .. } | BlockType::LegacyAttention { .. } => return Ok(()),
            BlockType::Mamba { .. } | BlockType::Mamba2 { .. } | BlockType::Mamba3 { .. } => {
                "Mamba"
            }
            BlockType::Gdn(_) => "Gated DeltaNet",
        };
        Err(boostr::error::Error::ModelError {
            reason: format!(
                "a {kind} layer cannot take document segments: its recurrent state carries \
                 across documents. Remove trainer.document_mask, or make every layer an \
                 attention layer."
            ),
        })
    }
}
