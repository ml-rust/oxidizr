//! Per-layer activation recompute for `trainer.activation_checkpointing`.

mod alias;
#[cfg(test)]
pub(in crate::model) mod fixtures;
mod segment;
