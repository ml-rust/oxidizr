//! Gated DeltaNet (GDN) layers of [`LitGPT`](super::LitGPT).
//!
//! - `layer`: [`GdnLayer`], the norm, mixer, norm and FFN of one layer, and
//!   [`GdnLayerWithIds`], its tensors with their stable ids
//! - `mixer`: boostr's `GdnBlock` built from tensors with ids, and the
//!   per-step snapshot
//! - `names`: checkpoint names
//! - `lora`: adapter base ids and the adapter merge

mod layer;
mod lora;
mod mixer;
mod names;

pub(super) use layer::{GdnLayer, GdnLayerWithIds};
