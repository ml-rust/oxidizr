//! `trainer.max_grad_value` clamps every gradient element inside a real training step.
//!
//! The step reports the norm the norm clip measured, which comes after the
//! clamp. With every element inside `[-v, v]`, that norm is at most
//! `v * sqrt(n)` over the `n` trainable elements.

mod common;

use common::{build_model, data_loader, scratch_dir, tiny_config, train_parts, trainer};
use oxidizr::config::LitConfig;
use oxidizr::data::DataPosition;

/// The reported gradient norm of one step from the stream start, and the element count.
fn one_step_norm(cfg: &LitConfig) -> (f64, usize) {
    let model = build_model(cfg);
    let mut parts = train_parts(&model);
    let elements: usize = parts.params.values().map(|t| t.elem_count()).sum();
    let mut loader = data_loader(cfg);
    let layout = loader.row_layout().expect("row layout");
    loader.seek(DataPosition::default()).expect("seek");
    let mut t = trainer(cfg, &mut parts, layout, 0, DataPosition::default());
    let (_, grad_norm) = t.training_step(&model, &mut loader).expect("training step");
    (grad_norm, elements)
}

#[test]
fn a_training_step_keeps_every_gradient_element_within_max_grad_value() {
    const MAX_VALUE: f64 = 1e-6;
    let dir = scratch_dir("grad-value-clip");
    let mut cfg = tiny_config(&dir, 1);
    // The norm clip must not fire, so the reported norm is the clamped one.
    cfg.trainer.max_grad_norm = 1e9;

    let (unclipped, elements) = one_step_norm(&cfg);
    let bound = MAX_VALUE * (elements as f64).sqrt();
    assert!(
        unclipped > bound,
        "the gradients ({unclipped}) are already inside the bound ({bound}): the test proves nothing"
    );

    cfg.trainer.max_grad_value = Some(MAX_VALUE);
    let (clipped, _) = one_step_norm(&cfg);
    assert!(
        clipped.is_finite() && clipped > 0.0,
        "clipped norm {clipped}"
    );
    assert!(
        clipped <= bound * (1.0 + 1e-4),
        "norm {clipped} exceeds {bound}: an element is outside +/-{MAX_VALUE}"
    );
    let _ = std::fs::remove_dir_all(dir);
}
