//! f16 dynamic loss scaling inside real training steps.
//!
//! An overflow step must leave the parameters and the optimizer timestep
//! unchanged, halve the scale and still advance the data position. Clean
//! steps must grow the scale after `growth_interval`. bf16 and f32 configs
//! must refuse `trainer.loss_scale`.

mod common;

#[cfg(feature = "f16")]
use boostr::optimizer::GradScalerState;
use boostr::{CpuClient, CpuDevice};
use common::{build_model, data_loader, scratch_dir, tiny_config, train_parts, write_config_yaml};
use oxidizr::config::{DTypePrecision, LitConfig};
use oxidizr::data::DataPosition;
use oxidizr::loss_scale::LossScaleConfig;
use oxidizr::trainer::{LitTrainer, TrainerRunConfig};
use oxidizr::weight_decay::resolve_weight_decay;

/// The error `LitTrainer::new` returns for `cfg`, or `None` when it builds.
fn trainer_error(cfg: &LitConfig) -> Option<String> {
    let model = build_model(cfg);
    let mut parts = train_parts(&model);
    let layout = data_loader(cfg).row_layout().expect("row layout");
    let device = CpuDevice::new();
    let weight_decay = resolve_weight_decay(&cfg.trainer, None, true).expect("rule");
    LitTrainer::new(
        cfg.trainer.clone(),
        &mut parts.params,
        parts.ids.clone(),
        parts.names.clone(),
        CpuClient::new(device.clone()),
        device,
        TrainerRunConfig {
            row_layout: layout,
            start_position: DataPosition::default(),
            batch_size: cfg.trainer.batch_size,
            steps_json_path: format!("{}/steps.json", cfg.trainer.checkpoint_dir),
            headless: true,
            start_step: 0,
            model_config: cfg.model.clone(),
            data_files: Vec::new(),
            weight_decay,
        },
    )
    .err()
    .map(|e| e.to_string())
}

#[test]
fn bf16_and_f32_configs_refuse_loss_scale() {
    for (dtype, name) in [(DTypePrecision::BF16, "bf16"), (DTypePrecision::F32, "f32")] {
        let dir = scratch_dir(&format!("loss-scale-refused-{name}"));
        let mut cfg = tiny_config(&dir, 1);
        cfg.model.dtype = dtype;
        cfg.trainer.loss_scale = Some(LossScaleConfig::default());

        let yaml = dir.join("config.yaml");
        write_config_yaml(&yaml, &cfg);
        let err = LitConfig::from_yaml(&yaml).expect_err("from_yaml must refuse loss_scale");
        let text = err.to_string();
        assert!(text.contains("trainer.loss_scale"), "{text}");
        assert!(text.contains(name), "{text}");

        let text = trainer_error(&cfg).expect("LitTrainer::new must refuse loss_scale");
        assert!(text.contains("trainer.loss_scale"), "{text}");
        assert!(text.contains(name), "{text}");
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn an_f32_trainer_does_not_scale() {
    let dir = scratch_dir("loss-scale-f32-off");
    let cfg = tiny_config(&dir, 1);
    let model = build_model(&cfg);
    let mut parts = train_parts(&model);
    let layout = data_loader(&cfg).row_layout().expect("row layout");
    let t = common::trainer(&cfg, &mut parts, layout, 0, DataPosition::default());
    assert_eq!(t.loss_scale_state(), None);
    let _ = std::fs::remove_dir_all(dir);
}

/// A four-step f16 config with `loss_scale` as its `trainer.loss_scale`.
#[cfg(feature = "f16")]
fn f16_config(tag: &str, loss_scale: Option<LossScaleConfig>) -> (LitConfig, std::path::PathBuf) {
    let dir = scratch_dir(tag);
    let mut cfg = tiny_config(&dir, 4);
    cfg.model.dtype = DTypePrecision::F16;
    cfg.trainer.loss_scale = loss_scale;
    (cfg, dir)
}

#[cfg(feature = "f16")]
#[test]
fn an_f16_trainer_without_the_key_scales_with_the_defaults() {
    let (cfg, dir) = f16_config("loss-scale-f16-default", None);
    let model = build_model(&cfg);
    let mut parts = train_parts(&model);
    let layout = data_loader(&cfg).row_layout().expect("row layout");
    let t = common::trainer(&cfg, &mut parts, layout, 0, DataPosition::default());
    assert_eq!(
        t.loss_scale_state(),
        Some(GradScalerState {
            scale: 65536.0,
            consecutive_ok: 0
        })
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// A scale of `2^20` puts the backward seed past f16's maximum, so the
/// first step must overflow.
#[cfg(feature = "f16")]
#[test]
fn an_overflow_step_changes_no_parameter_and_halves_the_scale() {
    let (cfg, dir) = f16_config(
        "loss-scale-overflow",
        Some(LossScaleConfig {
            initial: 1048576.0,
            ..LossScaleConfig::default()
        }),
    );
    let model = build_model(&cfg);
    let mut parts = train_parts(&model);
    let before = common::bits_by_name(&parts);
    let mut loader = data_loader(&cfg);
    let layout = loader.row_layout().expect("row layout");
    loader.seek(DataPosition::default()).expect("seek");
    {
        let mut t = common::trainer(&cfg, &mut parts, layout, 0, DataPosition::default());
        let (loss, grad_norm) = t.training_step(&model, &mut loader).expect("training step");
        assert!(
            !grad_norm.is_finite(),
            "the step must overflow: norm {grad_norm}"
        );
        assert!(loss.is_finite(), "the reported loss is unscaled: {loss}");
        assert_eq!(t.optimizer_timestep(), 0, "an overflow step must not count");
        assert_eq!(
            t.global_step(),
            1,
            "an overflow step still counts as a step"
        );
        assert_eq!(
            t.loss_scale_state(),
            Some(GradScalerState {
                scale: 524288.0,
                consecutive_ok: 0
            })
        );
        // The skipped step drew its batches: the next step reads the next rows.
        assert_eq!(
            loader.position(),
            layout.position_after_steps(1, cfg.trainer.gradient_accumulation)
        );
        t.save_checkpoint(&model, 1).expect("save");
    }
    assert_eq!(before, common::bits_by_name(&parts), "a parameter changed");

    let meta = dir.join(oxidizr::checkpoint_set::CheckpointLabel::Step(1).meta_file_name());
    let meta = oxidizr::config::CheckpointMeta::load(&meta.to_string_lossy()).expect("meta");
    assert_eq!(
        meta.data_position,
        Some(layout.position_after_steps(1, cfg.trainer.gradient_accumulation))
    );
    assert_eq!(
        meta.loss_scale,
        Some(GradScalerState {
            scale: 524288.0,
            consecutive_ok: 0
        })
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(feature = "f16")]
#[test]
fn clean_steps_grow_the_scale_after_the_interval() {
    let (cfg, dir) = f16_config(
        "loss-scale-growth",
        Some(LossScaleConfig {
            initial: 64.0,
            growth_interval: 2,
            ..LossScaleConfig::default()
        }),
    );
    let model = build_model(&cfg);
    let mut parts = train_parts(&model);
    let before = common::bits_by_name(&parts);
    let mut loader = data_loader(&cfg);
    let layout = loader.row_layout().expect("row layout");
    loader.seek(DataPosition::default()).expect("seek");
    {
        let mut t = common::trainer(&cfg, &mut parts, layout, 0, DataPosition::default());
        let expected = [(64.0, 1), (128.0, 0), (128.0, 1), (256.0, 0)];
        for (step, (scale, consecutive_ok)) in expected.into_iter().enumerate() {
            let (loss, grad_norm) = t.training_step(&model, &mut loader).expect("training step");
            assert!(loss.is_finite() && grad_norm.is_finite(), "step {step}");
            assert_eq!(
                t.loss_scale_state(),
                Some(GradScalerState {
                    scale,
                    consecutive_ok
                }),
                "after step {step}"
            );
        }
        assert_eq!(t.optimizer_timestep(), 4);
    }
    assert_ne!(before, common::bits_by_name(&parts), "no parameter trained");
    let _ = std::fs::remove_dir_all(dir);
}
