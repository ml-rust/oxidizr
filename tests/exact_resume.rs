//! A run resumed from a checkpoint set continues exactly where it stopped.
//!
//! The reference trains `STEPS` steps without a stop. The interrupted run
//! trains half, saves, and a new trainer resumes from the files on disk.
//! Every trainable tensor must match the reference bit for bit.

mod common;

use boostr::optimizer::GradScalerState;
use boostr::CpuRuntime;
use common::{
    bits_by_name, build_model, data_loader, scratch_dir, tiny_config, tiny_model_config,
    train_parts, trainer, trainer_with_decay, TrainParts,
};
use oxidizr::checkpoint_set::{
    manual_resume_target, optimizer_path_for, plan_optimizer_resume, remove_old_checkpoints,
    CheckpointLabel, NameMaps, OptimizerResume,
};
use oxidizr::config::{find_latest_checkpoint, CheckpointMeta, LitConfig};
use oxidizr::data::DataPosition;
use oxidizr::loss_scale::read_loss_scale_state;
use oxidizr::model::{load_checkpoint_into_model, LitGPT};
use oxidizr::trainer::LitTrainer;
use oxidizr::weight_decay::{
    inherited_weight_decay, resolve_weight_decay, WeightDecayRule, WeightDecaySpec,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const STEPS: usize = 6;
const HALF: usize = STEPS / 2;

type Bits = BTreeMap<String, Vec<u32>>;

/// Train `steps` steps from the stream start, call `save`, and return the trained tensors.
fn train_fresh(
    cfg: &LitConfig,
    steps: usize,
    save: impl FnOnce(&LitTrainer<'_, CpuRuntime>, &LitGPT<CpuRuntime>),
) -> Bits {
    let model = build_model(cfg);
    let mut parts = train_parts(&model);
    let mut loader = data_loader(cfg);
    let layout = loader.row_layout().expect("row layout");
    loader.seek(DataPosition::default()).expect("seek");
    {
        let mut t = trainer(cfg, &mut parts, layout, 0, DataPosition::default());
        for _ in 0..steps {
            t.training_step(&model, &mut loader).expect("training step");
        }
        save(&t, &model);
    }
    bits_by_name(&parts)
}

/// The reference run, and a directory holding the step-`HALF` and final sets of a stopped run.
fn reference_and_stopped_run(tag: &str) -> (LitConfig, Bits, PathBuf) {
    reference_and_stopped_run_with(tag, |_| {})
}

/// [`reference_and_stopped_run`] with `edit` applied to the config of both runs.
fn reference_and_stopped_run_with(
    tag: &str,
    edit: impl FnOnce(&mut LitConfig),
) -> (LitConfig, Bits, PathBuf) {
    let dir = scratch_dir(tag);
    let mut cfg = tiny_config(&dir, STEPS);
    edit(&mut cfg);
    let reference = train_fresh(&cfg, STEPS, |_, _| {});
    train_fresh(&cfg, HALF, |t, model| {
        t.save_checkpoint(model, HALF).expect("save step set");
        t.save_final_checkpoint(model).expect("save final set");
    });
    (cfg, reference, dir)
}

struct Resumed {
    bits: Bits,
    timestep_after_load: u64,
    /// Tensors in the no-decay group of the resumed trainer.
    no_decay_tensors: usize,
    /// Loss scaler state right after the resume, before any step.
    loss_scale_after_load: Option<GradScalerState>,
    /// Optimizer step count and loss scaler state after step `STEPS`.
    timestep_after_run: u64,
    loss_scale_after_run: Option<GradScalerState>,
}

/// Resume from `weights` as `commands::train` does, then train to `STEPS`.
fn resume(cfg: &LitConfig, weights: &str, meta: Option<&str>, start_step: usize) -> Resumed {
    let mut model = build_model(cfg);
    load_checkpoint_into_model(&mut model, weights).expect("load weights");
    let plan = plan_optimizer_resume(Some(weights), meta, start_step).expect("plan");
    let position = meta
        .map(|m| CheckpointMeta::load(m).expect("meta"))
        .and_then(|m| m.data_position)
        .expect("data position");
    // The weight decay rule comes from the meta, as `commands::train` resolves it.
    let inherited = inherited_weight_decay(meta, start_step).expect("inherited rule");
    let weight_decay =
        resolve_weight_decay(&cfg.trainer, inherited.as_ref(), true).expect("weight decay rule");
    let mut parts = train_parts(&model);
    let mut loader = data_loader(cfg);
    let layout = loader.row_layout().expect("row layout");
    loader.seek(position).expect("seek");
    // The loss scale comes from the meta, as `commands::train` restores it.
    let saved_scale = read_loss_scale_state(meta, start_step).expect("loss scale state");
    let (timestep_after_load, no_decay_tensors, loss_scale_after_load, after_run) = {
        let mut t = trainer_with_decay(cfg, &mut parts, layout, start_step, position, weight_decay);
        t.apply_optimizer_resume(&plan).expect("apply plan");
        t.restore_loss_scale(saved_scale)
            .expect("restore loss scale");
        let timestep = t.optimizer_timestep();
        let no_decay = t.decay_plan().no_decay_ids().len();
        let scale = t.loss_scale_state();
        for _ in start_step..STEPS {
            t.training_step(&model, &mut loader).expect("training step");
        }
        (
            timestep,
            no_decay,
            scale,
            (t.optimizer_timestep(), t.loss_scale_state()),
        )
    };
    Resumed {
        bits: bits_by_name(&parts),
        timestep_after_load,
        no_decay_tensors,
        loss_scale_after_load,
        timestep_after_run: after_run.0,
        loss_scale_after_run: after_run.1,
    }
}

fn assert_identical(reference: &Bits, resumed: &Bits) {
    assert_eq!(reference.len(), resumed.len());
    for (name, bits) in reference {
        assert_eq!(Some(bits), resumed.get(name), "tensor '{name}' differs");
    }
}

fn weights_path(cfg: &LitConfig, dir: &Path, label: CheckpointLabel) -> String {
    let name = label.weights_file_name(&cfg.model.checkpoint_prefix());
    dir.join(name).to_string_lossy().into_owned()
}

#[test]
fn auto_resume_is_bit_identical_to_an_uninterrupted_run() {
    let (cfg, reference, dir) = reference_and_stopped_run("resume-auto");
    let again = train_fresh(&cfg, STEPS, |_, _| {});
    assert_eq!(
        reference, again,
        "two uninterrupted runs differ: the step is not deterministic, so no resume can match"
    );
    let (weights, meta, step) = find_latest_checkpoint(&cfg.trainer.checkpoint_dir)
        .expect("scan")
        .expect("the step set is eligible");
    assert_eq!(step, HALF);
    assert!(optimizer_path_for(Path::new(&weights)).is_some_and(|p| p.exists()));
    let plan = plan_optimizer_resume(Some(&weights), Some(&meta), step).expect("plan");
    assert!(matches!(plan, OptimizerResume::Load(_)), "{plan:?}");

    let resumed = resume(&cfg, &weights, Some(&meta), step);
    assert_eq!(resumed.timestep_after_load, HALF as u64);
    assert_identical(&reference, &resumed.bits);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn an_old_meta_without_a_rule_resumes_bit_identical_under_decay_all() {
    // The run that wrote the old meta decayed every parameter.
    let (mut cfg, reference, dir) = reference_and_stopped_run_with("resume-old-rule", |cfg| {
        cfg.trainer.weight_decay_rule = Some(WeightDecayRule::DecayAll);
    });
    let meta_path = dir.join(CheckpointLabel::Step(HALF).meta_file_name());
    let mut meta = CheckpointMeta::load(&meta_path.to_string_lossy()).expect("meta");
    assert_eq!(meta.weight_decay_rule, Some(WeightDecayRule::DecayAll));
    meta.weight_decay_rule = None;
    meta.save_to(&meta_path).expect("rewrite meta");
    let text = fs::read_to_string(&meta_path).expect("read meta");
    assert!(!text.contains("weight_decay_rule"), "{text}");

    // The resumed config sets no rule, as a config from before the field does.
    cfg.trainer.weight_decay_rule = None;
    let weights = weights_path(&cfg, &dir, CheckpointLabel::Step(HALF));
    let meta = meta_path.to_string_lossy().into_owned();
    let resumed = resume(&cfg, &weights, Some(&meta), HALF);
    assert_eq!(resumed.no_decay_tensors, 0);
    assert_identical(&reference, &resumed.bits);

    // A fresh run with the same config uses skip_1d and differs: the inherited rule matters.
    let skip_1d = train_fresh(&cfg, STEPS, |_, _| {});
    assert!(reference.iter().any(|(n, b)| skip_1d.get(n) != Some(b)));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn a_skip_1d_run_resumes_bit_identical() {
    let (cfg, reference, dir) = reference_and_stopped_run_with("resume-skip-1d", |cfg| {
        cfg.trainer.weight_decay_rule = Some(WeightDecayRule::Skip1d);
    });
    let meta = dir
        .join(CheckpointLabel::Step(HALF).meta_file_name())
        .to_string_lossy()
        .into_owned();
    let recorded = WeightDecaySpec::of_meta(&CheckpointMeta::load(&meta).expect("meta"));
    assert_eq!(recorded.rule, WeightDecayRule::Skip1d);

    let weights = weights_path(&cfg, &dir, CheckpointLabel::Step(HALF));
    let resumed = resume(&cfg, &weights, Some(&meta), HALF);
    assert!(resumed.no_decay_tensors > 0, "skip_1d grouped no tensor");
    assert_eq!(resumed.timestep_after_load, HALF as u64);
    assert_identical(&reference, &resumed.bits);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn final_checkpoint_resumes_exactly() {
    let (cfg, reference, dir) = reference_and_stopped_run("resume-final");
    let weights = weights_path(&cfg, &dir, CheckpointLabel::Final);
    let (step, meta) = manual_resume_target(&weights).expect("final target");
    assert_eq!(step, HALF);
    let meta = meta.expect("final meta");
    assert!(meta.ends_with("checkpoint_final.meta.json"), "{meta}");

    let resumed = resume(&cfg, &weights, Some(&meta), step);
    assert_identical(&reference, &resumed.bits);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn resume_without_optimizer_state_warns_and_still_loads() {
    let (cfg, reference, dir) = reference_and_stopped_run("resume-old");
    let weights = weights_path(&cfg, &dir, CheckpointLabel::Step(HALF));
    let meta_path = dir.join(CheckpointLabel::Step(HALF).meta_file_name());
    // Make the set look like one written before optimizer state existed.
    let mut meta = CheckpointMeta::load(&meta_path.to_string_lossy()).expect("meta");
    meta.optimizer_file = None;
    meta.save_to(&meta_path).expect("rewrite meta");
    let optimizer = optimizer_path_for(Path::new(&weights)).expect("optimizer path");
    fs::remove_file(optimizer).expect("remove optimizer state");

    let meta = meta_path.to_string_lossy().into_owned();
    let plan = plan_optimizer_resume(Some(&weights), Some(&meta), HALF).expect("plan");
    assert_eq!(
        plan,
        OptimizerResume::Inexact {
            weights: weights.clone()
        }
    );
    let resumed = resume(&cfg, &weights, Some(&meta), HALF);
    assert_eq!(resumed.timestep_after_load, 0);
    // Fresh Adam moments change the result: this proves the exact tests discriminate.
    assert!(reference
        .iter()
        .any(|(n, b)| resumed.bits.get(n) != Some(b)));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn meta_naming_a_missing_optimizer_file_is_an_error() {
    let (cfg, _, dir) = reference_and_stopped_run("resume-missing");
    let weights = weights_path(&cfg, &dir, CheckpointLabel::Step(HALF));
    let optimizer = optimizer_path_for(Path::new(&weights)).expect("optimizer path");
    fs::remove_file(&optimizer).expect("remove optimizer state");
    let meta = dir.join(CheckpointLabel::Step(HALF).meta_file_name());

    let err = plan_optimizer_resume(Some(&weights), Some(&meta.to_string_lossy()), HALF)
        .expect_err("a committed set without its optimizer file must not resume");
    assert!(err.to_string().contains(".optimizer.safetensors"), "{err}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn interrupted_save_is_invisible_to_auto_resume() {
    let dir = scratch_dir("resume-interrupted");
    let prefix = "m_512_1k_f32";
    let touch = |name: String| fs::write(dir.join(name), b"").expect("write file");
    let committed = CheckpointLabel::Step(4);
    touch(committed.weights_file_name(prefix));
    touch(committed.optimizer_file_name(prefix));
    touch(committed.meta_file_name());
    // Stopped before the meta: weights and optimizer state only.
    let stopped = CheckpointLabel::Step(8);
    touch(stopped.weights_file_name(prefix));
    touch(stopped.optimizer_file_name(prefix));
    // Stopped before the weights: optimizer state only.
    touch(CheckpointLabel::Step(12).optimizer_file_name(prefix));

    let (weights, _, step) = find_latest_checkpoint(&dir.to_string_lossy())
        .expect("scan")
        .expect("the committed set is eligible");
    assert_eq!(step, 4);
    assert!(weights.ends_with(&committed.weights_file_name(prefix)));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn cleanup_removes_whole_old_sets_and_keeps_final() {
    let dir = scratch_dir("resume-cleanup");
    let prefix = "m_512_1k_f32";
    let mut labels: Vec<CheckpointLabel> = (1..=4).map(CheckpointLabel::Step).collect();
    labels.push(CheckpointLabel::Final);
    for label in &labels {
        for name in [
            label.weights_file_name(prefix),
            label.optimizer_file_name(prefix),
            label.meta_file_name(),
        ] {
            fs::write(dir.join(name), b"").expect("write file");
        }
    }

    remove_old_checkpoints(&dir, 2);

    for label in labels {
        let kept = !matches!(label, CheckpointLabel::Step(1) | CheckpointLabel::Step(2));
        for name in [
            label.weights_file_name(prefix),
            label.optimizer_file_name(prefix),
            label.meta_file_name(),
        ] {
            assert_eq!(dir.join(&name).exists(), kept, "{name}");
        }
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn tied_model_names_each_trainable_tensor_once() {
    let dir = scratch_dir("resume-tied");
    let mut cfg = tiny_config(&dir, STEPS);
    cfg.model = tiny_model_config(true);
    let model = build_model(&cfg);
    let TrainParts { ids, names, .. } = train_parts(&model);
    let maps = NameMaps::new(&names, &ids).expect("name maps");

    assert_eq!(
        maps.names().len(),
        ids.len(),
        "every trainable id has one name"
    );
    assert!(maps.ids().contains_key("model.embed_tokens.weight"));
    assert!(!maps.ids().contains_key("lm_head.weight"));
    let _ = fs::remove_dir_all(dir);
}

/// The f16 loss scale of [`f16_run`]. Three clean steps at a small scale,
/// then a growth to `2^18`. Step 4 scales its backward loss by `2^17`, past
/// f16's maximum, so it must overflow, skip and back off.
#[cfg(feature = "f16")]
fn f16_loss_scale() -> oxidizr::loss_scale::LossScaleConfig {
    oxidizr::loss_scale::LossScaleConfig {
        initial: 16.0,
        growth_factor: 16384.0,
        backoff_factor: 0.5,
        growth_interval: 3,
    }
}

#[cfg(feature = "f16")]
fn f16_run(cfg: &mut LitConfig) {
    cfg.model.dtype = oxidizr::config::DTypePrecision::F16;
    cfg.trainer.loss_scale = Some(f16_loss_scale());
}

#[cfg(feature = "f16")]
#[test]
fn an_f16_run_resumes_bit_identical_with_its_loss_scale() {
    let (cfg, reference, dir) = reference_and_stopped_run_with("resume-f16", f16_run);
    let mut reference_end = (0, None);
    let again = train_fresh(&cfg, STEPS, |t, _| {
        reference_end = (t.optimizer_timestep(), t.loss_scale_state());
    });
    assert_eq!(reference, again, "two uninterrupted f16 runs differ");
    assert!(
        reference_end.0 < STEPS as u64,
        "step 4 must overflow and skip: timestep {}",
        reference_end.0
    );

    let meta = dir
        .join(CheckpointLabel::Step(HALF).meta_file_name())
        .to_string_lossy()
        .into_owned();
    let saved = CheckpointMeta::load(&meta).expect("meta").loss_scale;
    let grown = GradScalerState {
        scale: 16.0 * 16384.0,
        consecutive_ok: 0,
    };
    assert_eq!(
        saved,
        Some(grown),
        "three clean steps must grow the scale once"
    );

    let weights = weights_path(&cfg, &dir, CheckpointLabel::Step(HALF));
    let resumed = resume(&cfg, &weights, Some(&meta), HALF);
    assert_eq!(resumed.loss_scale_after_load, Some(grown));
    assert_eq!(resumed.timestep_after_load, HALF as u64);
    assert_eq!(resumed.timestep_after_run, reference_end.0);
    assert_eq!(resumed.loss_scale_after_run, reference_end.1);
    assert_identical(&reference, &resumed.bits);
    let _ = fs::remove_dir_all(dir);
}

/// Without the saved state the scale restarts small, step 4 does not
/// overflow, and the run differs: the restore is what makes the resume exact.
#[cfg(feature = "f16")]
#[test]
fn an_f16_meta_without_loss_scale_state_resumes_inexact() {
    let (cfg, reference, dir) = reference_and_stopped_run_with("resume-f16-old", f16_run);
    let meta_path = dir.join(CheckpointLabel::Step(HALF).meta_file_name());
    let mut meta = CheckpointMeta::load(&meta_path.to_string_lossy()).expect("meta");
    meta.loss_scale = None;
    meta.save_to(&meta_path).expect("rewrite meta");
    let text = fs::read_to_string(&meta_path).expect("read meta");
    assert!(!text.contains("loss_scale"), "{text}");

    let weights = weights_path(&cfg, &dir, CheckpointLabel::Step(HALF));
    let meta = meta_path.to_string_lossy().into_owned();
    let resumed = resume(&cfg, &weights, Some(&meta), HALF);
    assert_eq!(
        resumed.loss_scale_after_load,
        Some(GradScalerState {
            scale: 16.0,
            consecutive_ok: 0
        })
    );
    assert!(reference
        .iter()
        .any(|(n, b)| resumed.bits.get(n) != Some(b)));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn an_f32_meta_records_no_loss_scale() {
    let (_, _, dir) = reference_and_stopped_run("resume-f32-no-scale");
    let meta_path = dir.join(CheckpointLabel::Step(HALF).meta_file_name());
    let text = fs::read_to_string(&meta_path).expect("read meta");
    assert!(!text.contains("loss_scale"), "{text}");
    let _ = fs::remove_dir_all(dir);
}
