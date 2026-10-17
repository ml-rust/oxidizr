//! A run resumed from a checkpoint set continues exactly where it stopped.
//!
//! The reference trains `STEPS` steps without a stop. The interrupted run
//! trains half, saves, and a new trainer resumes from the files on disk.
//! Every trainable tensor must match the reference bit for bit.

mod common;

use boostr::CpuRuntime;
use common::{
    bits_by_name, build_model, data_loader, scratch_dir, tiny_config, tiny_model_config,
    train_parts, trainer, TrainParts,
};
use oxidizr::checkpoint_set::{
    manual_resume_target, optimizer_path_for, plan_optimizer_resume, remove_old_checkpoints,
    CheckpointLabel, NameMaps, OptimizerResume,
};
use oxidizr::config::{find_latest_checkpoint, CheckpointMeta, LitConfig};
use oxidizr::data::DataPosition;
use oxidizr::model::{load_checkpoint_into_model, LitGPT};
use oxidizr::trainer::LitTrainer;
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
    let dir = scratch_dir(tag);
    let cfg = tiny_config(&dir, STEPS);
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
    let mut parts = train_parts(&model);
    let mut loader = data_loader(cfg);
    let layout = loader.row_layout().expect("row layout");
    loader.seek(position).expect("seek");
    let timestep_after_load = {
        let mut t = trainer(cfg, &mut parts, layout, start_step, position);
        t.apply_optimizer_resume(&plan).expect("apply plan");
        let timestep = t.optimizer_timestep();
        for _ in start_step..STEPS {
            t.training_step(&model, &mut loader).expect("training step");
        }
        timestep
    };
    Resumed {
        bits: bits_by_name(&parts),
        timestep_after_load,
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
