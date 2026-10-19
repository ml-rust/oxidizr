//! `trainer.activation_checkpointing` changes memory only, so a resume can toggle it.
//!
//! Run A trains `STEPS` steps under `none`. Run B trains `HALF` steps under
//! `all`, saves, and resumes under `none`. Run C does the reverse. The final
//! weights and the optimizer state of all three runs must match bit for bit.

mod common;

use boostr::{CpuDevice, CpuRuntime, DType, Tensor};
use common::{
    bits_by_name, build_model, data_loader, scratch_dir, tiny_config, train_parts, trainer,
    trainer_with_decay,
};
use oxidizr::checkpoint_set::{plan_optimizer_resume, read_optimizer_file, CheckpointLabel};
use oxidizr::config::{ActivationCheckpointing, CheckpointMeta, LitConfig};
use oxidizr::data::DataPosition;
use oxidizr::model::load_checkpoint_into_model;
use oxidizr::weight_decay::{inherited_weight_decay, resolve_weight_decay};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const STEPS: usize = 4;
const HALF: usize = STEPS / 2;

/// Final trainable tensors and optimizer state of one run, as bit patterns.
#[derive(Debug, PartialEq)]
struct Finished {
    weights: BTreeMap<String, Vec<u32>>,
    /// Each optimizer tensor, widened to `u64` bits so F64 `__step` fits.
    optimizer: BTreeMap<String, Vec<u64>>,
}

fn with_policy(cfg: &LitConfig, policy: ActivationCheckpointing) -> LitConfig {
    let mut cfg = cfg.clone();
    cfg.trainer.activation_checkpointing = policy;
    cfg
}

fn dir_of(cfg: &LitConfig) -> PathBuf {
    PathBuf::from(&cfg.trainer.checkpoint_dir)
}

fn path_str(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn tensor_bits(tensor: &Tensor<CpuRuntime>) -> Vec<u64> {
    let tensor = tensor.contiguous().expect("contiguous");
    match tensor.dtype() {
        DType::F64 => tensor.to_vec::<f64>().iter().map(|v| v.to_bits()).collect(),
        DType::F32 => tensor
            .to_vec::<f32>()
            .iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        other => panic!("optimizer tensor of unexpected dtype {other:?}"),
    }
}

/// The optimizer state of the final set in `cfg`'s checkpoint dir.
fn final_optimizer(cfg: &LitConfig) -> BTreeMap<String, Vec<u64>> {
    let name = CheckpointLabel::Final.optimizer_file_name(&cfg.model.checkpoint_prefix());
    let state = read_optimizer_file::<CpuRuntime>(&dir_of(cfg).join(name), &CpuDevice::new())
        .expect("read the final optimizer file");
    state
        .iter()
        .map(|(key, tensor)| (key.clone(), tensor_bits(tensor)))
        .collect()
}

/// Train `steps` steps from the stream start. A full run saves the final
/// set and returns it. A shorter run saves the step-`steps` set.
fn train_fresh(cfg: &LitConfig, steps: usize) -> Option<Finished> {
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
        if steps < STEPS {
            t.save_checkpoint(&model, steps).expect("save step set");
            return None;
        }
        t.save_final_checkpoint(&model).expect("save final set");
    }
    Some(Finished {
        weights: bits_by_name(&parts),
        optimizer: final_optimizer(cfg),
    })
}

/// Resume the step-`HALF` set in `cfg`'s dir as `commands::train` does,
/// train to `STEPS`, and save the final set.
fn resume(cfg: &LitConfig) -> Finished {
    let dir = dir_of(cfg);
    let label = CheckpointLabel::Step(HALF);
    let weights = path_str(&dir.join(label.weights_file_name(&cfg.model.checkpoint_prefix())));
    let meta = path_str(&dir.join(label.meta_file_name()));
    let meta_text = fs::read_to_string(&meta).expect("read meta");
    assert!(
        !meta_text.contains("activation_checkpointing"),
        "the meta must not record the policy: {meta_text}"
    );

    let mut model = build_model(cfg);
    load_checkpoint_into_model(&mut model, &weights).expect("load weights");
    let plan = plan_optimizer_resume(Some(&weights), Some(&meta), HALF).expect("plan");
    let position = CheckpointMeta::load(&meta)
        .expect("meta")
        .data_position
        .expect("data position");
    let inherited = inherited_weight_decay(Some(&meta), HALF).expect("inherited rule");
    let weight_decay =
        resolve_weight_decay(&cfg.trainer, inherited.as_ref(), true).expect("weight decay rule");
    let mut parts = train_parts(&model);
    let mut loader = data_loader(cfg);
    let layout = loader.row_layout().expect("row layout");
    loader.seek(position).expect("seek");
    {
        let mut t = trainer_with_decay(cfg, &mut parts, layout, HALF, position, weight_decay);
        t.apply_optimizer_resume(&plan).expect("apply plan");
        for _ in HALF..STEPS {
            t.training_step(&model, &mut loader).expect("training step");
        }
        t.save_final_checkpoint(&model).expect("save final set");
    }
    Finished {
        weights: bits_by_name(&parts),
        optimizer: final_optimizer(cfg),
    }
}

/// A run that trains its first half under `first` and resumes under `second`.
fn toggled(
    base: &LitConfig,
    first: ActivationCheckpointing,
    second: ActivationCheckpointing,
) -> Finished {
    assert!(train_fresh(&with_policy(base, first), HALF).is_none());
    resume(&with_policy(base, second))
}

fn assert_same(want: &Finished, got: &Finished, label: &str) {
    assert_eq!(want.weights.len(), got.weights.len(), "{label}");
    for (name, bits) in &want.weights {
        assert_eq!(
            got.weights.get(name),
            Some(bits),
            "{label}: weight {name} differs"
        );
    }
    assert_eq!(want.optimizer.len(), got.optimizer.len(), "{label}");
    for (key, bits) in &want.optimizer {
        assert_eq!(
            got.optimizer.get(key),
            Some(bits),
            "{label}: optimizer {key} differs"
        );
    }
}

/// Runs A, B and C over `edit`ed copies of the tiny config.
fn assert_toggle_is_exact(tag: &str, edit: impl Fn(&mut LitConfig)) {
    let config_in = |run: &str| {
        let mut cfg = tiny_config(&scratch_dir(&format!("{tag}-{run}")), STEPS);
        edit(&mut cfg);
        cfg
    };
    let a_cfg = config_in("a");
    let a = train_fresh(&a_cfg, STEPS).expect("run A finishes");
    let b_cfg = config_in("b");
    let b = toggled(
        &b_cfg,
        ActivationCheckpointing::All,
        ActivationCheckpointing::None,
    );
    let c_cfg = config_in("c");
    let c = toggled(
        &c_cfg,
        ActivationCheckpointing::None,
        ActivationCheckpointing::All,
    );
    assert!(!a.optimizer.is_empty(), "{tag}: no optimizer state");
    assert_same(&a, &b, &format!("{tag}: all then none"));
    assert_same(&a, &c, &format!("{tag}: none then all"));
    for cfg in [a_cfg, b_cfg, c_cfg] {
        let _ = fs::remove_dir_all(dir_of(&cfg));
    }
}

#[test]
fn a_dense_run_resumes_exactly_across_a_policy_toggle() {
    assert_toggle_is_exact("ac-dense", |_| {});
}

/// MoE layers route the auxiliary losses through the recompute too.
#[test]
fn an_moe_run_resumes_exactly_across_a_policy_toggle() {
    assert_toggle_is_exact("ac-moe", |cfg| {
        cfg.model.num_experts = Some(4);
        cfg.model.experts_per_tok = Some(2);
        cfg.model.shared_expert_enabled = Some(true);
        cfg.trainer.load_balance_alpha = 0.01;
        cfg.trainer.z_loss_alpha = 1e-3;
    });
}
