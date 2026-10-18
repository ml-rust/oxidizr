//! The weight decay rule reaches AdamW through `LitTrainer`'s parameter group.
//!
//! The grouping runs over a real hybrid model: one Mamba2 layer and one attention layer.

mod common;

use boostr::autograd::GradStore;
use boostr::optimizer::{AdamW, AdamWConfig};
use boostr::{CpuClient, CpuDevice, CpuRuntime, Tensor};
use common::{
    bits_by_name, build_model, data_loader, scratch_dir, tiny_config, train_parts, trainer,
    TrainParts,
};
use oxidizr::checkpoint_set::NameMaps;
use oxidizr::config::LitConfig;
use oxidizr::data::DataPosition;
use oxidizr::weight_decay::{DecayPlan, WeightDecayRule, WeightDecaySpec};
use std::collections::BTreeMap;
use std::path::Path;

/// `tiny_config` with layer 0 a Mamba2 layer. 16 * 2 == 4 heads * 8.
fn hybrid_config() -> LitConfig {
    let mut cfg = tiny_config(Path::new("unused-checkpoints"), 2);
    cfg.model.mamba_layers = Some(vec![0]);
    cfg.model.mamba2_num_heads = Some(4);
    cfg.model.mamba2_head_dim = Some(8);
    cfg.model.mamba2_state_size = Some(8);
    cfg.model.mamba2_chunk_size = Some(4);
    cfg.model.mamba2_n_groups = Some(1);
    cfg.model.mamba2_conv_kernel = Some(4);
    cfg.model.mamba2_expand = Some(2);
    cfg
}

fn skip_1d() -> WeightDecaySpec {
    WeightDecaySpec {
        rule: WeightDecayRule::Skip1d,
        no_decay_patterns: Vec::new(),
    }
}

/// The plan `LitTrainer::new` builds for `parts`, with each checkpoint name and rank.
fn plan_and_names(
    parts: &TrainParts,
    spec: WeightDecaySpec,
) -> (DecayPlan, BTreeMap<String, (bool, usize)>) {
    let maps = NameMaps::new(&parts.names, &parts.ids).expect("name maps");
    let plan = DecayPlan::for_params(spec, &parts.ids, &parts.params, maps.names()).expect("plan");
    let names = maps
        .names()
        .iter()
        .map(|(id, name)| {
            let excluded = plan.no_decay_ids().contains(id);
            (name.clone(), (excluded, parts.params[id].shape().len()))
        })
        .collect();
    (plan, names)
}

#[test]
fn skip_1d_groups_norms_biases_and_ssm_tensors_of_a_hybrid_model() {
    let cfg = hybrid_config();
    let parts = train_parts(&build_model(&cfg));
    let (_, names) = plan_and_names(&parts, skip_1d());
    let no_decay = |suffix: &str| {
        names
            .iter()
            .filter(|(name, _)| name.ends_with(suffix))
            .map(|(name, (excluded, _))| (name.clone(), *excluded))
            .collect::<Vec<_>>()
    };

    let a_log = no_decay(".A_log");
    assert!(
        !a_log.is_empty(),
        "the Mamba2 layer has no a_log: {names:?}"
    );
    for suffix in [".A_log", ".D", ".dt_bias", ".bias", "norm.weight"] {
        for (name, excluded) in no_decay(suffix) {
            assert!(excluded, "'{name}' decays");
        }
    }
    for (name, (excluded, rank)) in &names {
        if *rank < 2 {
            assert!(excluded, "rank-{rank} tensor '{name}' decays");
        }
    }
    for name in ["model.embed_tokens.weight", "lm_head.weight"] {
        assert_eq!(names.get(name).map(|(e, _)| *e), Some(false), "{name}");
    }
    let projections = names
        .iter()
        .filter(|(name, (_, rank))| *rank >= 2 && name.ends_with("_proj.weight"));
    let mut count = 0;
    for (name, (excluded, _)) in projections {
        assert!(!excluded, "projection '{name}' does not decay");
        count += 1;
    }
    assert!(count > 0, "no projection weight found: {names:?}");
}

#[test]
fn a_zero_gradient_step_moves_only_the_decay_group() {
    let cfg = hybrid_config();
    let mut parts = train_parts(&build_model(&cfg));
    let (plan, _) = plan_and_names(&parts, skip_1d());
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let mut grads = GradStore::new();
    for (id, tensor) in &parts.params {
        let zeros = vec![0.0f32; tensor.elem_count()];
        let grad = Tensor::<CpuRuntime>::from_slice(&zeros, tensor.shape(), &device)
            .expect("zero gradient");
        grads.insert(*id, grad);
    }
    let config = AdamWConfig {
        lr: 0.1,
        weight_decay: 0.1,
        ..Default::default()
    };
    let mut opt =
        AdamW::<CpuRuntime>::with_param_groups(config, plan.param_groups()).expect("optimizer");
    let before = bits_by_name(&parts);
    opt.step(&client, &mut parts.params, &grads)
        .expect("optimizer step");
    let after = bits_by_name(&parts);

    let mut shrunk = 0;
    for (name, id) in &parts.names {
        let (old, new) = (&before[name], &after[name]);
        if plan.no_decay_ids().contains(id) {
            assert_eq!(old, new, "no-decay tensor '{name}' moved");
            continue;
        }
        let magnitude =
            |bits: &[u32]| -> Vec<f32> { bits.iter().map(|b| f32::from_bits(*b).abs()).collect() };
        let (old, new) = (magnitude(old), magnitude(new));
        assert!(
            old.iter().zip(&new).all(|(o, n)| n <= o),
            "decay tensor '{name}' grew"
        );
        if old.iter().zip(&new).any(|(o, n)| n < o) {
            shrunk += 1;
        }
    }
    assert!(shrunk > 0, "no decay tensor shrank");
}

/// Trained tensors after `steps` steps of `cfg` from the stream start.
fn train(cfg: &LitConfig, steps: usize) -> BTreeMap<String, Vec<u32>> {
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
    }
    bits_by_name(&parts)
}

#[test]
fn skip_all_trains_like_zero_weight_decay() {
    let dir = scratch_dir("decay-skip-all");
    let mut cfg = tiny_config(&dir, 2);
    cfg.trainer.weight_decay_rule = Some(WeightDecayRule::SkipAll);
    let skip_all = train(&cfg, 2);

    cfg.trainer.weight_decay_rule = Some(WeightDecayRule::DecayAll);
    cfg.trainer.weight_decay = 0.0;
    assert_eq!(
        skip_all,
        train(&cfg, 2),
        "the zero-decay group is not applied"
    );

    cfg.trainer.weight_decay = 0.1;
    let decayed = train(&cfg, 2);
    assert!(
        skip_all.iter().any(|(n, b)| decayed.get(n) != Some(b)),
        "weight decay changed nothing, so this test proves nothing"
    );
    let _ = std::fs::remove_dir_all(dir);
}
