//! A tiny Gated DeltaNet + attention hybrid: checkpoint names, exact resume,
//! a falling loss, and LoRA on the GDN mixer.
//!
//! Layer 0 is GDN and layer 1 dense attention, from
//! `common::tiny_gdn_model_config`.

mod common;

use boostr::model::hybrid::GdnBlock;
use boostr::{CpuClient, CpuDevice, CpuRuntime, Tensor, VarBuilder, VarMap};
use common::{
    bits_by_name, build_model, data_loader, scratch_dir, tiny_config, tiny_gdn_model_config,
    train_parts, trainer, trainer_with_decay,
};
use oxidizr::checkpoint_names::to_checkpoint_name;
use oxidizr::checkpoint_set::{optimizer_path_for, plan_optimizer_resume, OptimizerResume};
use oxidizr::config::{find_latest_checkpoint, CheckpointMeta, LitConfig, LoraConfig};
use oxidizr::data::{DataPosition, LitDataLoader};
use oxidizr::loss_scale::read_loss_scale_state;
use oxidizr::model::{load_checkpoint_into_model, save_checkpoint_from_params, LitGPT};
use oxidizr::schedule::LrSchedule;
use oxidizr::trainer::LitTrainer;
use oxidizr::weight_decay::{inherited_weight_decay, resolve_weight_decay};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

const STEPS: usize = 6;
const HALF: usize = STEPS / 2;

/// The GDN mixer projections, which are also the GDN LoRA targets.
const GDN_TARGETS: [&str; 5] = [
    "in_proj_qkv",
    "in_proj_z",
    "alpha_proj",
    "beta_proj",
    "out_proj",
];

type Bits = BTreeMap<String, Vec<u32>>;

fn gdn_config(dir: &Path, max_steps: usize) -> LitConfig {
    let mut cfg = tiny_config(dir, max_steps);
    cfg.model = tiny_gdn_model_config();
    cfg
}

fn tensor_bits(tensor: &Tensor<CpuRuntime>) -> Vec<u32> {
    let values: Vec<f32> = tensor.contiguous().expect("contiguous").to_vec();
    values.iter().map(|v| v.to_bits()).collect()
}

/// Bit patterns of every parameter of `model`, frozen ones included.
fn model_bits(model: &LitGPT<CpuRuntime>) -> Bits {
    let params = model.named_parameters().expect("named parameters");
    params
        .into_iter()
        .map(|(name, _, var)| (name, tensor_bits(var.tensor())))
        .collect()
}

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

/// Resume from `weights` and `meta` as `commands::train` does, then train to `STEPS`.
fn resume(cfg: &LitConfig, weights: &str, meta: &str, start_step: usize) -> Bits {
    let mut model = build_model(cfg);
    load_checkpoint_into_model(&mut model, weights).expect("load weights");
    let plan = plan_optimizer_resume(Some(weights), Some(meta), start_step).expect("plan");
    assert!(matches!(plan, OptimizerResume::Load(_)), "{plan:?}");
    let position = CheckpointMeta::load(meta)
        .expect("meta")
        .data_position
        .expect("data position");
    let inherited = inherited_weight_decay(Some(meta), start_step).expect("inherited rule");
    let weight_decay =
        resolve_weight_decay(&cfg.trainer, inherited.as_ref(), true).expect("weight decay rule");
    let saved_scale = read_loss_scale_state(Some(meta), start_step).expect("loss scale state");
    let mut parts = train_parts(&model);
    let mut loader = data_loader(cfg);
    let layout = loader.row_layout().expect("row layout");
    loader.seek(position).expect("seek");
    {
        let mut t = trainer_with_decay(cfg, &mut parts, layout, start_step, position, weight_decay);
        t.apply_optimizer_resume(&plan).expect("apply plan");
        t.restore_loss_scale(saved_scale)
            .expect("restore loss scale");
        assert_eq!(t.optimizer_timestep(), start_step as u64);
        for _ in start_step..STEPS {
            t.training_step(&model, &mut loader).expect("training step");
        }
    }
    bits_by_name(&parts)
}

#[test]
fn a_gdn_hybrid_resumes_bit_identical() {
    let dir = scratch_dir("gdn-resume");
    let cfg = gdn_config(&dir, STEPS);
    let reference = train_fresh(&cfg, STEPS, |_, _| {});
    let again = train_fresh(&cfg, STEPS, |_, _| {});
    assert_eq!(
        reference, again,
        "two uninterrupted GDN runs differ: the step is not deterministic"
    );
    assert!(reference.contains_key("layers.0.linear_attn.A_log"));

    train_fresh(&cfg, HALF, |t, model| {
        t.save_checkpoint(model, HALF).expect("save step set");
    });
    let (weights, meta, step) = find_latest_checkpoint(&cfg.trainer.checkpoint_dir)
        .expect("scan")
        .expect("the step set is eligible");
    assert_eq!(step, HALF);
    assert!(optimizer_path_for(Path::new(&weights)).is_some_and(|p| p.exists()));

    let resumed = resume(&cfg, &weights, &meta, step);
    assert_eq!(reference.len(), resumed.len());
    for (name, bits) in &reference {
        assert_eq!(Some(bits), resumed.get(name), "tensor '{name}' differs");
    }
    let _ = fs::remove_dir_all(dir);
}

/// The checkpoint carries `model.layers.0.linear_attn.*`. oxidizr reloads it
/// bit for bit, and boostr's `GdnBlock::load` reads the mixer under those
/// names directly.
#[test]
fn gdn_checkpoint_names_round_trip() {
    let dir = scratch_dir("gdn-names");
    let mut cfg = gdn_config(&dir, STEPS);
    let model = build_model(&cfg);
    let params: HashMap<_, _> = model
        .named_parameters()
        .expect("named parameters")
        .into_iter()
        .map(|(_, id, var)| (id, var.tensor().clone()))
        .collect();
    let path = dir.join("gdn.safetensors").to_string_lossy().into_owned();
    save_checkpoint_from_params(&path, &model, &params).expect("save");

    let device = CpuDevice::new();
    let loaded = boostr::format::SafeTensors::open(&path)
        .expect("open checkpoint")
        .load_all::<CpuRuntime>(&device)
        .expect("load checkpoint");
    let mut on_disk: Vec<String> = loaded.keys().cloned().collect();
    on_disk.sort();
    let mut want: Vec<String> = model_bits(&model)
        .into_keys()
        .map(|name| to_checkpoint_name(&name))
        .collect();
    want.sort();
    assert_eq!(on_disk, want);
    let prefix = "model.layers.0.linear_attn.";
    for leaf in [
        "in_proj_qkv.weight",
        "in_proj_z.weight",
        "alpha_proj.weight",
        "beta_proj.weight",
        "out_proj.weight",
        "conv1d.weight",
        "A_log",
        "dt_bias",
        "norm.weight",
    ] {
        assert!(loaded.contains_key(&format!("{prefix}{leaf}")), "{leaf}");
    }
    assert_eq!(
        loaded[&format!("{prefix}conv1d.weight")].shape(),
        &[32, 1, 4]
    );

    // Another seed draws other values, so a match proves the load.
    cfg.trainer.seed = Some(7);
    let mut reloaded = build_model(&cfg);
    assert_ne!(model_bits(&reloaded), model_bits(&model));
    load_checkpoint_into_model(&mut reloaded, &path).expect("load");
    assert_eq!(model_bits(&reloaded), model_bits(&model));

    let mut mixer = VarMap::<CpuRuntime>::new();
    for (name, tensor) in &loaded {
        if let Some(leaf) = name.strip_prefix(prefix) {
            mixer.insert(leaf.to_string(), tensor.clone());
        }
    }
    let gdn = cfg.model.gdn.as_ref().expect("gdn spec");
    let client = CpuClient::new(device.clone());
    let mut vb = VarBuilder::new(&mut mixer, &device);
    let block = GdnBlock::load(
        &gdn.to_boostr(cfg.model.hidden_size),
        &mut vb,
        &client,
        true,
    )
    .expect("boostr reads the oxidizr mixer names");
    for (name, var) in block.named_parameters() {
        let saved = &loaded[&format!("{prefix}{name}")];
        assert_eq!(tensor_bits(var.tensor()), tensor_bits(saved), "{name}");
    }
    let _ = fs::remove_dir_all(dir);
}

/// Every row holds the same tokens, so both steps score the same batch, and
/// one optimizer step must lower its loss.
#[test]
fn two_steps_on_a_repeated_row_lower_the_loss() {
    let dir = scratch_dir("gdn-loss");
    let mut cfg = gdn_config(&dir, 2);
    cfg.trainer.learning_rate = 1e-2;
    cfg.trainer.warmup_steps = None;
    cfg.trainer.lr_schedule = LrSchedule::Constant;
    let model = build_model(&cfg);
    let mut parts = train_parts(&model);
    let row_len = cfg.model.max_seq_len + 1;
    let vocab = cfg.model.vocab_size as u32;
    let row: Vec<u32> = (0..row_len as u32).map(|i| (i * 5 + 1) % vocab).collect();
    let tokens: Vec<u32> = row.iter().cycle().take(row_len * 64).copied().collect();
    let mut loader = LitDataLoader::<CpuRuntime>::new(
        tokens,
        cfg.trainer.batch_size,
        cfg.model.max_seq_len,
        CpuDevice::new(),
    );
    let layout = loader.row_layout().expect("row layout");
    loader.seek(DataPosition::default()).expect("seek");
    let mut t = trainer(&cfg, &mut parts, layout, 0, DataPosition::default());
    let (first, _) = t.training_step(&model, &mut loader).expect("step 1");
    let (second, _) = t.training_step(&model, &mut loader).expect("step 2");
    assert!(first.is_finite() && second.is_finite(), "{first} {second}");
    assert!(second < first, "loss did not fall: {first} -> {second}");
    drop(t);
    let _ = fs::remove_dir_all(dir);
}

/// With `freeze_base`, a LoRA run on the GDN mixer owns only the adapter
/// factors. Training moves every `lora_b` off zero and leaves every other
/// tensor as built.
#[test]
fn a_lora_run_on_a_gdn_model_trains_only_adapter_factors() {
    let dir = scratch_dir("gdn-lora");
    let mut cfg = gdn_config(&dir, 2);
    cfg.model.lora = Some(LoraConfig {
        rank: 2,
        alpha: 4.0,
        target_modules: GDN_TARGETS.iter().map(|t| t.to_string()).collect(),
        freeze_base: true,
        train_modules: Vec::new(),
    });
    let model = build_model(&cfg);
    let frozen_before: Bits = model_bits(&model)
        .into_iter()
        .filter(|(name, _)| !name.ends_with(".lora_a") && !name.ends_with(".lora_b"))
        .collect();
    let mut parts = train_parts(&model);
    let mut names: Vec<String> = parts.names.iter().map(|(n, _)| n.clone()).collect();
    names.sort();
    let mut want: Vec<String> = GDN_TARGETS
        .iter()
        .flat_map(|t| ["lora_a", "lora_b"].map(|f| format!("layers.0.linear_attn.{t}.{f}")))
        .collect();
    want.sort();
    assert_eq!(names, want);

    let mut loader = data_loader(&cfg);
    let layout = loader.row_layout().expect("row layout");
    loader.seek(DataPosition::default()).expect("seek");
    {
        let mut t = trainer(&cfg, &mut parts, layout, 0, DataPosition::default());
        for _ in 0..2 {
            t.training_step(&model, &mut loader).expect("training step");
        }
    }
    for (name, bits) in bits_by_name(&parts) {
        if name.ends_with(".lora_b") {
            assert!(bits.iter().any(|b| *b != 0), "{name} never trained");
        }
    }
    let trained = model
        .rebuild_with_parameters(&parts.params)
        .expect("rebuild over the trained adapters");
    let frozen_after: Bits = model_bits(&trained)
        .into_iter()
        .filter(|(name, _)| frozen_before.contains_key(name))
        .collect();
    assert_eq!(frozen_after, frozen_before);
    let _ = fs::remove_dir_all(dir);
}
