//! `shared_expert_enabled` decides whether MoE layers hold a shared expert.
//!
//! The parameter set, the forward pass, and the exported `config.json` must
//! all follow the flag, so a saved checkpoint and its config agree.

mod common;

use boostr::model::UniversalConfig;
use boostr::{CpuClient, CpuDevice, CpuRuntime, Tensor};
use common::{build_model, scratch_dir, tiny_config};
use oxidizr::checkpoint_config::write_checkpoint_config;
use oxidizr::config::LitConfig;
use std::collections::BTreeSet;
use std::path::Path;

fn moe_config(shared_expert: Option<bool>) -> LitConfig {
    // No checkpoint is written, so the directory is never created.
    let unused_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("oxidizr-shared-expert-unused");
    let mut cfg = tiny_config(&unused_dir, 1);
    cfg.model.num_experts = Some(2);
    cfg.model.experts_per_tok = Some(2);
    cfg.model.shared_expert_enabled = shared_expert;
    cfg
}

fn parameter_names(cfg: &LitConfig) -> BTreeSet<String> {
    build_model(cfg)
        .named_parameters()
        .expect("named parameters")
        .into_iter()
        .map(|(name, _, _)| name)
        .collect()
}

fn forward_runs(cfg: &LitConfig) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let model = build_model(cfg);
    let input =
        Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4], &[1, 4], &device).expect("input ids");
    let output = model.forward(&client, &input).expect("forward");
    let logits = output
        .logits
        .tensor()
        .contiguous()
        .expect("contiguous")
        .to_vec::<f32>();
    assert_eq!(logits.len(), 4 * cfg.model.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
}

fn exported_shared_expert(cfg: &LitConfig, tag: &str) -> Option<bool> {
    let dir = scratch_dir(tag);
    write_checkpoint_config(&cfg.model, &dir.to_string_lossy()).expect("write config.json");
    let json = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
    let universal: UniversalConfig = serde_json::from_str(&json).expect("parse config.json");
    let _ = std::fs::remove_dir_all(dir);
    universal.moe.expect("an MoE config").shared_expert
}

#[test]
fn disabling_the_shared_expert_removes_exactly_its_tensors() {
    let with = parameter_names(&moe_config(Some(true)));
    let without = parameter_names(&moe_config(Some(false)));

    assert!(without.is_subset(&with), "disabling added parameters");
    let removed: Vec<&String> = with.difference(&without).collect();
    assert!(!removed.is_empty(), "the shared expert held no parameters");
    for name in &removed {
        assert!(
            name.contains(".moe.shared_expert."),
            "{name} is not a shared-expert tensor"
        );
    }
    assert!(
        without.iter().all(|name| !name.contains("shared_expert")),
        "a shared-expert tensor survived with the flag off"
    );
    // Every layer loses the gate, up and down projections of its shared expert.
    for layer in 0..moe_config(None).model.num_layers {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let prefix = format!("layers.{layer}.moe.shared_expert.{proj}");
            assert!(
                removed.iter().any(|name| name.starts_with(&prefix)),
                "{prefix} was not removed: {removed:?}"
            );
        }
    }
}

#[test]
fn an_absent_flag_keeps_the_shared_expert() {
    assert_eq!(
        parameter_names(&moe_config(None)),
        parameter_names(&moe_config(Some(true)))
    );
}

#[test]
fn forward_runs_with_and_without_the_shared_expert() {
    forward_runs(&moe_config(Some(true)));
    forward_runs(&moe_config(Some(false)));
}

#[test]
fn the_exported_config_states_the_flag_the_weights_follow() {
    assert_eq!(
        exported_shared_expert(&moe_config(Some(false)), "export-off"),
        Some(false)
    );
    assert_eq!(
        exported_shared_expert(&moe_config(Some(true)), "export-on"),
        Some(true)
    );
    // Absent means a shared expert. boostr reads an absent flag as none, so it is written out.
    assert_eq!(
        exported_shared_expert(&moe_config(None), "export-absent"),
        Some(true)
    );
}
