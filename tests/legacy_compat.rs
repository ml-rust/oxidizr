//! Files from older oxidizr versions still load.
//!
//! - A checkpoint meta without `hash_algo` resumes. A data hash that differs
//!   warns and is never an error.
//! - A config that still carries the removed GRPO keys parses.

mod common;

use common::{scratch_dir, tiny_config, tokens, write_config_yaml, write_token_file};
use oxidizr::config::{
    compute_model_config_hash, legacy_model_config_hash, CheckpointMeta, LitConfig,
};
use oxidizr::data_hash::{legacy_file_hash, HashAlgo};
use std::path::{Path, PathBuf};

/// A data file and the meta a current trainer writes for it.
fn data_and_meta(tag: &str) -> (PathBuf, LitConfig, String, CheckpointMeta) {
    let dir = scratch_dir(tag);
    let cfg = tiny_config(&dir, 2);
    let data = dir.join("tokens.bin");
    write_token_file(&data, &tokens(&cfg, 4000));
    let data_path = data.to_string_lossy().into_owned();
    let meta =
        CheckpointMeta::new(2, &cfg.model, Some(&data_path), "w.safetensors", None).expect("meta");
    (dir, cfg, data_path, meta)
}

/// `meta` as an older oxidizr wrote it: no `hash_algo` key, `DefaultHasher` hashes.
fn as_legacy_json(meta: &CheckpointMeta, cfg: &LitConfig, data_hash: &str) -> String {
    let mut json = serde_json::to_value(meta).expect("serialize meta");
    let map = json.as_object_mut().expect("meta object");
    map.remove("hash_algo");
    map.insert(
        "model_config_hash".into(),
        legacy_model_config_hash(&cfg.model).into(),
    );
    map.insert("data_file_hash".into(), data_hash.into());
    json.to_string()
}

fn load_json(dir: &Path, json: &str) -> CheckpointMeta {
    let path = dir.join("checkpoint_step_2.meta.json");
    std::fs::write(&path, json).expect("write meta");
    CheckpointMeta::load(&path.to_string_lossy()).expect("load meta")
}

#[test]
fn a_new_meta_records_the_stable_algorithm_and_checks_the_data_file() {
    let (dir, cfg, data_path, meta) = data_and_meta("meta-stable");
    assert_eq!(meta.hash_algo, HashAlgo::Fnv1a64);
    assert_eq!(
        meta.model_config_hash,
        compute_model_config_hash(&cfg.model)
    );

    let same = meta.validate(&cfg.model, &data_path).expect("validate");
    assert!(same.is_valid() && same.data_checked);
    assert_eq!(same.data_check_warning(), None);

    let other = dir.join("other.bin");
    write_token_file(&other, &tokens(&cfg, 3000));
    let changed = meta
        .validate(&cfg.model, &other.to_string_lossy())
        .expect("validate");
    assert!(!changed.data_matches, "a different data file matched");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_legacy_meta_with_a_differing_data_hash_resumes_with_a_warning() {
    let (dir, cfg, data_path, meta) = data_and_meta("meta-legacy-warn");
    let old = load_json(&dir, &as_legacy_json(&meta, &cfg, "0123456789abcdef"));
    assert_eq!(old.hash_algo, HashAlgo::Legacy);

    let validation = old
        .validate(&cfg.model, &data_path)
        .expect("never an error");

    assert!(
        validation.model_matches,
        "the legacy model hash did not match"
    );
    assert!(
        validation.is_valid(),
        "a legacy data hash blocked the resume"
    );
    assert!(!validation.data_checked);
    let warning = validation.data_check_warning().expect("a warning");
    assert!(warning.contains("data file is unchecked"), "{warning}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_legacy_meta_with_a_matching_legacy_data_hash_resumes_checked() {
    let (dir, cfg, data_path, meta) = data_and_meta("meta-legacy-match");
    let legacy = legacy_file_hash(&data_path).expect("legacy hash");
    let old = load_json(&dir, &as_legacy_json(&meta, &cfg, &legacy));

    let validation = old.validate(&cfg.model, &data_path).expect("validate");

    assert!(validation.is_valid() && validation.data_checked);
    assert_eq!(validation.data_check_warning(), None);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_config_with_the_removed_grpo_keys_still_parses() {
    let dir = scratch_dir("grpo-keys");
    let cfg = tiny_config(&dir.join("ckpt"), 2);
    let path = dir.join("config.yaml");
    write_config_yaml(&path, &cfg);
    let mut yaml: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&path).expect("read")).expect("parse");
    let trainer = yaml
        .get_mut("trainer")
        .and_then(|t| t.as_mapping_mut())
        .expect("trainer section");
    use serde_yaml::{Number, Value};
    trainer.insert(
        Value::String("group_size".to_string()),
        Value::Number(Number::from(4u64)),
    );
    trainer.insert(
        Value::String("beta_kl".to_string()),
        Value::Number(Number::from(0.04f64)),
    );
    std::fs::write(&path, serde_yaml::to_string(&yaml).expect("serialize")).expect("write");

    let loaded = LitConfig::from_yaml(&path).expect("an old config parses");
    assert_eq!(loaded.trainer.max_steps, 2);
    let _ = std::fs::remove_dir_all(dir);
}
