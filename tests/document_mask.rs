//! `trainer.document_mask` end to end on CPU.
//!
//! - A packed row of documents A and B has the loss of A and B in rows of their own.
//! - Changing document A leaves the logits of document B bit-identical.
//! - A stopped run resumes bit-identical, and its meta records the mask.
//! - Recompute under the mask trains bit-identical to a run without recompute.
//! - A model with Mamba2 and Gated DeltaNet layers masks and trains the same way.
//! - The YAML loader refuses the mask on a model it cannot mask.

mod common;

use boostr::{CpuClient, CpuDevice, CpuRuntime};
use common::{
    bits_by_name, build_model, document_masked, masked_data_loader, scratch_dir, tiny_config,
    tiny_model_config, train_parts, trainer, trainer_with_decay, write_config_yaml, DOC_SEPARATOR,
};
use oxidizr::checkpoint_set::plan_optimizer_resume;
use oxidizr::config::{
    check_document_mask_resume, find_latest_checkpoint, ActivationCheckpointing, CheckpointMeta,
    DocumentMaskRecord, GdnSpec, LitConfig, OxidizrModelConfig,
};
use oxidizr::data::{Batch, DataPosition, LitDataLoader};
use oxidizr::model::{load_checkpoint_into_model, LitGPT};
use oxidizr::weight_decay::{inherited_weight_decay, resolve_weight_decay};
use std::collections::BTreeMap;
use std::fs;
use std::ops::Range;
use std::path::Path;

const SEP: u32 = DOC_SEPARATOR;
const SEQ: usize = 8;
const STEPS: usize = 6;
const HALF: usize = STEPS / 2;

type Bits = BTreeMap<String, Vec<u32>>;

/// [`tiny_model_config`] with GQA: 4 query heads over 2 KV heads.
fn gqa() -> OxidizrModelConfig {
    OxidizrModelConfig {
        kv_heads: Some(2),
        ..tiny_model_config(false)
    }
}

/// [`tiny_model_config`] with MLA attention.
fn mla() -> OxidizrModelConfig {
    OxidizrModelConfig {
        kv_latent_dim: Some(8),
        q_latent_dim: Some(8),
        d_rope: Some(2),
        ..tiny_model_config(false)
    }
}

/// GQA layer 0, Mamba2 layer 1 and Gated DeltaNet layer 2.
fn hybrid() -> OxidizrModelConfig {
    OxidizrModelConfig {
        num_layers: 3,
        kv_heads: Some(2),
        mamba_layers: Some(vec![1]),
        mamba2_num_heads: Some(4),
        mamba2_head_dim: Some(8),
        mamba2_state_size: Some(4),
        mamba2_chunk_size: Some(4),
        mamba2_n_groups: Some(1),
        mamba2_conv_kernel: Some(4),
        mamba2_expand: Some(2),
        gdn_layers: Some(vec![2]),
        gdn: Some(GdnSpec {
            state_size: 4,
            key_heads: 2,
            value_heads: 4,
            conv_kernel: 3,
            chunk_size: 4,
            rms_eps: 1e-6,
        }),
        ..tiny_model_config(false)
    }
}

/// A seeded model of `model`.
fn model_of(model: OxidizrModelConfig) -> LitGPT<CpuRuntime> {
    let mut cfg = tiny_config(Path::new("unused"), 1);
    cfg.model = model;
    build_model(&cfg)
}

/// One batch of `rows`, each `SEQ + 1` tokens, under the document mask.
fn masked_batch(rows: &[Vec<u32>]) -> Batch<CpuRuntime> {
    LitDataLoader::<CpuRuntime>::new(rows.concat(), rows.len(), SEQ, CpuDevice::new())
        .with_document_mask(SEP)
        .next_batch()
        .expect("masked batch")
}

/// Host logits `[rows * SEQ * vocab]` of `batch`, with its segments.
fn logits(model: &LitGPT<CpuRuntime>, batch: &Batch<CpuRuntime>) -> Vec<f32> {
    let client = CpuClient::new(CpuDevice::new());
    let out = model
        .forward_with(
            &client,
            &batch.inputs,
            &ActivationCheckpointing::None,
            batch.segments.as_ref(),
        )
        .expect("forward");
    out.logits
        .tensor()
        .contiguous()
        .expect("contiguous")
        .to_vec()
}

/// Cross-entropy summed over `positions` of `row`, in f64 from host logits.
fn ce_sum(
    logits: &[f32],
    targets: &[i64],
    vocab: usize,
    row: usize,
    positions: Range<usize>,
) -> f64 {
    positions
        .map(|p| {
            let at = (row * SEQ + p) * vocab;
            let z = &logits[at..at + vocab];
            let max = z
                .iter()
                .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
            let lse = max
                + z.iter()
                    .map(|&v| (f64::from(v) - max).exp())
                    .sum::<f64>()
                    .ln();
            lse - f64::from(z[targets[row * SEQ + p] as usize])
        })
        .sum()
}

/// The logits of `positions` of `row`.
fn row_logits(logits: &[f32], vocab: usize, row: usize, positions: Range<usize>) -> Vec<u32> {
    let span = (row * SEQ + positions.start) * vocab..(row * SEQ + positions.end) * vocab;
    logits[span].iter().map(|v| v.to_bits()).collect()
}

/// Document A is 4 tokens with its separator, document B the 4 tokens after it.
fn packed_matches_separate_rows(model: OxidizrModelConfig, label: &str, rel: f64) {
    let vocab = model.vocab_size;
    let net = model_of(model);
    let packed = vec![1, 2, 3, SEP, 7, 8, 9, 10, 11];
    let row_a = vec![1, 2, 3, SEP, 12, 13, 14, 15, 16];
    let row_b = vec![7, 8, 9, 10, 11, 12, 13, 14, 15];
    let batch = masked_batch(&[packed, row_a, row_b]);
    let mask: Vec<f32> = batch
        .loss_mask
        .as_ref()
        .expect("loss mask")
        .contiguous()
        .expect("contiguous")
        .to_vec();
    assert_eq!(
        &mask[..SEQ],
        &[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "{label}"
    );

    let z = logits(&net, &batch);
    let targets: Vec<i64> = batch.targets.contiguous().expect("contiguous").to_vec();
    // The separator input at position 3 is unscored: its target opens document B.
    let packed_loss = ce_sum(&z, &targets, vocab, 0, 0..3) + ce_sum(&z, &targets, vocab, 0, 4..8);
    let separate = ce_sum(&z, &targets, vocab, 1, 0..3) + ce_sum(&z, &targets, vocab, 2, 0..4);
    assert!(
        (packed_loss - separate).abs() <= rel * separate.abs(),
        "{label}: packed loss {packed_loss} vs separate rows {separate}"
    );
}

#[test]
fn a_packed_row_has_the_loss_of_its_documents_in_separate_rows() {
    packed_matches_separate_rows(gqa(), "gqa", 1e-6);
    packed_matches_separate_rows(mla(), "mla", 1e-6);
}

#[test]
fn a_recurrent_packed_row_has_the_loss_of_its_documents_in_separate_rows() {
    packed_matches_separate_rows(hybrid(), "hybrid", 1e-4);
}

fn document_b_ignores_document_a(model: OxidizrModelConfig, label: &str) {
    let vocab = model.vocab_size;
    let net = model_of(model);
    let batch = masked_batch(&[
        vec![1, 2, 3, SEP, 7, 8, 9, 10, 11],
        vec![4, 6, 2, SEP, 7, 8, 9, 10, 11],
    ]);
    let z = logits(&net, &batch);
    assert_eq!(
        row_logits(&z, vocab, 0, 4..SEQ),
        row_logits(&z, vocab, 1, 4..SEQ),
        "{label}: document B saw document A"
    );
    assert_ne!(
        row_logits(&z, vocab, 0, 0..4),
        row_logits(&z, vocab, 1, 0..4),
        "{label}: the changed document A kept its logits"
    );
}

#[test]
fn changing_document_a_leaves_document_b_bit_identical() {
    document_b_ignores_document_a(gqa(), "gqa");
    document_b_ignores_document_a(mla(), "mla");
}

#[test]
fn changing_document_a_leaves_a_recurrent_document_b_unchanged() {
    let vocab = hybrid().vocab_size;
    let net = model_of(hybrid());
    let batch = masked_batch(&[
        vec![1, 2, 3, SEP, 7, 8, 9, 10, 11],
        vec![4, 6, 2, SEP, 7, 8, 9, 10, 11],
    ]);
    let z = logits(&net, &batch);
    let at = |row: usize, positions: Range<usize>| -> Vec<f32> {
        let span = (row * SEQ + positions.start) * vocab..(row * SEQ + positions.end) * vocab;
        z[span].to_vec()
    };
    let drift = at(0, 4..SEQ)
        .iter()
        .zip(at(1, 4..SEQ))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(drift <= 1e-5, "document B saw document A: drift {drift}");
    assert_ne!(
        row_logits(&z, vocab, 0, 0..4),
        row_logits(&z, vocab, 1, 0..4),
        "the changed document A kept its logits"
    );
}

/// Trains `steps` steps from the stream start and saves the step set when `save`.
fn train_fresh(cfg: &LitConfig, steps: usize, save: bool) -> Bits {
    let model = build_model(cfg);
    let mut parts = train_parts(&model);
    let mut loader = masked_data_loader(cfg);
    let layout = loader.row_layout().expect("row layout");
    {
        let mut t = trainer(cfg, &mut parts, layout, 0, DataPosition::default());
        for _ in 0..steps {
            t.training_step(&model, &mut loader).expect("training step");
        }
        if save {
            t.save_checkpoint(&model, steps).expect("save the step set");
        }
    }
    bits_by_name(&parts)
}

/// Resumes the latest set in `cfg`'s checkpoint directory, as `commands::train` does,
/// then trains to `STEPS`.
fn resume(cfg: &LitConfig) -> Bits {
    let (weights, meta, step) = find_latest_checkpoint(&cfg.trainer.checkpoint_dir)
        .expect("scan")
        .expect("a committed step set");
    let current = cfg.trainer.document_mask.and_then(|m| m.record());
    check_document_mask_resume(Some(&meta), step, current, true).expect("the same mask");
    let mut model = build_model(cfg);
    load_checkpoint_into_model(&mut model, &weights).expect("load weights");
    let plan = plan_optimizer_resume(Some(&weights), Some(&meta), step).expect("plan");
    let position = CheckpointMeta::load(&meta)
        .expect("meta")
        .data_position
        .expect("data position");
    let inherited = inherited_weight_decay(Some(&meta), step).expect("inherited rule");
    let decay = resolve_weight_decay(&cfg.trainer, inherited.as_ref(), true).expect("rule");
    let mut parts = train_parts(&model);
    let mut loader = masked_data_loader(cfg);
    let layout = loader.row_layout().expect("row layout");
    loader.seek(position).expect("seek");
    {
        let mut t = trainer_with_decay(cfg, &mut parts, layout, step, position, decay);
        t.apply_optimizer_resume(&plan).expect("optimizer state");
        for _ in step..STEPS {
            t.training_step(&model, &mut loader).expect("training step");
        }
    }
    bits_by_name(&parts)
}

#[test]
fn a_masked_run_resumes_bit_identical_and_records_its_mask() {
    let dir = scratch_dir("document-mask-resume");
    let cfg = document_masked(tiny_config(&dir, STEPS));
    let reference = train_fresh(&cfg, STEPS, false);
    assert_eq!(
        reference,
        train_fresh(&cfg, STEPS, false),
        "the step is not deterministic"
    );
    train_fresh(&cfg, HALF, true);

    let (_, meta, step) = find_latest_checkpoint(&cfg.trainer.checkpoint_dir)
        .expect("scan")
        .expect("a committed step set");
    assert_eq!(step, HALF);
    let recorded = CheckpointMeta::load(&meta).expect("meta").document_mask;
    assert_eq!(recorded, Some(DocumentMaskRecord { separator_id: SEP }));
    let dropped = check_document_mask_resume(Some(&meta), step, None, true)
        .expect_err("a headless resume without the mask");
    assert!(dropped.to_string().contains("'off'"), "{dropped}");

    assert_eq!(reference, resume(&cfg), "the resumed run differs");
    let mut unmasked = cfg.clone();
    unmasked.trainer.document_mask = None;
    assert_ne!(
        reference,
        train_fresh(&unmasked, STEPS, false),
        "the mask changed nothing"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn recompute_under_the_mask_trains_bit_identical() {
    let dir = scratch_dir("document-mask-recompute");
    let mut cfg = document_masked(tiny_config(&dir, STEPS));
    cfg.model = gqa();
    let plain = train_fresh(&cfg, 2, false);
    cfg.trainer.activation_checkpointing = ActivationCheckpointing::All;
    assert_eq!(
        plain,
        train_fresh(&cfg, 2, false),
        "recompute changed the step"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn a_recurrent_model_trains_under_the_mask_with_and_without_recompute() {
    let dir = scratch_dir("document-mask-recurrent");
    let mut cfg = document_masked(tiny_config(&dir, STEPS));
    cfg.model = hybrid();
    let plain = train_fresh(&cfg, 2, false);
    let mut unmasked = cfg.clone();
    unmasked.trainer.document_mask = None;
    assert_ne!(
        plain,
        train_fresh(&unmasked, 2, false),
        "the mask changed nothing"
    );
    cfg.trainer.activation_checkpointing = ActivationCheckpointing::All;
    assert_eq!(
        plain,
        train_fresh(&cfg, 2, false),
        "recompute changed the step"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn the_yaml_loader_accepts_a_mask_on_a_recurrent_model() {
    let dir = scratch_dir("document-mask-yaml-recurrent");
    let mut cfg = document_masked(tiny_config(&dir, STEPS));
    cfg.model = hybrid();
    let path = dir.join("hybrid.yaml");
    write_config_yaml(&path, &cfg);
    LitConfig::from_yaml(&path).expect("Mamba and GDN layers take the mask");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn the_yaml_loader_refuses_a_mask_with_a_sliding_window() {
    let dir = scratch_dir("document-mask-yaml");
    let mut cfg = document_masked(tiny_config(&dir, STEPS));
    cfg.model.sliding_window = Some(4);
    let path = dir.join("window.yaml");
    write_config_yaml(&path, &cfg);
    let e = LitConfig::from_yaml(&path)
        .expect_err("window and mask")
        .to_string();
    assert!(
        e.contains("model.sliding_window 4") && e.contains("trainer.document_mask"),
        "{e}"
    );
    let _ = fs::remove_dir_all(dir);
}
