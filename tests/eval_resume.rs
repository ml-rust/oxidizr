//! A run with evals resumed from a checkpoint continues exactly where it stopped:
//! the weights, `evals.json`, and the early-stop counters.

mod common;

use common::{
    bits_by_name, build_model, eval_loaders, holdout_eval, scratch_dir, tiny_config, train_parts,
    trainer_with_decay,
};
use oxidizr::checkpoint_set::{plan_optimizer_resume, CheckpointLabel};
use oxidizr::config::{find_latest_checkpoint, CheckpointMeta, LitConfig};
use oxidizr::data::DataPosition;
use oxidizr::eval::{read_eval_resume, EarlyStoppingConfig};
use oxidizr::fit::{
    CheckpointNote, EvalReport, FitStart, FitSummary, JsonSink, StepReport, StepSink,
};
use oxidizr::loss_scale::read_loss_scale_state;
use oxidizr::metrics::GpuMemory;
use oxidizr::model::load_checkpoint_into_model;
use oxidizr::prelude::{Error, Result};
use oxidizr::run_record::EVALS_FILE;
use oxidizr::weight_decay::{inherited_weight_decay, resolve_weight_decay};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const STEPS: usize = 6;
const HALF: usize = STEPS / 2;
const HOLDOUT: usize = 40;

type Bits = BTreeMap<String, Vec<u32>>;

fn no_gpu() -> Option<GpuMemory> {
    None
}

/// A sink that stops the run right after the interval checkpoint of step `at` commits.
struct StopAfterSave<S> {
    inner: S,
    at: usize,
}

impl<S: StepSink> StepSink for StopAfterSave<S> {
    fn on_start(&mut self, start: &FitStart) -> Result<()> {
        self.inner.on_start(start)
    }
    fn on_step(&mut self, report: &StepReport) -> Result<()> {
        self.inner.on_step(report)
    }
    fn on_eval(&mut self, step: usize, report: &EvalReport) -> Result<()> {
        self.inner.on_eval(step, report)
    }
    fn on_checkpoint(&mut self, note: &CheckpointNote) -> Result<()> {
        self.inner.on_checkpoint(note)?;
        if *note == (CheckpointNote::Saved { step: self.at }) {
            return Err(Error::TrainingError {
                reason: format!("stopped after the step {} save", self.at),
            });
        }
        Ok(())
    }
    fn on_error(&mut self, step: usize, error: &Error) {
        self.inner.on_error(step, error)
    }
    fn on_finish(&mut self, summary: &FitSummary) -> Result<()> {
        self.inner.on_finish(summary)
    }
}

/// `tiny_config` in `dir` with an eval after every step and a save every `save` steps.
fn config(dir: &Path, save: usize, early: Option<EarlyStoppingConfig>) -> LitConfig {
    let mut cfg = tiny_config(dir, STEPS);
    cfg.trainer.save_interval = save;
    let mut eval = holdout_eval(HOLDOUT, 1, 2);
    eval.early_stopping = early;
    cfg.trainer.eval = Some(eval);
    cfg
}

struct Session {
    bits: Bits,
    final_step: usize,
    stop_reason: Option<String>,
}

/// Fits `cfg` from the latest checkpoint in its directory, or from step 0 when
/// `resume` is false, as `commands::train` sets the run up.
fn session(cfg: &LitConfig, resume: bool, sink: &mut dyn StepSink) -> (Session, Result<()>) {
    let mut model = build_model(cfg);
    let (start_step, found) = if resume {
        let (weights, meta, step) = find_latest_checkpoint(&cfg.trainer.checkpoint_dir)
            .expect("scan")
            .expect("a committed step set");
        load_checkpoint_into_model(&mut model, &weights).expect("load weights");
        (step, Some((weights, meta)))
    } else {
        (0, None)
    };
    let weights = found.as_ref().map(|(w, _)| w.as_str());
    let meta = found.as_ref().map(|(_, m)| m.as_str());
    let plan = plan_optimizer_resume(weights, meta, start_step).expect("optimizer plan");
    let inherited = inherited_weight_decay(meta, start_step).expect("inherited rule");
    let decay = resolve_weight_decay(&cfg.trainer, inherited.as_ref(), true).expect("rule");
    let scale = read_loss_scale_state(meta, start_step).expect("loss scale");

    let (mut train, evaluator) = eval_loaders(cfg);
    let resume_state =
        read_eval_resume(meta, start_step, evaluator.holdout_rows()).expect("eval state");
    let layout = train.row_layout().expect("row layout");
    let position = meta
        .map(|m| CheckpointMeta::load(m).expect("meta"))
        .and_then(|m| m.data_position)
        .unwrap_or_default();
    train.seek(position).expect("seek");

    let mut parts = train_parts(&model);
    let (final_step, stop_reason, result) = {
        let mut t = trainer_with_decay(cfg, &mut parts, layout, start_step, position, decay);
        t.apply_optimizer_resume(&plan).expect("optimizer state");
        t.restore_loss_scale(scale).expect("loss scale");
        t.attach_eval(evaluator.resumed(resume_state));
        let result = t.fit_with_sink(&model, &mut train, sink);
        (
            t.global_step(),
            t.early_stop_reason().map(str::to_string),
            result,
        )
    };
    let session = Session {
        bits: bits_by_name(&parts),
        final_step,
        stop_reason,
    };
    (session, result)
}

/// A headless sink whose output nobody reads.
fn quiet() -> JsonSink<std::io::Sink, std::io::Sink> {
    JsonSink::new(std::io::sink(), std::io::sink(), no_gpu)
}

fn final_meta(dir: &Path) -> CheckpointMeta {
    let path = dir.join(CheckpointLabel::Final.meta_file_name());
    CheckpointMeta::load(&path.to_string_lossy()).expect("final meta")
}

#[test]
fn a_run_with_evals_resumes_bit_identical_including_evals_json() {
    let dir = scratch_dir("eval-exact-resume");
    let (reference_dir, stopped_dir) = (dir.join("reference"), dir.join("stopped"));
    let reference_cfg = config(&reference_dir, HALF, None);
    let stopped_cfg = config(&stopped_dir, HALF, None);

    let (reference, result) = session(&reference_cfg, false, &mut quiet());
    result.expect("reference run");
    assert_eq!(reference.final_step, STEPS);

    let mut stop = StopAfterSave {
        inner: quiet(),
        at: HALF,
    };
    let (stopped, result) = session(&stopped_cfg, false, &mut stop);
    assert!(result.is_err(), "the stop sink ends the first session");
    assert_eq!(stopped.final_step, HALF);
    let (resumed, result) = session(&stopped_cfg, true, &mut quiet());
    result.expect("resumed run");

    assert_eq!(resumed.final_step, STEPS);
    assert_eq!(
        reference.bits, resumed.bits,
        "weights differ after the resume"
    );
    let reference_evals = fs::read_to_string(reference_dir.join(EVALS_FILE)).expect("evals");
    let resumed_evals = fs::read_to_string(stopped_dir.join(EVALS_FILE)).expect("evals");
    assert_eq!(reference_evals.lines().count(), STEPS);
    assert_eq!(reference_evals, resumed_evals);
    let (a, b) = (final_meta(&reference_dir), final_meta(&stopped_dir));
    assert_eq!(a.best_eval_loss, b.best_eval_loss);
    assert_eq!(a.evals_since_best, b.evals_since_best);
    assert_eq!(a.holdout_rows, Some(HOLDOUT));
    assert_eq!(b.holdout_rows, Some(HOLDOUT));
    let _ = fs::remove_dir_all(dir);
}

/// No eval after the first beats it by 1e9: evals 2 and 3 spend a patience of 2.
fn never_improves() -> Option<EarlyStoppingConfig> {
    Some(EarlyStoppingConfig {
        patience: 2,
        min_delta: 1e9,
    })
}

#[test]
fn a_resumed_run_continues_the_early_stop_counters() {
    let dir = scratch_dir("eval-resume-counters");
    let cfg = config(&dir, 2, never_improves());
    let mut stop = StopAfterSave {
        inner: quiet(),
        at: 2,
    };
    let (_, result) = session(&cfg, false, &mut stop);
    assert!(result.is_err());
    let saved = dir.join(CheckpointLabel::Step(2).meta_file_name());
    let meta = CheckpointMeta::load(&saved.to_string_lossy()).expect("step 2 meta");
    assert_eq!(meta.evals_since_best, Some(1));
    assert!(meta.best_eval_loss.is_some());

    // Fresh counters would stop at step 5. The restored ones stop at step 3.
    let (resumed, result) = session(&cfg, true, &mut quiet());
    result.expect("resumed run");
    assert_eq!(resumed.final_step, 3);
    assert!(resumed.stop_reason.is_some());
    assert_eq!(final_meta(&dir).evals_since_best, Some(2));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn a_resume_after_the_stopping_eval_runs_no_step() {
    let dir = scratch_dir("eval-resume-spent");
    let cfg = config(&dir, 3, never_improves());
    let mut stop = StopAfterSave {
        inner: quiet(),
        at: 3,
    };
    let (stopped, result) = session(&cfg, false, &mut stop);
    assert!(result.is_err());
    assert!(stopped.stop_reason.is_some(), "eval 3 spends the patience");

    let mut out = Vec::new();
    let mut sink = JsonSink::new(&mut out, std::io::sink(), no_gpu);
    let (resumed, result) = session(&cfg, true, &mut sink);
    result.expect("resumed run");
    assert_eq!(resumed.final_step, 3);
    assert_eq!(resumed.bits, stopped.bits);
    let stdout = String::from_utf8(out).expect("utf-8");
    assert!(!stdout.contains(r#""step": 4"#), "{stdout}");
    assert!(stdout.contains(r#""stopped_early": true"#), "{stdout}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn a_resume_with_another_holdout_is_an_error() {
    let dir = scratch_dir("eval-resume-mismatch");
    let cfg = config(&dir, HALF, None);
    let mut stop = StopAfterSave {
        inner: quiet(),
        at: HALF,
    };
    let (_, result) = session(&cfg, false, &mut stop);
    assert!(result.is_err());
    let meta = dir.join(CheckpointLabel::Step(HALF).meta_file_name());
    let meta = meta.to_string_lossy();

    for holdout in [0, HOLDOUT + 1] {
        let err = read_eval_resume(Some(&meta), HALF, holdout).expect_err("another holdout");
        let text = err.to_string();
        assert!(
            text.contains(&format!("records {HOLDOUT} held-out rows")),
            "{text}"
        );
        assert!(text.contains(&format!("holds out {holdout}")), "{text}");
    }
    let resume = read_eval_resume(Some(&meta), HALF, HOLDOUT).expect("same holdout");
    assert!(resume.best_eval_loss.is_some());
    // A run at step 0 starts fresh, whatever the meta says.
    let fresh = read_eval_resume(Some(&meta), 0, 0).expect("step 0");
    assert_eq!(fresh, Default::default());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn a_run_without_eval_writes_no_eval_meta_keys() {
    let dir = scratch_dir("eval-absent-meta");
    let mut cfg = tiny_config(&dir, 2);
    cfg.trainer.save_interval = 1;
    let model = build_model(&cfg);
    let mut parts = train_parts(&model);
    let mut loader = common::data_loader(&cfg);
    let layout = loader.row_layout().expect("layout");
    {
        let mut t = common::trainer(&cfg, &mut parts, layout, 0, DataPosition::default());
        t.fit_with_sink(&model, &mut loader, &mut quiet())
            .expect("fit");
        assert!(t.eval_outcome().is_none());
    }
    for label in [CheckpointLabel::Step(1), CheckpointLabel::Final] {
        let text = fs::read_to_string(dir.join(label.meta_file_name())).expect("meta");
        for key in ["holdout_rows", "best_eval_loss", "evals_since_best"] {
            assert!(!text.contains(key), "{key} in {text}");
        }
    }
    assert!(!dir.join(EVALS_FILE).exists());
    assert!(!dir.join(CheckpointLabel::Best.meta_file_name()).exists());
    let _ = fs::remove_dir_all(dir);
}
