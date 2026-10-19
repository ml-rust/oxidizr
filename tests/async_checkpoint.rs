//! Background checkpoint writes: the same files as synchronous writes, an
//! exact resume, notes that fire once a set is committed, and write errors
//! that stop the run.

mod common;

use boostr::format::SafeTensors;
use common::{
    assert_same_checkpoints, build_model, fit_session, holdout_eval, scratch_dir, tiny_config,
    tiny_lora, train_parts, StopAtStep,
};
use oxidizr::checkpoint_set::CheckpointLabel;
use oxidizr::config::LitConfig;
use oxidizr::fit::{
    CheckpointNote, EvalReport, FitStart, FitSummary, JsonSink, StepReport, StepSink,
};
use oxidizr::metrics::GpuMemory;
use oxidizr::model::{stage_checkpoint_tensors, StageScope};
use oxidizr::prelude::{Error, Result};
use std::fs;
use std::path::{Path, PathBuf};

const STEPS: usize = 6;
const HALF: usize = STEPS / 2;

fn no_gpu() -> Option<GpuMemory> {
    None
}

/// A headless sink whose output nobody reads.
fn quiet() -> JsonSink<std::io::Sink, std::io::Sink> {
    JsonSink::new(std::io::sink(), std::io::sink(), no_gpu)
}

/// `tiny_config` in `dir`, saving every `save` steps, with or without the async writer.
fn config(dir: &Path, save: usize, async_checkpoint: bool) -> LitConfig {
    let mut cfg = tiny_config(dir, STEPS);
    cfg.trainer.save_interval = save;
    cfg.trainer.async_checkpoint = async_checkpoint;
    cfg
}

/// [`config`] with LoRA, an eval after every step, and two interval sets kept.
fn lora_eval_config(dir: &Path, async_checkpoint: bool) -> LitConfig {
    let mut cfg = config(dir, 2, async_checkpoint);
    cfg.model.lora = Some(tiny_lora());
    cfg.trainer.eval = Some(holdout_eval(40, 1, 2));
    cfg.trainer.max_checkpoints = 2;
    cfg
}

/// The frozen tensors of a fresh model of `cfg` match those in the weights file at `path`.
///
/// The fresh copy stages without the trainer's frozen cache.
fn assert_frozen_tensors_match(cfg: &LitConfig, path: &Path) {
    let model = build_model(cfg);
    let parts = train_parts(&model);
    let frozen = stage_checkpoint_tensors("fresh", &model, &parts.params, StageScope::Frozen)
        .expect("stage frozen tensors");
    assert!(!frozen.is_empty(), "the LoRA model has no frozen tensor");
    let mut file = SafeTensors::open(path).expect("open weights");
    for (name, tensor) in &frozen {
        let stored = file.read_tensor_bytes(name).expect("read frozen tensor");
        assert!(
            stored == tensor.to_bytes().expect("tensor bytes"),
            "frozen tensor {name} differs in {}",
            path.display()
        );
    }
}

#[test]
fn async_and_sync_runs_write_byte_equal_sets() {
    let dir = scratch_dir("async-equal");
    let (sync_dir, async_dir) = (dir.join("sync"), dir.join("async"));
    let sync_cfg = lora_eval_config(&sync_dir, false);
    let async_cfg = lora_eval_config(&async_dir, true);

    let (sync_run, result) = fit_session(&sync_cfg, false, &mut quiet());
    result.expect("sync run");
    let (async_run, result) = fit_session(&async_cfg, false, &mut quiet());
    result.expect("async run");
    assert_eq!(
        sync_run.bits, async_run.bits,
        "the runs trained differently"
    );

    let checked = assert_same_checkpoints(&sync_dir, &async_dir);
    let prefix = sync_cfg.model.checkpoint_prefix();
    for label in [
        CheckpointLabel::Step(4),
        CheckpointLabel::Step(6),
        CheckpointLabel::Best,
        CheckpointLabel::Final,
    ] {
        for name in [
            label.weights_file_name(&prefix),
            label.optimizer_file_name(&prefix),
            label.meta_file_name(),
        ] {
            assert!(checked.contains(&name), "{name} missing from {checked:?}");
        }
    }
    let pruned = CheckpointLabel::Step(2).weights_file_name(&prefix);
    assert!(!checked.contains(&pruned), "max_checkpoints kept {pruned}");
    for label in [CheckpointLabel::Step(6), CheckpointLabel::Best] {
        let weights = async_dir.join(label.weights_file_name(&prefix));
        assert_frozen_tensors_match(&async_cfg, &weights);
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn an_async_step_set_resumes_bit_identical() {
    let dir = scratch_dir("async-resume");
    let reference_cfg = config(&dir.join("reference"), HALF, false);
    let stopped_cfg = config(&dir.join("stopped"), HALF, true);

    let (reference, result) = fit_session(&reference_cfg, false, &mut quiet());
    result.expect("reference run");
    let mut stop = StopAtStep {
        inner: quiet(),
        at: HALF + 1,
    };
    let (_, result) = fit_session(&stopped_cfg, false, &mut stop);
    assert!(result.is_err(), "the stop sink ends the first session");

    let (resumed, result) = fit_session(&stopped_cfg, true, &mut quiet());
    result.expect("resumed run");
    assert_eq!(resumed.final_step, STEPS);
    assert_eq!(reference.bits, resumed.bits, "the resume diverged");
    let _ = fs::remove_dir_all(dir);
}

/// Records every checkpoint note, and checks that its meta is on disk when it fires.
struct NoteRecorder {
    dir: PathBuf,
    notes: Vec<CheckpointNote>,
}

impl StepSink for NoteRecorder {
    fn on_start(&mut self, _start: &FitStart) -> Result<()> {
        Ok(())
    }
    fn on_step(&mut self, _report: &StepReport) -> Result<()> {
        Ok(())
    }
    fn on_eval(&mut self, _step: usize, _report: &EvalReport) -> Result<()> {
        Ok(())
    }
    fn on_checkpoint(&mut self, note: &CheckpointNote) -> Result<()> {
        let label = match note {
            CheckpointNote::Saved { step } => Some(CheckpointLabel::Step(*step)),
            CheckpointNote::Best { .. } => Some(CheckpointLabel::Best),
            CheckpointNote::Final { .. } => None,
        };
        if let Some(label) = label {
            let meta = self.dir.join(label.meta_file_name());
            assert!(
                meta.exists(),
                "{note:?} fired before {} existed",
                meta.display()
            );
        }
        self.notes.push(note.clone());
        Ok(())
    }
    fn on_error(&mut self, _step: usize, _error: &Error) {}
    fn on_finish(&mut self, _summary: &FitSummary) -> Result<()> {
        Ok(())
    }
}

#[test]
fn notes_fire_after_the_commit_and_before_final() {
    for async_checkpoint in [false, true] {
        let dir = scratch_dir(&format!("async-notes-{async_checkpoint}"));
        let mut cfg = config(&dir, 1, async_checkpoint);
        cfg.trainer.eval = Some(holdout_eval(40, 1, 2));
        let mut sink = NoteRecorder {
            dir: dir.clone(),
            notes: Vec::new(),
        };
        let (_, result) = fit_session(&cfg, false, &mut sink);
        result.expect("fit");

        let saved: Vec<usize> = sink
            .notes
            .iter()
            .filter_map(|note| match note {
                CheckpointNote::Saved { step } => Some(*step),
                _ => None,
            })
            .collect();
        assert_eq!(saved, (1..=STEPS).collect::<Vec<_>>(), "{:?}", sink.notes);
        assert!(
            sink.notes.contains(&CheckpointNote::Best { step: 1 }),
            "the first eval is always a best: {:?}",
            sink.notes
        );
        let last = sink.notes.last().expect("a note");
        assert!(
            matches!(last, CheckpointNote::Final { .. }),
            "a note came after Final: {:?}",
            sink.notes
        );
        let finals = sink
            .notes
            .iter()
            .filter(|note| matches!(note, CheckpointNote::Final { .. }))
            .count();
        assert_eq!(finals, 1);
        let _ = fs::remove_dir_all(dir);
    }
}

#[test]
fn a_checkpoint_dir_under_a_file_fails_the_run_naming_step_and_label() {
    for async_checkpoint in [false, true] {
        let dir = scratch_dir(&format!("async-unwritable-{async_checkpoint}"));
        let blocker = dir.join("blocker");
        fs::write(&blocker, b"a regular file").expect("write blocker");
        let cfg = config(&blocker.join("ckpt"), 2, async_checkpoint);

        let (session, result) = fit_session(&cfg, false, &mut quiet());
        let err = result
            .expect_err("no checkpoint can be written")
            .to_string();
        assert!(err.contains("checkpoint `step_2` of step 2"), "{err}");
        assert!(err.contains("was not written"), "{err}");
        assert!(err.contains("is writable"), "{err}");
        assert!(
            session.final_step < STEPS,
            "the run went on after the write error: step {}",
            session.final_step
        );
        let _ = fs::remove_dir_all(dir);
    }
}
