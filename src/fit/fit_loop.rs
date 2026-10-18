//! The one fit loop of `LitTrainer`: steps, logs, checkpoints, the final save.

use super::json_sink::JsonSink;
use super::sink::{CheckpointNote, FitStart, FitSummary, NullSink, StepReport, StepSink};
use super::throughput::Throughput;
use super::tui_sink::TuiSink;
use crate::checkpoint_config::{checkpoint_config_exists, write_checkpoint_config};
use crate::checkpoint_sidecars::copy_checkpoint_sidecars;
use crate::data::DataLoader;
use crate::metrics::{mfu, model_flops_per_token, ScalarLog};
use crate::model::LitGPT;
use crate::prelude::{DType, Result};
use crate::run_record::StepRecord;
use crate::trainer::LitTrainer;
use boostr::ops::{FusedOptimizerOps, TensorOps};
use boostr::runtime::{Runtime, RuntimeClient};
use std::path::Path;

impl<'a, R> LitTrainer<'a, R>
where
    R: Runtime<DType = DType>,
    R::Client: RuntimeClient<R>
        + TensorOps<R>
        + FusedOptimizerOps<R>
        + boostr::ScalarOps<R>
        + boostr::BinaryOps<R>
        + boostr::CompareOps<R>
        + boostr::ConvOps<R>
        + boostr::FlashAttentionOps<R>
        + boostr::IndexingOps<R>
        + boostr::NormalizationOps<R>
        + boostr::QuantMatmulOps<R>
        + boostr::RoPEOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::DequantOps<R>
        + boostr::GatedDeltaNetOps<R>,
{
    /// Trains the remaining steps with the output mode the run config chose.
    ///
    /// Headless output is JSON lines on stdout. Interactive output is a progress bar.
    /// Data-parallel ranks other than 0 report to a [`NullSink`].
    pub fn fit<D: DataLoader<R>>(&mut self, model: &LitGPT<R>, data_loader: &mut D) -> Result<()> {
        if self.quiet() {
            self.fit_with_sink(model, data_loader, &mut NullSink)
        } else if self.headless {
            self.fit_with_sink(model, data_loader, &mut JsonSink::stdio())
        } else {
            self.fit_with_sink(model, data_loader, &mut TuiSink::new())
        }
    }

    /// Trains the remaining steps and reports every event to `sink`.
    ///
    /// The loop writes the same `steps.json` lines and checkpoints for every sink.
    /// A log step is a multiple of `log_interval`, or the first step of this session.
    /// `log_interval: 0` logs the first step of the session only.
    /// Under the `tensorboard` feature each log step and eval also goes to a
    /// new event file beside `steps.json`.
    /// A step error goes to [`StepSink::on_error`] and returns at once.
    ///
    /// With an attached eval, an eval runs after every multiple of
    /// `trainer.eval.interval` and after the last step. When early stopping
    /// fires, the loop stops at that step boundary and writes the final checkpoint.
    ///
    /// # Data parallelism
    /// In a world above 1 the loop first copies rank 0's model tensors to
    /// every rank. Only rank 0 writes `config.json`, `steps.json`, evals and
    /// checkpoints. Only rank 0 evaluates. After every step, rank 0's
    /// early-stop decision reaches every rank, so all ranks stop together.
    /// Every rank waits at a barrier after the final save.
    /// An error on this rank calls [`abort_collectives`](Self::abort_collectives)
    /// before it returns, so peers do not wait in a collective.
    pub fn fit_with_sink<D: DataLoader<R>>(
        &mut self,
        model: &LitGPT<R>,
        data_loader: &mut D,
        sink: &mut dyn StepSink,
    ) -> Result<()> {
        let result = self.run_fit(model, data_loader, sink);
        if result.is_err() {
            self.abort_collectives();
        }
        result
    }

    /// The body of [`fit_with_sink`](Self::fit_with_sink).
    fn run_fit<D: DataLoader<R>>(
        &mut self,
        model: &LitGPT<R>,
        data_loader: &mut D,
        sink: &mut dyn StepSink,
    ) -> Result<()> {
        // blazr reads config.json to serve a checkpoint directly.
        self.save_blazr_config_if_needed();
        self.sync_initial_parameters(model)?;
        // A resumed rank 0 can start stopped. Every rank must agree before step one.
        self.sync_early_stop()?;

        if !self.quiet() {
            let run_dir = Path::new(&self.steps_json_path)
                .parent()
                .unwrap_or(Path::new("."));
            self.metrics.scalars = ScalarLog::open(run_dir);
        }
        // FLOPs per token and the peak of one device, when MFU is configured.
        let mfu_basis = match self.metrics.peak_tflops() {
            Some(peak) => Some((model_flops_per_token(model, &self.model_config)?, peak)),
            None => None,
        };

        let total_steps = self.lr_plan.total_steps();
        // Samples of every rank. `batch_size` counts one rank's rows.
        let mut clock = Throughput::start(
            self.batch_size * self.config.gradient_accumulation * self.world_size(),
        );
        sink.on_start(&FitStart {
            start_step: self.start_step,
            total_steps,
            warmup_steps: self.lr_plan.warmup_steps(),
            base_lr: self.lr_plan.base_lr(),
        })?;
        clock.restart_step_clock();

        let mut last = None;
        for index in self.start_step..total_steps {
            // Early stopping ends the run at a step boundary, before the next step draws data.
            if self.early_stop_reason().is_some() {
                break;
            }
            let (loss, grad_norm) = match self.training_step(model, data_loader) {
                Ok(result) => result,
                Err(e) => {
                    sink.on_error(index + 1, &e);
                    return Err(e);
                }
            };
            clock.end_step();
            let completed = index + 1;
            let report = self.step_report(index, (loss, grad_norm), &clock, mfu_basis);
            self.metrics.last_rates = Some((report.tokens_per_sec, report.mfu));
            if report.logged {
                self.record_step(&report);
                self.metrics.scalars.step(&report);
            }
            sink.on_step(&report)?;
            last = Some((loss, grad_norm));

            // Before the interval save, so its meta holds this eval's counters.
            self.eval_if_due(model, completed, total_steps, sink)?;
            if self.is_save_step(completed) {
                self.save_checkpoint(model, completed)?;
                sink.on_checkpoint(&CheckpointNote::Saved { step: completed })?;
            }
            // Every rank learns rank 0's stop decision before the next step draws data.
            self.sync_early_stop()?;
        }

        let path = format!(
            "{}/{}_final.safetensors",
            self.config.checkpoint_dir,
            self.model_config.checkpoint_prefix()
        );
        sink.on_checkpoint(&CheckpointNote::Final { path })?;
        self.save_final_checkpoint(model)?;
        sink.on_finish(&FitSummary {
            final_step: self.global_step(),
            last_loss: last.map(|(loss, _)| loss),
            last_grad_norm: last.map(|(_, grad_norm)| grad_norm),
            stopped_early: self.early_stop_reason().map(str::to_string),
        })
    }

    /// The report of the 0-indexed step `index`, which returned `(loss, grad_norm)`.
    ///
    /// `mfu_basis` holds FLOPs per token and one device's peak TFLOP/s.
    fn step_report(
        &self,
        index: usize,
        (loss, grad_norm): (f64, f64),
        clock: &Throughput,
        mfu_basis: Option<(f64, f64)>,
    ) -> StepReport {
        let completed = index + 1;
        let seq_len = self.model_config.max_seq_len;
        let session_samples_per_sec = clock.session_rate(completed - self.start_step);
        let recent_samples_per_sec = clock.recent_rate();
        let tokens_per_sec = session_samples_per_sec * seq_len as f64;
        let total_steps = self.lr_plan.total_steps();
        StepReport {
            step: completed,
            loss,
            grad_norm,
            learning_rate: self.lr_at(index),
            epoch: self.epoch_progress(completed),
            logged: self.is_log_step(index),
            session_samples_per_sec,
            recent_samples_per_sec,
            eta_secs: clock.mean_step_secs() * total_steps.saturating_sub(completed) as f64,
            tokens: clock.samples_per_step() * seq_len,
            scored_tokens: self.metrics.last_scored,
            tokens_per_sec,
            recent_tokens_per_sec: recent_samples_per_sec * seq_len as f64,
            mfu: mfu_basis.map(|(flops, peak)| mfu(flops, tokens_per_sec, self.world_size(), peak)),
            spike: self.metrics.last_spike,
        }
    }

    /// Whether the 0-indexed step `index` writes a log line.
    fn is_log_step(&self, index: usize) -> bool {
        let interval = self.config.log_interval;
        index == self.start_step || (interval > 0 && index.is_multiple_of(interval))
    }

    /// Whether an interval checkpoint is due once `completed` steps are done.
    fn is_save_step(&self, completed: usize) -> bool {
        self.config.save_interval > 0 && completed.is_multiple_of(self.config.save_interval)
    }

    /// Appends the `steps.json` line of `report`.
    ///
    /// A write error prints a warning and never stops the run.
    /// Only rank 0 writes.
    fn record_step(&self, report: &StepReport) {
        if self.quiet() {
            return;
        }
        let record = StepRecord {
            step: report.step,
            loss: report.loss,
            grad_norm: report.grad_norm,
            learning_rate: report.learning_rate,
            epoch: report.epoch,
            tokens: report.tokens,
            tokens_per_sec: report.tokens_per_sec,
            scored_tokens: report.scored_tokens,
            mfu: report.mfu,
        };
        if let Err(e) = record.append_to(Path::new(&self.steps_json_path)) {
            eprintln!("Warning: {e}");
        }
    }

    /// Writes the blazr-compatible config.json to the checkpoint directory.
    ///
    /// Every error is a warning in interactive mode and silent in headless mode.
    /// Only rank 0 writes.
    fn save_blazr_config_if_needed(&self) {
        if self.quiet() {
            return;
        }
        let dir = &self.config.checkpoint_dir;
        if !Path::new(dir).exists() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                if !self.headless {
                    eprintln!("Warning: Failed to create checkpoint directory: {}", e);
                }
                return;
            }
        }

        // config.json is rewritten at every start. It derives from the model
        // config that this run's weights follow, so a stale file from an older
        // oxidizr cannot disagree with the weights. Sidecars copy once.
        let first_write = !checkpoint_config_exists(dir);
        match write_checkpoint_config(&self.model_config, dir) {
            Ok(()) => {
                if !self.headless && first_write {
                    println!("📝 Saved config.json for blazr inference compatibility");
                }
            }
            Err(e) => {
                if !self.headless {
                    eprintln!("Warning: Failed to save blazr config.json: {}", e);
                }
            }
        }
        if first_write {
            if let Err(e) = copy_checkpoint_sidecars(&self.config.checkpoint_sidecars, dir) {
                if !self.headless {
                    eprintln!("Warning: Failed to copy checkpoint sidecar files: {}", e);
                }
            }
        }
    }
}
