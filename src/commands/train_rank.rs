//! One rank of a training run: its model, loaders, trainer and fit loop.
//!
//! Every rank builds its own model, loader and trainer on its own device.
//! Rank 0 alone prints, evaluates, and fills the run record.

use crate::checkpoint_set::OptimizerResume;
use crate::commands::train_data::{run_loaders, RunLoaders};
use crate::commands::train_plan::RunPlan;
use crate::commands::train_report::{
    cuda_memory_info, estimate_params, finish_rank, print_banner, run_config_json, BannerFacts,
};
use crate::commands::train_resume::resume_data_position;
use crate::data::{PrefetchDataLoader, RowOrder};
use crate::dp::RankSlot;
use crate::eval::read_eval_resume;
use crate::model::{load_checkpoint_into_model, LitGPT};
use crate::prelude::{DType, VarMap};
use crate::run_record::{RunProgress, STEPS_FILE};
use crate::trainer::{LitTrainer, TrainerRunConfig};
use anyhow::Result;
use boostr::ops::{FusedOptimizerOps, TensorOps};
use boostr::runtime::{Runtime, RuntimeClient};
use boostr::tensor::TensorId;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

/// What one rank trains with.
pub struct RankRun<'a, R: Runtime> {
    pub plan: &'a RunPlan,
    pub slot: RankSlot,
    pub device: R::Device,
    pub client: R::Client,
    /// The run record. Only rank 0 receives it.
    pub progress: Option<&'a Mutex<RunProgress>>,
}

/// Apply `update` to the run record, when this rank holds it.
fn note(progress: Option<&Mutex<RunProgress>>, update: impl FnOnce(&mut RunProgress)) {
    if let Some(progress) = progress {
        let mut guard = progress.lock().unwrap_or_else(PoisonError::into_inner);
        update(&mut guard);
    }
}

/// Trains one rank of `run.plan` to the end, or to its first error.
///
/// At world 1 this is the whole single-device run. At world above 1 the
/// trainer joins the rank's communicator, and its abort hook records the
/// failure for the launcher.
///
/// # Errors
/// Returns the first error of this rank: model, loader, resume or step.
pub fn train_rank<R>(run: RankRun<'_, R>) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client: RuntimeClient<R>
        + TensorOps<R>
        + FusedOptimizerOps<R>
        + boostr::RandomOps<R>
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
    let RankRun {
        plan,
        slot,
        device,
        client,
        progress,
    } = run;
    let (config, resume) = (&plan.config, &plan.resume);
    let rank0 = slot.rank == 0;
    let say = rank0 && !plan.headless;
    if say {
        println!("Using runtime: {}", R::name());
        if slot.world > 1 {
            println!(
                "Data parallel: {} ranks, GPUs {:?}",
                slot.world, plan.gpu_ids
            );
        }
        println!(
            "Initializing model with {} parameters...",
            estimate_params(&config.model)
        );
    }

    // Model migration owns initialization through boostr VarBuilder/Init and must
    // expose trainable parameters with stable TensorIds.
    let mut varmap = VarMap::<R>::new();
    let mut model = LitGPT::<R>::new_seeded(config, &mut varmap, &device, &client)?;
    if let Some(weights) = &resume.weights {
        load_checkpoint_into_model(&mut model, weights)?;
        if say {
            println!("Checkpoint weights loaded: {weights}");
        }
    }
    let batch_size = config.trainer.batch_size;
    if say {
        println!("Model initialized successfully");
        println!(
            "Using batch_size={} per rank, seq_len={} (from config)",
            batch_size, config.model.max_seq_len
        );
    }

    let RunLoaders {
        train,
        evaluator,
        holdout_rows,
    } = run_loaders::<R>(config, &plan.source, &device)?;
    // Before the seek: the held-out rows decide what the saved data position means.
    let meta = resume.meta_path.as_deref();
    let eval_resume = read_eval_resume(meta, resume.start_step, holdout_rows)?;
    // Only rank 0 evaluates. The others learn its early-stop decision by broadcast.
    let evaluator = evaluator.filter(|_| rank0).map(|e| e.resumed(eval_resume));
    if let (Some(eval), true) = (&evaluator, say) {
        println!(
            "Eval: {} held-out rows, {} batches per eval",
            eval.holdout_rows(),
            eval.batches()
        );
    }

    let row_order = if config.trainer.shuffle_rows {
        RowOrder::Shuffled {
            seed: config.trainer.seed.unwrap_or(0),
        }
    } else {
        RowOrder::Stream
    };
    let mut loader = train.with_row_order(row_order);
    if slot.world > 1 {
        loader = loader.with_rank(slot.rank, slot.world)?;
    }
    let dataset_size = loader.dataset_size();
    let row_layout = loader.row_layout()?;
    let accum = config.trainer.gradient_accumulation;
    let start_position = resume_data_position(meta, resume.start_step, &row_layout, accum, !rank0)?;
    loader.seek(start_position)?;
    if say {
        let prefetch = match plan.prefetch {
            0 => String::new(),
            n => format!(" (prefetch={n})"),
        };
        println!(
            "Data loader ready with {} tokens in {} rows of {} (row order: {:?}){}",
            dataset_size,
            row_layout.num_rows(),
            config.model.max_seq_len + 1,
            row_order,
            prefetch
        );
        println!(
            "Data position: epoch {} row {}\n",
            start_position.epoch, start_position.row
        );
    }

    // Whole global batches per epoch; the partial batch at an epoch's end is dropped.
    let steps_per_epoch = row_layout.batches_per_epoch() as f64 / accum as f64;
    let total_steps = row_layout.total_steps(accum, config.trainer.num_epochs);
    // Rows per optimizer step across every rank: global batch x accumulation.
    let effective_batch_size = row_layout.batch_size() * accum;
    note(progress, |p| {
        p.dataset_size = Some(dataset_size);
        p.config = Some(run_config_json(
            plan,
            &row_layout,
            total_steps,
            effective_batch_size,
        ));
    });

    // There is no global tape: the optimizer owns a HashMap keyed by the model's
    // logical TensorIds, and every step rebuilds Vars with those same IDs.
    let trainable = model.trainable_parameters();
    let parameter_ids: Vec<TensorId> = trainable.iter().map(|(id, _)| *id).collect();
    let mut params: std::collections::HashMap<_, _> = trainable
        .into_iter()
        .map(|(id, var)| (id, var.tensor().clone()))
        .collect();
    let parameter_names: Vec<(String, TensorId)> = model
        .named_trainable_parameters()?
        .into_iter()
        .map(|(name, id, _)| (name, id))
        .collect();
    let trainable_params = params.values().map(|t| t.elem_count()).sum::<usize>();

    let steps_json_path = Path::new(&plan.run_dir)
        .join(STEPS_FILE)
        .to_string_lossy()
        .into_owned();
    let mut trainer = LitTrainer::<R>::new(
        config.trainer.clone(),
        &mut params,
        parameter_ids,
        parameter_names,
        client.clone(),
        device.clone(),
        TrainerRunConfig {
            row_layout,
            start_position,
            batch_size,
            steps_json_path,
            headless: plan.headless,
            start_step: resume.start_step,
            model_config: config.model.clone(),
            data_files: plan.data_files.clone(),
            weight_decay: plan.weight_decay.clone(),
        },
    )?;
    if slot.world > 1 {
        trainer = trainer
            .with_communicator(Arc::clone(&slot.comm))?
            .with_abort_hook(Arc::clone(&slot.abort));
    }
    // A quiet rank restores the same state without repeating rank 0's warnings.
    let optimizer_resume = match &plan.optimizer_resume {
        OptimizerResume::Inexact { .. } if !rank0 => OptimizerResume::Fresh,
        other => other.clone(),
    };
    trainer.apply_optimizer_resume(&optimizer_resume)?;
    if rank0 || (plan.loss_scale_state.is_some() && trainer.loss_scale_state().is_some()) {
        trainer.restore_loss_scale(plan.loss_scale_state)?;
    }
    // Every rank restores the same state, so every rank flags the same steps.
    trainer.restore_spike_guard(plan.spike_guard_state)?;
    if let Some(evaluator) = evaluator {
        trainer.attach_eval(evaluator);
    }
    if rank0 {
        // Headless stdout carries JSON metrics only.
        let decay_summary = trainer.decay_plan().summary(config.trainer.weight_decay);
        if plan.headless {
            eprintln!("{decay_summary}");
        } else {
            println!("{decay_summary}");
        }
    }
    if say {
        println!("Trainer initialized with AdamW optimizer\n");
        if let Some(mem_info) = cuda_memory_info() {
            println!("GPU Memory: {}\n", mem_info);
        }
        print_banner(&BannerFacts {
            config,
            dataset_size,
            trainable_params,
            row_layout: &row_layout,
            steps_per_epoch,
            total_steps,
            effective_batch_size,
        });
    }

    let training_result = if plan.prefetch > 0 {
        let mut prefetch_loader = PrefetchDataLoader::<R>::new(loader, plan.prefetch)?;
        trainer.fit(&model, &mut prefetch_loader)
    } else {
        trainer.fit(&model, &mut loader)
    };
    note(progress, |p| {
        p.final_step = trainer.global_step();
        p.last_loss = trainer.last_loss();
        p.eval = trainer.eval_outcome();
        p.metrics = trainer.run_metrics();
    });

    let stopped = trainer.early_stop_reason();
    if let (Some(reason), true, true) = (stopped, say, training_result.is_ok()) {
        println!("\nEarly stop at step {}: {reason}", trainer.global_step());
    }
    finish_rank(training_result, config, say)
}
