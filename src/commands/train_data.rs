//! The loaders of a training run: the training rows, and the eval rows under `trainer.eval`.

use crate::config::LitConfig;
use crate::data::{create_dummy_data, LitDataLoader};
use crate::eval::{Evaluator, HoldoutPlan};
use crate::prelude::DType;
use anyhow::Result;
use boostr::runtime::Runtime;

/// Generated tokens of a run without `--data`.
const GENERATED_TOKENS: usize = 1_000_000;

/// Where the training tokens come from.
pub enum TokenSource {
    /// A memory-mapped `.bin` file of u32 tokens.
    File(String),
    /// Random tokens of a run without `--data`.
    Generated(Vec<u32>),
}

impl TokenSource {
    /// `--data` when given, else random tokens under the model's vocab.
    pub fn for_run(data_file: Option<&str>, config: &LitConfig) -> Self {
        match data_file {
            Some(path) => Self::File(path.to_string()),
            None => Self::Generated(create_dummy_data(config.model.vocab_size, GENERATED_TOKENS)),
        }
    }

    /// A loader over every row of this source, with the run's batch and row width.
    pub fn open<R: Runtime<DType = DType>>(
        &self,
        config: &LitConfig,
        device: &R::Device,
    ) -> Result<LitDataLoader<R>> {
        let (batch_size, seq_len) = (config.trainer.batch_size, config.model.max_seq_len);
        Ok(match self {
            Self::File(path) => {
                LitDataLoader::<R>::from_file(path, batch_size, seq_len, device.clone())?
            }
            Self::Generated(tokens) => {
                LitDataLoader::<R>::new(tokens.clone(), batch_size, seq_len, device.clone())
            }
        })
    }
}

/// The training loader, and the eval of `trainer.eval` when set.
pub struct RunLoaders<R: Runtime<DType = DType>> {
    /// Reads the training rows. The row order is still the default.
    pub train: LitDataLoader<R>,
    pub evaluator: Option<Evaluator<R>>,
    /// Rows held out of `train`. 0 without a holdout.
    pub holdout_rows: usize,
}

/// Opens the loaders of a run over `source`, each with the run's loss mask.
///
/// Under a holdout, training reads rows `[0, N - h)` and the eval reads
/// `[N - h, N)` of the same source. An eval on `trainer.eval.data_file`
/// reads that file, and training keeps every row.
pub fn run_loaders<R: Runtime<DType = DType>>(
    config: &LitConfig,
    source: &TokenSource,
    device: &R::Device,
) -> Result<RunLoaders<R>> {
    let train = with_loss_mask(source.open(config, device)?, config)?;
    let Some(eval) = &config.trainer.eval else {
        return Ok(RunLoaders {
            train,
            evaluator: None,
            holdout_rows: 0,
        });
    };
    let plan = HoldoutPlan::for_config(eval, train.total_rows(), config.trainer.batch_size)?;
    let eval_source = match &eval.data_file {
        Some(path) => TokenSource::File(path.clone()).open(config, device),
        None => source.open(config, device),
    }?;
    let evaluator = Evaluator::new(eval, plan, with_loss_mask(eval_source, config)?)?;
    let train = if plan.holdout_rows > 0 {
        train.with_row_range(0, plan.train_rows)?
    } else {
        train
    };
    Ok(RunLoaders {
        train,
        evaluator: Some(evaluator),
        holdout_rows: plan.holdout_rows,
    })
}

/// Applies the loss mask of `config` to `loader`.
///
/// Masked loss scoring is opt-in. Without `speech_layout` or `target_mask`
/// the loader emits no mask and every position is scored.
/// `try_with_speech_layout` rejects a corpus whose layout sidecar disagrees
/// with the configured layout. `LitConfig::validate` rejects a config that
/// sets both, so the two arms never overlap on a config from `from_yaml`.
fn with_loss_mask<R: Runtime<DType = DType>>(
    loader: LitDataLoader<R>,
    config: &LitConfig,
) -> Result<LitDataLoader<R>> {
    Ok(
        match (&config.model.speech_layout, &config.trainer.target_mask) {
            (Some(spec), None) => loader.try_with_speech_layout(spec.layout())?,
            (None, Some(spec)) => loader.with_target_mask(spec.target_mask()),
            (None, None) => loader,
            (Some(_), Some(_)) => anyhow::bail!(
                "model.speech_layout and trainer.target_mask are mutually exclusive: both \
                 configure the loader's loss mask"
            ),
        },
    )
}
