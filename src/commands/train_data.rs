//! The loaders of a training run: the training rows, and the eval rows under `trainer.eval`.

use crate::config::LitConfig;
use crate::data::{create_dummy_data, LitDataLoader};
use crate::eval::{Evaluator, HoldoutSplit};
use crate::prelude::DType;
use anyhow::Result;
use boostr::runtime::Runtime;

/// Generated tokens of a run without data files.
const GENERATED_TOKENS: usize = 1_000_000;

/// Where the training tokens come from.
pub enum TokenSource {
    /// Memory-mapped `.bin` files of u32 tokens, read as one stream in order.
    Files(Vec<String>),
    /// Random tokens of a run without data files.
    Generated(Vec<u32>),
}

impl TokenSource {
    /// The data files when any are given, else random tokens under the model's vocab.
    pub fn for_run(data_files: &[String], config: &LitConfig) -> Self {
        if data_files.is_empty() {
            Self::Generated(create_dummy_data(config.model.vocab_size, GENERATED_TOKENS))
        } else {
            Self::Files(data_files.to_vec())
        }
    }

    /// A loader over every row of this source, with the run's batch and row width.
    ///
    /// A file whose `.tokens.json` sidecar records an id at or above
    /// `model.vocab_size` is an error.
    pub fn open<R: Runtime<DType = DType>>(
        &self,
        config: &LitConfig,
        device: &R::Device,
    ) -> Result<LitDataLoader<R>> {
        let (batch_size, seq_len) = (config.trainer.batch_size, config.model.max_seq_len);
        Ok(match self {
            Self::Files(paths) => {
                let loader =
                    LitDataLoader::<R>::from_files(paths, batch_size, seq_len, device.clone())?;
                loader.check_token_ids(config.model.vocab_size)?;
                loader
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
/// Under a holdout, each source of `N` rows keeps rows `[0, N - h)` for
/// training, and the eval reads the last `h` rows of every source in order.
/// The two sides share no row. An eval on `trainer.eval.data_file` reads
/// that file, and training keeps every row. Weights in `trainer.data` mix
/// the training rows. The eval reads its rows unmixed.
pub fn run_loaders<R: Runtime<DType = DType>>(
    config: &LitConfig,
    source: &TokenSource,
    device: &R::Device,
) -> Result<RunLoaders<R>> {
    let train = with_loss_mask(source.open(config, device)?, config)?;
    let Some(eval) = &config.trainer.eval else {
        return Ok(RunLoaders {
            train: with_mixture(train, config)?,
            evaluator: None,
            holdout_rows: 0,
        });
    };
    let split = HoldoutSplit::for_sources(eval, &train.source_rows(), config.trainer.batch_size)?;
    let eval_source = match &eval.data_file {
        Some(path) => TokenSource::Files(vec![path.clone()]).open(config, device),
        None => source.open(config, device),
    }?;
    let evaluator = Evaluator::new(eval, &split, with_loss_mask(eval_source, config)?)?;
    let train = if split.plan.holdout_rows > 0 {
        train.with_row_ranges(&split.train_ranges)?
    } else {
        train
    };
    // After the holdout: the mixture covers the training rows only, and the eval stays unmixed.
    let train = with_mixture(train, config)?;
    Ok(RunLoaders {
        train,
        evaluator: Some(evaluator),
        holdout_rows: split.plan.holdout_rows,
    })
}

/// Applies the weighted mixture of `trainer.data` to `loader`, when its sources set weights.
fn with_mixture<R: Runtime<DType = DType>>(
    loader: LitDataLoader<R>,
    config: &LitConfig,
) -> Result<LitDataLoader<R>> {
    let Some(data) = &config.trainer.data else {
        return Ok(loader);
    };
    Ok(match data.weights() {
        Some(weights) => loader.with_mixture(&weights, data.rows_per_epoch)?,
        None => loader,
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
