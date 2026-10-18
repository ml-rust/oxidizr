//! The `trainer.eval` block of a config and its checks.

use serde::{Deserialize, Serialize};

fn default_interval() -> usize {
    500
}

fn default_batches() -> usize {
    50
}

fn default_patience() -> usize {
    5
}

/// Validation runs on held-out rows or a separate eval file.
///
/// Without `data_file`, exactly one of `holdout_rows` and `holdout_fraction`
/// names the rows split off the end of the training data. Training never
/// reads those rows.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalConfig {
    /// Token file the eval reads in full. Training keeps every row of its own data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_file: Option<String>,
    /// Whole rows split off the end of the training data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holdout_rows: Option<usize>,
    /// Fraction of the training rows split off the end, in `(0, 1)`. Rounds down.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holdout_fraction: Option<f64>,
    /// Steps between evals. An eval also runs after the last step.
    #[serde(default = "default_interval")]
    pub interval: usize,
    /// Batches per eval. Capped at the whole batches the eval rows hold.
    #[serde(default = "default_batches")]
    pub batches: usize,
    /// Stop training once the eval loss stops improving.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early_stopping: Option<EarlyStoppingConfig>,
}

/// When an eval loss counts as an improvement, and how many misses stop the run.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EarlyStoppingConfig {
    /// Evals in a row without an improvement that stop the run.
    #[serde(default = "default_patience")]
    pub patience: usize,
    /// An eval improves only when its loss is below `best - min_delta`.
    #[serde(default)]
    pub min_delta: f64,
}

impl EvalConfig {
    /// Checks every key. Each error names the key it rejects.
    pub fn validate(&self) -> anyhow::Result<()> {
        match (&self.data_file, self.holdout_rows, self.holdout_fraction) {
            (Some(_), Some(_), _) => anyhow::bail!(
                "trainer.eval.holdout_rows and trainer.eval.data_file are both set. Remove \
                 trainer.eval.holdout_rows to eval on the file, or remove trainer.eval.data_file."
            ),
            (Some(_), _, Some(_)) => anyhow::bail!(
                "trainer.eval.holdout_fraction and trainer.eval.data_file are both set. Remove \
                 trainer.eval.holdout_fraction to eval on the file, or remove trainer.eval.data_file."
            ),
            (Some(path), None, None) if path.trim().is_empty() => anyhow::bail!(
                "trainer.eval.data_file is empty. Set it to a token file, or remove it."
            ),
            (None, None, None) => anyhow::bail!(
                "trainer.eval names no eval rows. Set trainer.eval.holdout_rows, \
                 trainer.eval.holdout_fraction or trainer.eval.data_file."
            ),
            (None, Some(_), Some(_)) => anyhow::bail!(
                "trainer.eval.holdout_rows and trainer.eval.holdout_fraction are both set. \
                 Keep exactly one of them."
            ),
            (None, Some(0), None) => anyhow::bail!(
                "trainer.eval.holdout_rows is 0. Set it to at least trainer.batch_size rows."
            ),
            (None, None, Some(fraction)) if !(fraction > 0.0 && fraction < 1.0) => {
                anyhow::bail!(
                    "trainer.eval.holdout_fraction is {fraction}. Set it to a number between \
                     0 and 1, both excluded."
                )
            }
            _ => {}
        }
        if self.interval == 0 {
            anyhow::bail!("trainer.eval.interval is 0. Set it to a positive step count.");
        }
        if self.batches == 0 {
            anyhow::bail!("trainer.eval.batches is 0. Set it to a positive batch count.");
        }
        if let Some(stop) = &self.early_stopping {
            if stop.patience == 0 {
                anyhow::bail!(
                    "trainer.eval.early_stopping.patience is 0. Set it to a positive eval count."
                );
            }
            if !(stop.min_delta.is_finite() && stop.min_delta >= 0.0) {
                anyhow::bail!(
                    "trainer.eval.early_stopping.min_delta is {}. Set it to a finite number \
                     of 0 or more.",
                    stop.min_delta
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(yaml: &str) -> anyhow::Result<EvalConfig> {
        let config: EvalConfig = serde_yaml::from_str(yaml)?;
        config.validate()?;
        Ok(config)
    }

    fn error(yaml: &str) -> String {
        match parse(yaml) {
            Ok(config) => panic!("{yaml} must be refused, got {config:?}"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn defaults_fill_interval_batches_and_patience() {
        let config = parse("holdout_rows: 64\nearly_stopping: {}").expect("valid");
        assert_eq!(config.interval, 500);
        assert_eq!(config.batches, 50);
        let stop = config.early_stopping.expect("early stopping");
        assert_eq!(stop.patience, 5);
        assert_eq!(stop.min_delta, 0.0);
        assert!(parse("holdout_fraction: 0.05").is_ok());
        assert!(parse("data_file: eval.bin").is_ok());
    }

    #[test]
    fn each_error_names_its_key() {
        let cases = [
            ("interval: 10", "trainer.eval.holdout_rows"),
            ("holdout_rows: 4\nholdout_fraction: 0.1", "holdout_fraction"),
            ("holdout_rows: 0", "trainer.eval.holdout_rows is 0"),
            (
                "holdout_fraction: 1.0",
                "trainer.eval.holdout_fraction is 1",
            ),
            (
                "holdout_fraction: 0.0",
                "trainer.eval.holdout_fraction is 0",
            ),
            (
                "data_file: e.bin\nholdout_rows: 4",
                "trainer.eval.data_file",
            ),
            (
                "data_file: e.bin\nholdout_fraction: 0.1",
                "trainer.eval.data_file",
            ),
            ("data_file: ' '", "trainer.eval.data_file is empty"),
            ("holdout_rows: 4\ninterval: 0", "trainer.eval.interval"),
            ("holdout_rows: 4\nbatches: 0", "trainer.eval.batches"),
            (
                "holdout_rows: 4\nearly_stopping: {patience: 0}",
                "trainer.eval.early_stopping.patience",
            ),
            (
                "holdout_rows: 4\nearly_stopping: {min_delta: -0.1}",
                "trainer.eval.early_stopping.min_delta",
            ),
        ];
        for (yaml, key) in cases {
            let text = error(yaml);
            assert!(text.contains(key), "{yaml}: {text}");
        }
    }

    #[test]
    fn an_unknown_key_is_an_error() {
        let text = error("holdout_rows: 4\nintervall: 10");
        assert!(text.contains("intervall"), "{text}");
    }
}
