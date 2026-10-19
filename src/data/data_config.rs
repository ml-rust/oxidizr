//! The `trainer.data` block of a config, and how it combines with `-d`.
//!
//! ```yaml
//! trainer:
//!   data:
//!     sources:
//!       - path: data/a.bin
//!       - path: data/b.bin
//! ```
//!
//! The run reads the sources as one row stream in list order. Repeated
//! `-d/--data` flags name the same list on the command line.
//!
//! A `weight` on every source turns the stream into a weighted mixture.
//! Each epoch then holds `rows_per_epoch` rows, and each source supplies its
//! share of them. See [`crate::data::LitDataLoader::with_mixture`].

use super::mixture::weight_error;
use crate::config_paths::expand_env_vars;
use serde::{Deserialize, Serialize};

/// The training data of a run: token files read as one stream in order, or
/// as a weighted mixture.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    /// Token files, read in this order.
    #[serde(default)]
    pub sources: Vec<DataSourceSpec>,
    /// Rows per epoch of a weighted mixture. Absent: every training row of
    /// every source. Needs a weight on every source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows_per_epoch: Option<usize>,
}

/// One token file of [`DataConfig::sources`].
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DataSourceSpec {
    /// Path of a raw little-endian `u32` token file. `${VAR}` expands.
    pub path: String,
    /// Share of each epoch's rows, relative to the other weights. Finite
    /// and above 0. Set on every source or on none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
}

impl DataConfig {
    /// Expands `${VAR}` in every source path.
    ///
    /// # Errors
    /// Returns an error naming the key of a path whose variable is unset.
    pub fn expand_env_vars(&mut self) -> anyhow::Result<()> {
        for (i, source) in self.sources.iter_mut().enumerate() {
            source.path =
                expand_env_vars(&source.path, &format!("trainer.data.sources[{i}].path"))?;
        }
        Ok(())
    }

    /// Checks every key. Each error names the key it rejects.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.sources.is_empty() {
            anyhow::bail!(
                "trainer.data.sources is empty. List at least one token file, or remove \
                 trainer.data."
            );
        }
        for (i, source) in self.sources.iter().enumerate() {
            if source.path.trim().is_empty() {
                anyhow::bail!(
                    "trainer.data.sources[{i}].path is empty. Set it to a token file, or \
                     remove the entry."
                );
            }
            if let Some(reason) = source.weight.and_then(|w| weight_error(i, w)) {
                anyhow::bail!(reason);
            }
        }
        let weighted = self.sources.iter().position(|s| s.weight.is_some());
        let unweighted = self.sources.iter().position(|s| s.weight.is_none());
        if let (Some(set), Some(unset)) = (weighted, unweighted) {
            anyhow::bail!(
                "trainer.data.sources[{set}].weight is set, and trainer.data.sources[{unset}].weight \
                 is not. Set a weight on every source, or remove every weight."
            );
        }
        match self.rows_per_epoch {
            Some(0) => anyhow::bail!(
                "trainer.data.rows_per_epoch is 0: an epoch needs at least 1 row. Set it above \
                 0, or remove it to read every training row per epoch."
            ),
            Some(rows) if weighted.is_none() => anyhow::bail!(
                "trainer.data.rows_per_epoch is {rows}, and no source sets a weight. Set a \
                 weight on every source of trainer.data.sources, or remove \
                 trainer.data.rows_per_epoch."
            ),
            _ => Ok(()),
        }
    }

    /// Source paths in order.
    pub fn paths(&self) -> Vec<String> {
        self.sources.iter().map(|s| s.path.clone()).collect()
    }

    /// The weight of every source, in source order. `None` when no source sets one.
    ///
    /// A config that passed [`validate`](Self::validate) sets all or none.
    pub fn weights(&self) -> Option<Vec<f64>> {
        self.sources.iter().map(|s| s.weight).collect()
    }
}

/// The run's data files: `-d/--data` flags in order, else `trainer.data.sources`.
///
/// No file from either place means a run on generated tokens.
///
/// # Errors
/// Returns an error when both name files.
pub fn resolve_data_files(
    cli: &[String],
    config: Option<&DataConfig>,
) -> anyhow::Result<Vec<String>> {
    match config {
        Some(data) if !cli.is_empty() && !data.sources.is_empty() => anyhow::bail!(
            "-d/--data names {} file(s) and trainer.data.sources names {}. Name the data \
             files in one place: remove the -d flags, or remove trainer.data.",
            cli.len(),
            data.sources.len()
        ),
        Some(data) if cli.is_empty() => Ok(data.paths()),
        _ => Ok(cli.to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(yaml: &str) -> DataConfig {
        serde_yaml::from_str(yaml).expect("data config")
    }

    fn refused(yaml: &str) -> String {
        config(yaml).validate().expect_err("refused").to_string()
    }

    #[test]
    fn sources_parse_in_order() {
        let data = config("sources: [{path: a.bin}, {path: b.bin}]");
        assert_eq!(data.paths(), vec!["a.bin", "b.bin"]);
        assert_eq!(data.weights(), None);
        assert!(data.validate().is_ok());
        assert!(serde_yaml::from_str::<DataConfig>("sources: [{file: a.bin}]").is_err());
    }

    #[test]
    fn weights_and_rows_per_epoch_parse() {
        let data = config(
            "sources: [{path: a.bin, weight: 0.7}, {path: b.bin, weight: 0.3}]\n\
             rows_per_epoch: 1000",
        );
        assert!(data.validate().is_ok());
        assert_eq!(data.weights(), Some(vec![0.7, 0.3]));
        assert_eq!(data.rows_per_epoch, Some(1000));
        assert!(config("sources: [{path: a.bin, weight: 2}]")
            .validate()
            .is_ok());
    }

    #[test]
    fn bad_weights_and_rows_per_epoch_are_refused() {
        for bad in ["0", "-1.5", ".nan", ".inf"] {
            let text = refused(&format!("sources: [{{path: a.bin, weight: {bad}}}]"));
            assert!(text.contains("trainer.data.sources[0].weight"), "{text}");
        }
        let text = refused("sources: [{path: a.bin}, {path: b.bin, weight: 1}]");
        assert!(
            text.contains("sources[1].weight is set") && text.contains("sources[0].weight"),
            "{text}"
        );
        let text = refused("sources: [{path: a.bin, weight: 1}]\nrows_per_epoch: 0");
        assert!(text.contains("trainer.data.rows_per_epoch is 0"), "{text}");
        let text = refused("sources: [{path: a.bin}]\nrows_per_epoch: 10");
        assert!(text.contains("no source sets a weight"), "{text}");
    }

    #[test]
    fn empty_sources_and_paths_are_refused() {
        let err = config("sources: []").validate().expect_err("empty list");
        assert!(
            err.to_string().contains("trainer.data.sources is empty"),
            "{err}"
        );
        let err = config("sources: [{path: ' '}]")
            .validate()
            .expect_err("empty path");
        assert!(
            err.to_string().contains("trainer.data.sources[0].path"),
            "{err}"
        );
    }

    #[test]
    fn cli_or_config_names_the_files() {
        let cli = vec!["x.bin".to_string(), "y.bin".to_string()];
        let data = config("sources: [{path: a.bin}]");
        assert_eq!(resolve_data_files(&cli, None).expect("cli"), cli);
        assert_eq!(
            resolve_data_files(&[], Some(&data)).expect("yaml"),
            vec!["a.bin"]
        );
        assert!(resolve_data_files(&[], None).expect("none").is_empty());
        let err = resolve_data_files(&cli, Some(&data)).expect_err("both");
        let text = err.to_string();
        assert!(
            text.contains("-d/--data") && text.contains("trainer.data.sources"),
            "{text}"
        );
    }
}
