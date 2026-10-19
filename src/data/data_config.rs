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

use crate::config_paths::expand_env_vars;
use serde::{Deserialize, Serialize};

/// The training data of a run: token files read as one stream in order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    /// Token files, read in this order.
    #[serde(default)]
    pub sources: Vec<DataSourceSpec>,
}

/// One token file of [`DataConfig::sources`].
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DataSourceSpec {
    /// Path of a raw little-endian `u32` token file. `${VAR}` expands.
    pub path: String,
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
        }
        Ok(())
    }

    /// Source paths in order.
    pub fn paths(&self) -> Vec<String> {
        self.sources.iter().map(|s| s.path.clone()).collect()
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

    #[test]
    fn sources_parse_in_order() {
        let data = config("sources: [{path: a.bin}, {path: b.bin}]");
        assert_eq!(data.paths(), vec!["a.bin", "b.bin"]);
        assert!(data.validate().is_ok());
        assert!(serde_yaml::from_str::<DataConfig>("sources: [{file: a.bin}]").is_err());
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
