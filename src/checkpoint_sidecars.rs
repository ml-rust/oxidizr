//! Sidecar files that define a model's input space and must travel with its
//! checkpoint — a custom tokenizer vocab, a speech-corpus layout file. These
//! are not part of the model's weights or `config.json`, but a checkpoint
//! trained against a custom vocab is unreadable without the vocab beside it,
//! so `trainer.checkpoint_sidecars` names them and the trainer copies them
//! into the checkpoint directory whenever it writes `config.json`.

use std::path::Path;

/// A sidecar path named in `trainer.checkpoint_sidecars` does not exist on
/// disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingSidecarError {
    /// The path as it appears after `${VAR}` expansion.
    pub path: String,
}

impl std::fmt::Display for MissingSidecarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "trainer.checkpoint_sidecars names '{}', which does not exist",
            self.path
        )
    }
}

impl std::error::Error for MissingSidecarError {}

/// Checks every sidecar path exists, failing on the first that does not.
/// Run at config-load time so a missing vocab file is caught before training
/// starts, not when the checkpoint is later found unreadable.
pub fn validate_checkpoint_sidecars(paths: &[String]) -> Result<(), MissingSidecarError> {
    for path in paths {
        if !Path::new(path).exists() {
            return Err(MissingSidecarError { path: path.clone() });
        }
    }
    Ok(())
}

/// Copies each sidecar file into `checkpoint_dir`, keeping its file name.
/// Called whenever the trainer writes `checkpoint_dir/config.json`, so the
/// two always land together.
pub fn copy_checkpoint_sidecars(paths: &[String], checkpoint_dir: &str) -> anyhow::Result<()> {
    for path in paths {
        let source = Path::new(path);
        let file_name = source.file_name().ok_or_else(|| {
            anyhow::anyhow!("trainer.checkpoint_sidecars entry '{path}' names no file")
        })?;
        let dest = Path::new(checkpoint_dir).join(file_name);
        std::fs::copy(source, &dest).map_err(|e| {
            anyhow::anyhow!(
                "failed to copy checkpoint sidecar '{path}' to '{}': {e}",
                dest.display()
            )
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(test_name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("oxidizr-checkpoint-sidecars-test-{test_name}"))
    }

    #[test]
    fn validate_passes_when_every_path_exists() {
        let dir = scratch_dir("validate-ok");
        std::fs::create_dir_all(&dir).unwrap();
        let vocab = dir.join("vocab.json");
        std::fs::write(&vocab, "{}").unwrap();

        let paths = vec![vocab.to_str().unwrap().to_string()];
        assert!(validate_checkpoint_sidecars(&paths).is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn validate_names_the_missing_path() {
        let dir = scratch_dir("validate-missing");
        std::fs::create_dir_all(&dir).unwrap();
        let missing = dir.join("does-not-exist.json");
        let missing_str = missing.to_str().unwrap().to_string();

        let err = validate_checkpoint_sidecars(std::slice::from_ref(&missing_str)).unwrap_err();
        assert_eq!(err.path, missing_str);
        assert!(err.to_string().contains(&missing_str));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn copy_places_each_file_under_its_own_name_in_the_checkpoint_dir() {
        let dir = scratch_dir("copy-ok");
        let source_dir = dir.join("source");
        let checkpoint_dir = dir.join("checkpoint");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&checkpoint_dir).unwrap();

        let vocab = source_dir.join("vocab.json");
        std::fs::write(&vocab, r#"{"a":1}"#).unwrap();

        let paths = vec![vocab.to_str().unwrap().to_string()];
        copy_checkpoint_sidecars(&paths, checkpoint_dir.to_str().unwrap()).unwrap();

        let copied = checkpoint_dir.join("vocab.json");
        assert!(copied.exists());
        assert_eq!(std::fs::read_to_string(copied).unwrap(), r#"{"a":1}"#);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn copy_errors_naming_the_missing_source() {
        let dir = scratch_dir("copy-missing");
        std::fs::create_dir_all(&dir).unwrap();
        let missing = dir.join("missing.json");
        let missing_str = missing.to_str().unwrap().to_string();

        let err =
            copy_checkpoint_sidecars(std::slice::from_ref(&missing_str), dir.to_str().unwrap()).unwrap_err();
        assert!(err.to_string().contains(&missing_str));

        std::fs::remove_dir_all(&dir).ok();
    }
}
