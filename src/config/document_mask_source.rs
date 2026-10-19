//! Where a run's document separator comes from, and the resume check of the mask.

use super::document_mask::{describe_document_mask, DocumentMaskConfig, DocumentMaskRecord};
use super::{CheckpointMeta, LitConfig};
use crate::data::{read_token_meta, token_meta_path};
use std::path::PathBuf;

/// The separator of `mask` over the token files `files`.
///
/// `separator_id` comes from the config when set, else from the first
/// `.tokens.json` sidecar. Every sidecar present must agree with it.
///
/// # Errors
/// - A sidecar disagrees with the config or with an earlier sidecar: the
///   error names both sources and both ids.
/// - No config value and no sidecar: the error names the key to set.
/// - A sidecar does not parse.
pub fn resolve_separator(mask: &DocumentMaskConfig, files: &[PathBuf]) -> anyhow::Result<u32> {
    let mut found: Option<(u32, String)> = mask
        .separator_id
        .map(|id| (id, "trainer.document_mask.separator_id".to_string()));
    for path in files {
        let Some(meta) = read_token_meta(path)? else {
            continue;
        };
        let here = format!("sidecar {}", token_meta_path(path).display());
        match &found {
            Some((id, from)) if *id != meta.separator_id => anyhow::bail!(
                "{from} gives separator_id {id}, and {here} gives {}. Every data file must end \
                 its documents with one id. Tokenize every source with one tokenizer, or fix \
                 trainer.document_mask.separator_id.",
                meta.separator_id
            ),
            Some(_) => {}
            None => found = Some((meta.separator_id, here)),
        }
    }
    found.map(|(id, _)| id).ok_or_else(|| {
        anyhow::anyhow!(
            "trainer.document_mask needs a separator id, and no data file has a .tokens.json \
             sidecar. Set trainer.document_mask.separator_id, or tokenize the data with \
             `oxidizr tokenize`."
        )
    })
}

/// The token files whose sidecars decide the separator: `train_files`, then the eval file.
pub fn document_mask_files(config: &LitConfig, train_files: &[String]) -> Vec<PathBuf> {
    let eval_file = config
        .trainer
        .eval
        .as_ref()
        .and_then(|eval| eval.data_file.as_ref());
    train_files
        .iter()
        .chain(eval_file)
        .map(PathBuf::from)
        .collect()
}

/// The separator of `config` over `train_files`. `None` when `trainer.document_mask` is off.
///
/// # Errors
/// The errors of [`resolve_separator`].
pub fn resolve_run_separator(
    config: &LitConfig,
    train_files: &[String],
) -> anyhow::Result<Option<u32>> {
    config
        .trainer
        .document_mask
        .as_ref()
        .map(|mask| resolve_separator(mask, &document_mask_files(config, train_files)))
        .transpose()
}

/// Checks the document mask of a run at `start_step` against the meta at `meta_path`.
///
/// A fresh run has nothing to match. A resume whose mask was added, removed
/// or changed warns on stderr. Under `headless` it is an error instead.
/// A resume without a meta reads as a run without a mask.
///
/// # Errors
/// Returns an error when the meta is unreadable, or on a headless mismatch.
pub fn check_document_mask_resume(
    meta_path: Option<&str>,
    start_step: usize,
    current: Option<DocumentMaskRecord>,
    headless: bool,
) -> anyhow::Result<()> {
    if start_step == 0 {
        return Ok(());
    }
    let previous = match meta_path {
        Some(path) => {
            CheckpointMeta::load(path)
                .map_err(|e| {
                    anyhow::anyhow!(
                        "checkpoint meta '{path}' was not read: {e}. Restore the file to resume \
                         with its document mask."
                    )
                })?
                .document_mask
        }
        None => None,
    };
    if previous == current {
        return Ok(());
    }
    let message = format!(
        "trainer.document_mask '{}' differs from the checkpoint's '{}'. The resumed run masks \
         attention and loss differently from the run it continues, so it is not exact.",
        describe_document_mask(current),
        describe_document_mask(previous)
    );
    if headless {
        anyhow::bail!(
            "{message} Restore the checkpoint's trainer.document_mask, or resume without \
             --headless to accept the change."
        );
    }
    eprintln!("WARNING: {message}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{
        write_token_meta, TokenMeta, TokenOptions, TokenizerRecord, TOKEN_META_VERSION,
    };
    use crate::data_hash::HashAlgo;
    use std::path::Path;

    /// The sidecar path of the token file `path`, as error messages print it.
    fn shown(path: &Path) -> String {
        token_meta_path(path).display().to_string()
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("oxidizr-docmask-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn sidecar(separator_id: u32) -> TokenMeta {
        TokenMeta {
            format_version: TOKEN_META_VERSION,
            oxidizr_version: "0.0.0".to_owned(),
            splintr_version: "unknown".to_owned(),
            tokenizer: TokenizerRecord::Name {
                name: "test".to_owned(),
            },
            separator_id,
            bos_id: None,
            tokenizer_vocab_size: 16,
            max_token_id: Some(15),
            num_tokens: 0,
            num_documents: 0,
            skipped_empty_documents: 0,
            hash_algo: HashAlgo::Fnv1a64,
            file_hash: "0000000000000000".to_owned(),
            inputs: Vec::new(),
            options: TokenOptions {
                format: "auto".to_owned(),
                field: "text".to_owned(),
                txt_docs: "file".to_owned(),
                allow_special: false,
            },
        }
    }

    /// A token file at `dir/name`, with a sidecar recording `separator` when given.
    fn file(dir: &Path, name: &str, separator: Option<u32>) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, b"").expect("token file");
        if let Some(id) = separator {
            write_token_meta(&path, &sidecar(id)).expect("sidecar");
        }
        path
    }

    fn mask(separator_id: Option<u32>) -> DocumentMaskConfig {
        DocumentMaskConfig { separator_id }
    }

    #[test]
    fn the_sidecar_gives_the_separator() {
        let dir = scratch("sidecar");
        let files = vec![file(&dir, "a.bin", None), file(&dir, "b.bin", Some(7))];
        assert_eq!(resolve_separator(&mask(None), &files).expect("resolved"), 7);
        assert_eq!(
            resolve_separator(&mask(Some(7)), &files).expect("agrees"),
            7
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_config_value_that_disagrees_names_both() {
        let dir = scratch("config-mismatch");
        let files = vec![file(&dir, "a.bin", Some(7))];
        let e = resolve_separator(&mask(Some(3)), &files)
            .expect_err("3 against 7")
            .to_string();
        assert!(
            e.contains("trainer.document_mask.separator_id gives separator_id 3"),
            "{e}"
        );
        assert!(
            e.contains(&shown(&files[0])) && e.contains("gives 7"),
            "{e}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn two_sidecars_that_disagree_name_both() {
        let dir = scratch("sidecar-mismatch");
        let files = vec![file(&dir, "a.bin", Some(7)), file(&dir, "b.bin", Some(9))];
        let e = resolve_separator(&mask(None), &files)
            .expect_err("7 against 9")
            .to_string();
        assert!(
            e.contains(&shown(&files[0])) && e.contains(&shown(&files[1])),
            "{e}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn no_sidecar_and_no_config_value_names_the_key() {
        let dir = scratch("none");
        let files = vec![file(&dir, "a.bin", None)];
        let e = resolve_separator(&mask(None), &files)
            .expect_err("no source")
            .to_string();
        assert!(e.contains("trainer.document_mask.separator_id"), "{e}");
        assert_eq!(
            resolve_separator(&mask(Some(4)), &files).expect("config"),
            4
        );
        assert_eq!(
            resolve_separator(&mask(Some(4)), &[]).expect("generated"),
            4
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_eval_file_joins_the_train_files() {
        let mut cfg = LitConfig::nano_baseline();
        cfg.trainer.eval = Some(crate::eval::EvalConfig {
            data_file: Some("eval.bin".to_string()),
            holdout_rows: None,
            holdout_fraction: None,
            interval: 1,
            batches: 1,
            early_stopping: None,
        });
        let files = document_mask_files(&cfg, &["train.bin".to_string()]);
        assert_eq!(
            files,
            vec![PathBuf::from("train.bin"), PathBuf::from("eval.bin")]
        );
        assert_eq!(resolve_run_separator(&cfg, &[]).expect("off"), None);
    }

    /// A meta file in `dir` recording `record`.
    fn meta_with(dir: &Path, record: Option<DocumentMaskRecord>) -> String {
        let cfg = LitConfig::nano_baseline();
        let mut meta = CheckpointMeta::with_data_hash(2, &cfg.model, &[], "w", None);
        meta.document_mask = record;
        let path = dir.join("checkpoint_step_2.meta.json");
        meta.save_to(&path).expect("save meta");
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn a_changed_mask_warns_and_fails_headless() {
        let dir = scratch("resume");
        let seven = Some(DocumentMaskRecord { separator_id: 7 });
        let three = Some(DocumentMaskRecord { separator_id: 3 });
        let path = meta_with(&dir, seven);
        check_document_mask_resume(Some(&path), 2, seven, true).expect("same mask");
        check_document_mask_resume(Some(&path), 2, three, false).expect("warns only");
        let changed = check_document_mask_resume(Some(&path), 2, three, true)
            .expect_err("headless change")
            .to_string();
        assert!(changed.contains("'separator_id 3'") && changed.contains("'separator_id 7'"));
        let removed = check_document_mask_resume(Some(&path), 2, None, true)
            .expect_err("headless removal")
            .to_string();
        assert!(removed.contains("'off'"), "{removed}");
        check_document_mask_resume(Some(&path), 0, None, true).expect("a fresh run");

        let off = meta_with(&dir, None);
        let added = check_document_mask_resume(Some(&off), 2, seven, true)
            .expect_err("headless addition")
            .to_string();
        assert!(
            added.contains("differs from the checkpoint's 'off'"),
            "{added}"
        );
        let text = std::fs::read_to_string(&off).expect("meta text");
        assert!(!text.contains("document_mask"), "{text}");
        let _ = std::fs::remove_dir_all(dir);
    }
}
