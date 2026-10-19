//! The data sources a checkpoint meta records, and the resume check against them.
//!
//! A meta lists each source with its path, its [`compute_file_hash`] hash and
//! its whole rows. A meta from before `data_sources` existed reads as one
//! source: its `data_file_path` and `data_file_hash`, with unknown rows.
//!
//! A resume compares sources by position. A changed count, hash or row count
//! is a mismatch. A changed path alone is not: a file can move.

use crate::config::CheckpointMeta;
use crate::data_hash::{compute_file_hash, legacy_file_hash, HashAlgo};
use serde::{Deserialize, Serialize};

/// One data source as a checkpoint meta records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataSourceRecord {
    /// The path the run read.
    pub path: String,
    /// [`compute_file_hash`] of the file.
    pub hash: String,
    /// Whole rows of `max_seq_len + 1` tokens in the file.
    pub rows: usize,
}

/// The data part of a resume check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataCheck {
    /// The data matches, or a legacy hash leaves it unchecked.
    pub matches: bool,
    /// False when a legacy meta's hash differs: `matches` is then true unchecked.
    pub checked: bool,
    /// The recorded hashes, comma-separated in source order.
    pub checkpoint_hash: String,
    /// The current hashes, comma-separated in source order.
    pub current_hash: String,
    /// What differs, with the next action. `None` when `matches`.
    pub mismatch: Option<String>,
}

/// The records of the token files at `paths` under rows of `seq_len + 1` tokens.
///
/// # Errors
/// Returns an error naming a file that cannot be read.
pub fn source_records(paths: &[String], seq_len: usize) -> anyhow::Result<Vec<DataSourceRecord>> {
    paths
        .iter()
        .map(|path| {
            let hash = compute_file_hash(path)?;
            let bytes = std::fs::metadata(path)
                .map_err(|e| anyhow::anyhow!("failed to read data file '{path}': {e}"))?
                .len();
            let rows = (bytes / 4) as usize / (seq_len + 1);
            Ok(DataSourceRecord {
                path: path.clone(),
                hash,
                rows,
            })
        })
        .collect()
}

/// Records `sources` in `meta`.
///
/// `data_sources` lists every source. A one-source run also fills
/// `data_file_path` and `data_file_hash`, as metas before `data_sources` did.
/// Under several sources both stay empty, so an older oxidizr never matches
/// the meta against one file.
pub fn set_meta_sources(meta: &mut CheckpointMeta, sources: &[DataSourceRecord]) {
    meta.data_sources = sources.to_vec();
    let (path, hash) = match sources {
        [one] => (one.path.clone(), one.hash.clone()),
        _ => (String::new(), String::new()),
    };
    meta.data_file_path = path;
    meta.data_file_hash = hash;
}

/// A recorded source. `rows` is `None` for a meta before `data_sources`.
struct Recorded {
    path: String,
    hash: String,
    rows: Option<usize>,
}

/// The sources `meta` records. A legacy meta with an empty hash records none.
fn recorded_sources(meta: &CheckpointMeta) -> Vec<Recorded> {
    if !meta.data_sources.is_empty() {
        return meta
            .data_sources
            .iter()
            .map(|s| Recorded {
                path: s.path.clone(),
                hash: s.hash.clone(),
                rows: Some(s.rows),
            })
            .collect();
    }
    if meta.data_file_hash.is_empty() {
        return Vec::new();
    }
    vec![Recorded {
        path: meta.data_file_path.clone(),
        hash: meta.data_file_hash.clone(),
        rows: None,
    }]
}

const RESUME_HINT: &str =
    "Resume with the original data files in their original order, or start a fresh run.";

/// Checks the run's data files `paths` against the sources `meta` records.
///
/// A legacy meta (no `hash_algo`) with one current file keeps the legacy
/// rule: a differing hash leaves the data unchecked and never blocks.
///
/// # Errors
/// Returns an error when a data file cannot be read.
pub fn check_meta_sources(
    meta: &CheckpointMeta,
    paths: &[String],
    seq_len: usize,
) -> anyhow::Result<DataCheck> {
    let recorded = recorded_sources(meta);
    let checkpoint_hash = join(recorded.iter().map(|r| r.hash.as_str()));
    if meta.hash_algo == HashAlgo::Legacy {
        if let [path] = paths {
            let hash = legacy_file_hash(path)?;
            return Ok(DataCheck {
                matches: true,
                checked: meta.data_file_hash == hash,
                checkpoint_hash: meta.data_file_hash.clone(),
                current_hash: hash,
                mismatch: None,
            });
        }
    }
    let current = source_records(paths, seq_len)?;
    let mismatch = first_mismatch(&recorded, &current, seq_len);
    Ok(DataCheck {
        matches: mismatch.is_none(),
        checked: true,
        checkpoint_hash,
        current_hash: join(current.iter().map(|c| c.hash.as_str())),
        mismatch,
    })
}

fn join<'a>(items: impl Iterator<Item = &'a str>) -> String {
    items.collect::<Vec<_>>().join(", ")
}

/// The first difference between `recorded` and `current`, with the next action.
fn first_mismatch(
    recorded: &[Recorded],
    current: &[DataSourceRecord],
    seq_len: usize,
) -> Option<String> {
    if recorded.len() != current.len() {
        return Some(format!(
            "the checkpoint trained on {} data source(s) [{}], and this run names {} [{}]. \
             {RESUME_HINT}",
            recorded.len(),
            join(recorded.iter().map(|r| r.path.as_str())),
            current.len(),
            join(current.iter().map(|c| c.path.as_str()))
        ));
    }
    for (i, (was, now)) in recorded.iter().zip(current).enumerate() {
        if was.hash != now.hash {
            let reordered = recorded.iter().any(|r| r.hash == now.hash);
            let note = if reordered {
                " The file is another source of the checkpoint: the sources are reordered."
            } else {
                ""
            };
            return Some(format!(
                "data source {i} '{}' (hash {}) differs from the checkpoint's source {i} '{}' \
                 (hash {}).{note} {RESUME_HINT}",
                now.path, now.hash, was.path, was.hash
            ));
        }
        if let Some(rows) = was.rows.filter(|&rows| rows != now.rows) {
            return Some(format!(
                "data source {i} '{}' holds {} rows of {} tokens, and the checkpoint's source \
                 {i} '{}' held {rows}. {RESUME_HINT}",
                now.path,
                now.rows,
                seq_len + 1,
                was.path
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(path: &str, hash: &str, rows: Option<usize>) -> Recorded {
        Recorded {
            path: path.to_string(),
            hash: hash.to_string(),
            rows,
        }
    }

    fn cur(path: &str, hash: &str, rows: usize) -> DataSourceRecord {
        DataSourceRecord {
            path: path.to_string(),
            hash: hash.to_string(),
            rows,
        }
    }

    #[test]
    fn equal_sources_match_and_a_moved_file_still_matches() {
        let was = [rec("a.bin", "h1", Some(5)), rec("b.bin", "h2", Some(3))];
        let now = [cur("new/a.bin", "h1", 5), cur("b.bin", "h2", 3)];
        assert_eq!(first_mismatch(&was, &now, 7), None);
    }

    #[test]
    fn each_kind_of_change_names_the_source() {
        let was = [rec("a.bin", "h1", Some(5)), rec("b.bin", "h2", Some(3))];
        let fewer = first_mismatch(&was, &[cur("a.bin", "h1", 5)], 7).expect("count");
        assert!(
            fewer.contains("2 data source(s)") && fewer.contains("names 1"),
            "{fewer}"
        );
        let swapped = [cur("b.bin", "h2", 3), cur("a.bin", "h1", 5)];
        let text = first_mismatch(&was, &swapped, 7).expect("order");
        assert!(
            text.contains("data source 0 'b.bin'") && text.contains("reordered"),
            "{text}"
        );
        let rows =
            first_mismatch(&was, &[cur("a.bin", "h1", 5), cur("b.bin", "h2", 4)], 7).expect("rows");
        assert!(
            rows.contains("data source 1 'b.bin' holds 4 rows of 8 tokens"),
            "{rows}"
        );
        assert!(rows.contains(RESUME_HINT), "{rows}");
    }

    #[test]
    fn unknown_rows_are_not_compared() {
        let was = [rec("a.bin", "h1", None)];
        assert_eq!(first_mismatch(&was, &[cur("a.bin", "h1", 99)], 7), None);
    }
}
