//! Input discovery for `oxidizr tokenize`: files, directory walks, formats.
//!
//! - A file named by `-i` is always read.
//! - A directory is walked recursively for `.txt` and `.jsonl` files.
//!   Symlinked directories are not entered.
//! - Files from one directory are sorted by path bytes. `-i` order holds
//!   between arguments.

use super::args::FormatArg;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Extensions a directory walk keeps.
const WALK_EXTENSIONS: [&str; 2] = ["txt", "jsonl"];

/// How one input file is read.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FileFormat {
    /// Plain UTF-8 text.
    Txt,
    /// One JSON object per line.
    Jsonl,
}

impl FileFormat {
    /// The name the sidecar records.
    pub fn name(self) -> &'static str {
        match self {
            Self::Txt => "txt",
            Self::Jsonl => "jsonl",
        }
    }
}

/// One input file and its format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputFile {
    /// The path as discovered.
    pub path: PathBuf,
    /// How the file is read.
    pub format: FileFormat,
}

/// Every input file of `inputs`, in read order.
///
/// # Errors
/// Returns an error for a missing input, a file whose format `auto` cannot
/// tell, a file listed twice, the output among the inputs, or no files at all.
pub fn discover(inputs: &[PathBuf], format: FormatArg, output: &Path) -> Result<Vec<InputFile>> {
    let mut files = Vec::new();
    for input in inputs {
        let meta = fs::metadata(input).map_err(|e| {
            anyhow!(
                "input '{}' could not be read: {e}. Pass an existing file or directory to -i.",
                input.display()
            )
        })?;
        if meta.is_dir() {
            let mut found = Vec::new();
            walk(input, &mut found)?;
            found.sort_by(|a, b| {
                a.as_os_str()
                    .as_encoded_bytes()
                    .cmp(b.as_os_str().as_encoded_bytes())
            });
            for path in found {
                let format = file_format(&path, format)?;
                files.push(InputFile { path, format });
            }
        } else {
            let format = file_format(input, format)?;
            files.push(InputFile {
                path: input.clone(),
                format,
            });
        }
    }
    if files.is_empty() {
        return Err(anyhow!(
            "no .txt or .jsonl files under the -i inputs. Pass files, or directories that hold them."
        ));
    }
    reject_repeats(&files, output)?;
    Ok(files)
}

/// Collects the walk files under `dir`, in no particular order.
fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let unreadable = |path: &Path, e: std::io::Error| {
        anyhow!(
            "'{}' could not be listed: {e}. Check its permissions.",
            path.display()
        )
    };
    for entry in fs::read_dir(dir).map_err(|e| unreadable(dir, e))? {
        let entry = entry.map_err(|e| unreadable(dir, e))?;
        let path = entry.path();
        let kind = entry.file_type().map_err(|e| unreadable(&path, e))?;
        if kind.is_dir() {
            walk(&path, out)?;
        } else if has_walk_extension(&path) && path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn has_walk_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| WALK_EXTENSIONS.iter().any(|w| ext.eq_ignore_ascii_case(w)))
}

/// The format of `path` under `--format`.
fn file_format(path: &Path, format: FormatArg) -> Result<FileFormat> {
    match format {
        FormatArg::Txt => return Ok(FileFormat::Txt),
        FormatArg::Jsonl => return Ok(FileFormat::Jsonl),
        FormatArg::Auto => {}
    }
    let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("txt") {
        Ok(FileFormat::Txt)
    } else if ext.eq_ignore_ascii_case("jsonl") {
        Ok(FileFormat::Jsonl)
    } else {
        Err(anyhow!(
            "--format auto cannot tell the format of '{}' from its extension. Pass \
             --format txt or --format jsonl.",
            path.display()
        ))
    }
}

/// Refuses a file read twice, and the output path among the inputs.
fn reject_repeats(files: &[InputFile], output: &Path) -> Result<()> {
    let mut seen: HashMap<PathBuf, &Path> = HashMap::new();
    for file in files {
        let real = fs::canonicalize(&file.path).map_err(|e| {
            anyhow!(
                "input '{}' could not be resolved: {e}. Check the path and its permissions.",
                file.path.display()
            )
        })?;
        if let Some(first) = seen.insert(real, &file.path) {
            return Err(anyhow!(
                "input '{}' is listed twice (also as '{}'). Remove one of the -i values.",
                file.path.display(),
                first.display()
            ));
        }
    }
    if let Ok(real_output) = fs::canonicalize(output) {
        if let Some(input) = seen.get(&real_output) {
            return Err(anyhow!(
                "output '{}' is also input '{}'. Pass another -o path.",
                output.display(),
                input.display()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "oxidizr-tokenize-input-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn touch(path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent");
        }
        fs::write(path, b"x").expect("write file");
    }

    #[test]
    fn a_walk_is_recursive_filtered_and_sorted_by_path_bytes() {
        let dir = scratch_dir("walk");
        for name in [
            "b.txt",
            "a/z.jsonl",
            "a/b.TXT",
            "B.txt",
            "notes.md",
            "c/d/e.txt",
        ] {
            touch(&dir.join(name));
        }
        let files = discover(
            std::slice::from_ref(&dir),
            FormatArg::Auto,
            &dir.join("out.bin"),
        )
        .expect("walk");
        let names: Vec<String> = files
            .iter()
            .map(|f| {
                f.path
                    .strip_prefix(&dir)
                    .expect("under dir")
                    .display()
                    .to_string()
            })
            .collect();
        assert_eq!(
            names,
            ["B.txt", "a/b.TXT", "a/z.jsonl", "b.txt", "c/d/e.txt"]
        );
        assert_eq!(files[2].format, FileFormat::Jsonl);
        assert_eq!(files[1].format, FileFormat::Txt);
        fs::remove_dir_all(&dir).expect("remove scratch dir");
    }

    #[test]
    fn argument_order_holds_between_inputs() {
        let dir = scratch_dir("order");
        touch(&dir.join("z.txt"));
        touch(&dir.join("a.txt"));
        let inputs = [dir.join("z.txt"), dir.join("a.txt")];
        let files = discover(&inputs, FormatArg::Auto, &dir.join("out.bin")).expect("discover");
        assert_eq!(files[0].path, dir.join("z.txt"));
        assert_eq!(files[1].path, dir.join("a.txt"));
        fs::remove_dir_all(&dir).expect("remove scratch dir");
    }

    #[test]
    fn auto_refuses_an_unknown_extension_by_path() {
        let dir = scratch_dir("ext");
        touch(&dir.join("data.csv"));
        let message = discover(&[dir.join("data.csv")], FormatArg::Auto, &dir.join("o.bin"))
            .expect_err("csv must fail under auto")
            .to_string();
        assert!(
            message.contains("data.csv") && message.contains("--format"),
            "{message}"
        );
        let forced = discover(&[dir.join("data.csv")], FormatArg::Txt, &dir.join("o.bin"));
        assert_eq!(forced.expect("forced txt")[0].format, FileFormat::Txt);
        fs::remove_dir_all(&dir).expect("remove scratch dir");
    }

    #[test]
    fn a_file_listed_twice_is_refused() {
        let dir = scratch_dir("twice");
        touch(&dir.join("a.txt"));
        let message = discover(
            &[dir.join("a.txt"), dir.clone()],
            FormatArg::Auto,
            &dir.join("o.bin"),
        )
        .expect_err("a repeat must fail")
        .to_string();
        assert!(message.contains("listed twice"), "{message}");
        fs::remove_dir_all(&dir).expect("remove scratch dir");
    }

    #[test]
    fn the_output_cannot_be_an_input() {
        let dir = scratch_dir("overlap");
        touch(&dir.join("a.txt"));
        let message = discover(&[dir.join("a.txt")], FormatArg::Auto, &dir.join("a.txt"))
            .expect_err("overlap must fail")
            .to_string();
        assert!(message.contains("is also input"), "{message}");
        fs::remove_dir_all(&dir).expect("remove scratch dir");
    }

    #[test]
    fn missing_and_empty_inputs_are_refused() {
        let dir = scratch_dir("empty");
        let missing = discover(&[dir.join("nope.txt")], FormatArg::Auto, &dir.join("o.bin"));
        assert!(missing
            .expect_err("missing")
            .to_string()
            .contains("nope.txt"));
        touch(&dir.join("readme.md"));
        let empty = discover(
            std::slice::from_ref(&dir),
            FormatArg::Auto,
            &dir.join("o.bin"),
        );
        assert!(empty
            .expect_err("empty")
            .to_string()
            .contains("no .txt or .jsonl"));
        fs::remove_dir_all(&dir).expect("remove scratch dir");
    }
}
