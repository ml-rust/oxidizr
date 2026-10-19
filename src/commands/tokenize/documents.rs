//! Document reading for `oxidizr tokenize`.
//!
//! - Text, `--txt-docs file`: the whole file is one document.
//! - Text, `--txt-docs blank-line`: a line of only whitespace ends a
//!   document. A document keeps its lines' bytes, minus the last line ending.
//! - JSONL: each non-blank line is one object. `--field` holds the text.
//!
//! Each file is hashed while it is read, so every byte is read once.

use super::args::TxtDocs;
use super::input::{FileFormat, InputFile};
use crate::data::InputRecord;
use crate::data_hash::{hex, Fnv1a64};
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

/// The callback that receives each document.
pub type Emit<'a> = dyn FnMut(String) -> Result<()> + 'a;

/// How documents are cut out of the input files.
#[derive(Clone, Debug)]
pub struct ReadOptions {
    /// The JSONL field that holds the text.
    pub field: String,
    /// Where a text-file document ends.
    pub txt_docs: TxtDocs,
}

/// A reader that hashes and counts every byte it returns.
struct HashingReader<R> {
    inner: R,
    hasher: Fnv1a64,
    bytes: u64,
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
}

/// Calls `emit` with each document of `file`, in file order.
///
/// Empty documents reach `emit` too. The caller counts and skips them.
///
/// # Errors
/// Returns an error naming `file:line` for invalid UTF-8, invalid JSON, a
/// line that is not an object, or a missing or non-string field. An error
/// from `emit` stops the read and returns as is.
pub fn read_documents(
    file: &InputFile,
    options: &ReadOptions,
    emit: &mut Emit<'_>,
) -> Result<InputRecord> {
    let opened = File::open(&file.path).map_err(|e| {
        anyhow!(
            "input '{}' could not be opened: {e}. Check its permissions.",
            file.path.display()
        )
    })?;
    let mut reader = HashingReader {
        inner: opened,
        hasher: Fnv1a64::new(),
        bytes: 0,
    };
    let shown = file.path.display().to_string();
    match (file.format, options.txt_docs) {
        (FileFormat::Txt, TxtDocs::File) => read_whole(&mut reader, &shown, emit)?,
        (FileFormat::Txt, TxtDocs::BlankLine) => {
            read_paragraphs(BufReader::new(&mut reader), &shown, emit)?
        }
        (FileFormat::Jsonl, _) => {
            read_jsonl(BufReader::new(&mut reader), &shown, &options.field, emit)?
        }
    }
    Ok(InputRecord {
        path: shown,
        format: file.format.name().to_owned(),
        bytes: reader.bytes,
        hash: hex(reader.hasher.finish()),
    })
}

fn io_error(path: &str, e: std::io::Error) -> anyhow::Error {
    anyhow!("input '{path}' could not be read: {e}. Check the file and the disk.")
}

fn read_whole(reader: &mut impl Read, path: &str, emit: &mut Emit<'_>) -> Result<()> {
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|e| io_error(path, e))?;
    let text = String::from_utf8(bytes).map_err(|e| {
        anyhow!(
            "input '{path}' is not valid UTF-8 at byte {}. Convert the file to UTF-8.",
            e.utf8_error().valid_up_to()
        )
    })?;
    emit(text)
}

/// Each line of `reader` with its 1-based number, line ending kept.
fn for_each_line(
    mut reader: impl BufRead,
    path: &str,
    mut visit: impl FnMut(usize, &str) -> Result<()>,
) -> Result<()> {
    let mut buf = Vec::new();
    let mut number = 0usize;
    loop {
        buf.clear();
        let n = reader
            .read_until(b'\n', &mut buf)
            .map_err(|e| io_error(path, e))?;
        if n == 0 {
            return Ok(());
        }
        number += 1;
        let line = std::str::from_utf8(&buf).map_err(|e| {
            anyhow!(
                "{path}:{number}: not valid UTF-8 at byte {} of the line. Convert the file \
                 to UTF-8.",
                e.valid_up_to()
            )
        })?;
        visit(number, line)?;
    }
}

fn read_paragraphs(reader: impl BufRead, path: &str, emit: &mut Emit<'_>) -> Result<()> {
    let mut doc = String::new();
    for_each_line(reader, path, |_, line| {
        if line.trim().is_empty() {
            if !doc.is_empty() {
                emit(take_document(&mut doc))?;
            }
        } else {
            doc.push_str(line);
        }
        Ok(())
    })?;
    if !doc.is_empty() {
        emit(take_document(&mut doc))?;
    }
    Ok(())
}

/// Takes `doc` without its last line ending.
fn take_document(doc: &mut String) -> String {
    if doc.ends_with('\n') {
        doc.pop();
        if doc.ends_with('\r') {
            doc.pop();
        }
    }
    std::mem::take(doc)
}

fn read_jsonl(reader: impl BufRead, path: &str, field: &str, emit: &mut Emit<'_>) -> Result<()> {
    for_each_line(reader, path, |number, line| {
        if line.trim().is_empty() {
            return Ok(());
        }
        let value: Value = serde_json::from_str(line).map_err(|e| {
            anyhow!("{path}:{number}: not valid JSON: {e}. Fix or remove that line.")
        })?;
        let mut object = match value {
            Value::Object(object) => object,
            other => {
                return Err(anyhow!(
                    "{path}:{number}: a JSONL line must be a JSON object, found {}. Fix or \
                     remove that line.",
                    kind(&other)
                ))
            }
        };
        match object.remove(field) {
            Some(Value::String(text)) => emit(text),
            Some(other) => Err(anyhow!(
                "{path}:{number}: field \"{field}\" is {}, not a string. Fix that line, or pass \
                 --field with the field that holds the text.",
                kind(&other)
            )),
            None => Err(anyhow!(
                "{path}:{number}: no \"{field}\" field. Pass --field with the field that \
                 holds the text."
            )),
        }
    })
}

fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn docs_of(read: impl FnOnce(&mut Emit<'_>) -> Result<()>) -> Result<Vec<String>> {
        let mut docs = Vec::new();
        read(&mut |doc| {
            docs.push(doc);
            Ok(())
        })?;
        Ok(docs)
    }

    #[test]
    fn blank_lines_end_documents_and_last_line_endings_drop() {
        let text = "one\r\ntwo\n\n  \nthree\n\n\nfour";
        let docs = docs_of(|emit| read_paragraphs(Cursor::new(text), "t.txt", emit)).expect("read");
        assert_eq!(docs, ["one\r\ntwo", "three", "four"]);
    }

    #[test]
    fn jsonl_reads_the_field_and_skips_blank_lines() {
        let text = concat!(
            "{\"text\":\"a\",\"id\":1}\n",
            "\n",
            "{\"text\":\"\"}\n",
            "{\"body\":\"x\",\"text\":\"b\\n\"}\n"
        );
        let docs = docs_of(|emit| read_jsonl(Cursor::new(text), "d.jsonl", "text", emit));
        let docs = docs.expect("read");
        assert_eq!(docs, ["a", "", "b\n"]);
    }

    #[test]
    fn a_bad_jsonl_line_is_named_by_file_and_line() {
        let text = "{\"text\":\"a\"}\n{\"text\": oops}\n";
        let message = docs_of(|emit| read_jsonl(Cursor::new(text), "d.jsonl", "text", emit))
            .expect_err("bad json must fail")
            .to_string();
        assert!(message.starts_with("d.jsonl:2:"), "{message}");
    }

    #[test]
    fn non_string_missing_and_non_object_lines_are_named() {
        let cases = [
            ("{\"text\": 5}\n", "d.jsonl:1: field \"text\" is a number"),
            ("{\"body\": \"x\"}\n", "d.jsonl:1: no \"text\" field"),
            (
                "[1, 2]\n",
                "d.jsonl:1: a JSONL line must be a JSON object, found an array",
            ),
        ];
        for (text, want) in cases {
            let message = docs_of(|emit| read_jsonl(Cursor::new(text), "d.jsonl", "text", emit))
                .expect_err("must fail")
                .to_string();
            assert!(message.starts_with(want), "{message}");
        }
    }

    #[test]
    fn invalid_utf8_is_named_by_file_and_line() {
        let bytes: &[u8] = b"ok\n\xff\xfe\n";
        let message = docs_of(|emit| read_paragraphs(Cursor::new(bytes), "t.txt", emit))
            .expect_err("bad utf-8 must fail")
            .to_string();
        assert!(message.starts_with("t.txt:2:"), "{message}");
    }

    #[test]
    fn a_whole_file_is_one_document_and_is_hashed() {
        let mut reader = HashingReader {
            inner: Cursor::new(b"line one\nline two\n".to_vec()),
            hasher: Fnv1a64::new(),
            bytes: 0,
        };
        let docs = docs_of(|emit| read_whole(&mut reader, "t.txt", emit)).expect("read");
        assert_eq!(docs, ["line one\nline two\n"]);
        let mut want = Fnv1a64::new();
        want.update(b"line one\nline two\n");
        assert_eq!(reader.hasher.finish(), want.finish());
        assert_eq!(reader.bytes, 18);
    }
}
