//! `oxidizr tokenize` end to end: ids, separators, batching, errors, the sidecar.
//!
//! - Tests without a vocabulary feature use the byte-level `tokenizer.json`
//!   from `common`. They run in the default build.
//! - The `cl100k_base` tests need `--features vocab-cl100k`.
//! - The llama3 test reads `$BOOSTR_MODELS_DIR/llama-3.2-1b/tokenizer.json`.
//!   It skips with a message when that file is absent.

mod common;

use common::{
    read_tokens_le, scratch_dir, write_byte_level_tokenizer, BYTE_LEVEL_BOS, BYTE_LEVEL_EOS,
};
use oxidizr::commands::tokenize::{FormatArg, TxtDocs, DEFAULT_BATCH_BYTES};
use oxidizr::commands::{run_tokenize, TokenizeArgs};
use oxidizr::data::{read_token_meta, token_meta_path, TokenizerRecord};
use oxidizr::data_hash::{hex, Fnv1a64};
use splintr::{AnyTokenizer, SpecialMode};
use std::path::{Path, PathBuf};

fn args(inputs: Vec<PathBuf>, output: &Path, tokenizer: &Path) -> TokenizeArgs {
    TokenizeArgs {
        inputs,
        output: output.to_path_buf(),
        tokenizer: tokenizer.to_string_lossy().into_owned(),
        format: FormatArg::Auto,
        field: "text".to_owned(),
        txt_docs: TxtDocs::File,
        allow_special: false,
        batch_bytes: DEFAULT_BATCH_BYTES,
    }
}

/// Writes one JSONL file with a `text` field per document.
fn write_jsonl(path: &Path, docs: &[&str]) {
    let lines: Vec<String> = docs
        .iter()
        .map(|doc| serde_json::json!({ "text": doc }).to_string())
        .collect();
    std::fs::write(path, lines.join("\n") + "\n").expect("write jsonl");
}

/// Each non-empty document as `encode_with(doc, mode)`, then `eos`.
fn expected(tokenizer: &AnyTokenizer, docs: &[&str], eos: u32, mode: SpecialMode) -> Vec<u32> {
    let mut out = Vec::new();
    for doc in docs.iter().filter(|doc| !doc.is_empty()) {
        out.extend(tokenizer.encode_with(doc, &mode).expect("encode"));
        out.push(eos);
    }
    out
}

fn fnv(bytes: &[u8]) -> String {
    let mut hasher = Fnv1a64::new();
    hasher.update(bytes);
    hex(hasher.finish())
}

fn load(path: &Path) -> AnyTokenizer {
    splintr::from_json_path(path).expect("load tokenizer.json")
}

const DOCS: [&str; 5] = [
    "the theory of things",
    "",
    "héllo wörld, multi-byte 🚀 text\nwith a second line",
    "  leading spaces and trailing tabs\t\t",
    "x",
];

#[test]
fn ids_are_encode_with_then_the_separator() {
    let dir = scratch_dir("tokenize-ids");
    let tok_path = write_byte_level_tokenizer(&dir, false);
    let input = dir.join("docs.jsonl");
    write_jsonl(&input, &DOCS);
    let output = dir.join("out.bin");
    run_tokenize(args(vec![input], &output, &tok_path)).expect("tokenize");

    let want = expected(
        &load(&tok_path),
        &DOCS,
        BYTE_LEVEL_EOS,
        SpecialMode::Ordinary,
    );
    assert_eq!(read_tokens_le(&output), want);
    let meta = read_token_meta(&output).expect("read").expect("a sidecar");
    assert_eq!(meta.num_documents, 4);
    assert_eq!(meta.skipped_empty_documents, 1);
    assert_eq!(meta.separator_id, BYTE_LEVEL_EOS);
    assert_eq!(meta.bos_id, None);
}

#[test]
fn batch_bytes_does_not_change_the_output() {
    let dir = scratch_dir("tokenize-batches");
    let tok_path = write_byte_level_tokenizer(&dir, false);
    let large = "the large document ".repeat(60_000);
    let mut docs: Vec<String> = (0..300)
        .map(|i| format!("document {i}: the {i} things"))
        .collect();
    docs.insert(150, large);
    let doc_refs: Vec<&str> = docs.iter().map(String::as_str).collect();
    let input = dir.join("docs.jsonl");
    write_jsonl(&input, &doc_refs);

    let small = dir.join("small.bin");
    let mut small_args = args(vec![input.clone()], &small, &tok_path);
    small_args.batch_bytes = 1;
    run_tokenize(small_args).expect("tokenize with 1-byte batches");
    let big = dir.join("big.bin");
    run_tokenize(args(vec![input], &big, &tok_path)).expect("tokenize with the default");

    let small_bytes = std::fs::read(&small).expect("read");
    assert!(
        small_bytes == std::fs::read(&big).expect("read"),
        "the files differ"
    );
    let want = expected(
        &load(&tok_path),
        &doc_refs,
        BYTE_LEVEL_EOS,
        SpecialMode::Ordinary,
    );
    assert_eq!(read_tokens_le(&small), want);
    let small_meta = read_token_meta(&small).expect("read").expect("sidecar");
    assert_eq!(Some(small_meta), read_token_meta(&big).expect("read"));
}

#[test]
fn a_bad_jsonl_line_names_file_and_line_and_leaves_no_output() {
    let dir = scratch_dir("tokenize-bad-line");
    let tok_path = write_byte_level_tokenizer(&dir, false);
    let input = dir.join("docs.jsonl");
    std::fs::write(&input, "{\"text\":\"a\"}\n\n{\"text\": 3}\n").expect("write");
    let output = dir.join("out.bin");
    let message = run_tokenize(args(vec![input.clone()], &output, &tok_path))
        .expect_err("a non-string field must fail")
        .to_string();
    assert!(
        message.contains(&format!("{}:3:", input.display())),
        "{message}"
    );
    assert!(!output.exists());
    assert!(!dir.join("out.bin.partial").exists());
    assert!(!token_meta_path(&output).exists());

    std::fs::write(&input, "{\"text\":\"a\"}\nnot json\n").expect("write");
    let message = run_tokenize(args(vec![input.clone()], &output, &tok_path))
        .expect_err("invalid json must fail")
        .to_string();
    assert!(
        message.contains(&format!("{}:2:", input.display())),
        "{message}"
    );
}

#[test]
fn special_token_text_is_ordinary_unless_allowed() {
    let dir = scratch_dir("tokenize-special");
    let tok_path = write_byte_level_tokenizer(&dir, false);
    let docs = ["before<|endoftext|>after"];
    let input = dir.join("docs.jsonl");
    write_jsonl(&input, &docs);
    let tokenizer = load(&tok_path);

    let ordinary = dir.join("ordinary.bin");
    run_tokenize(args(vec![input.clone()], &ordinary, &tok_path)).expect("tokenize");
    let ids = read_tokens_le(&ordinary);
    assert_eq!(
        ids,
        expected(&tokenizer, &docs, BYTE_LEVEL_EOS, SpecialMode::Ordinary)
    );
    assert_eq!(ids.iter().filter(|&&id| id == BYTE_LEVEL_EOS).count(), 1);

    let allowed = dir.join("allowed.bin");
    let mut allow = args(vec![input], &allowed, &tok_path);
    allow.allow_special = true;
    run_tokenize(allow).expect("tokenize");
    let ids = read_tokens_le(&allowed);
    assert_eq!(
        ids,
        expected(&tokenizer, &docs, BYTE_LEVEL_EOS, SpecialMode::All)
    );
    assert_eq!(ids.iter().filter(|&&id| id == BYTE_LEVEL_EOS).count(), 2);
}

#[test]
fn the_sidecar_describes_the_file() {
    let dir = scratch_dir("tokenize-sidecar");
    let tok_path = write_byte_level_tokenizer(&dir, false);
    let input = dir.join("docs.jsonl");
    write_jsonl(&input, &DOCS);
    let output = dir.join("out.bin");
    run_tokenize(args(vec![input.clone()], &output, &tok_path)).expect("tokenize");

    let bytes = std::fs::read(&output).expect("read");
    let ids = read_tokens_le(&output);
    let meta = read_token_meta(&output).expect("read").expect("sidecar");
    assert_eq!(token_meta_path(&output), dir.join("out.bin.tokens.json"));
    assert_eq!(meta.format_version, 1);
    assert!(!meta.splintr_version.is_empty());
    assert_eq!(meta.num_tokens, ids.len() as u64);
    assert_eq!(meta.max_token_id, ids.iter().copied().max());
    assert_eq!(meta.file_hash, fnv(&bytes));
    assert_eq!(meta.tokenizer_vocab_size, 259);
    let tok_bytes = std::fs::read(&tok_path).expect("read tokenizer");
    let want = TokenizerRecord::File {
        path: tok_path.display().to_string(),
        hash: fnv(&tok_bytes),
    };
    assert_eq!(meta.tokenizer, want);
    let input_bytes = std::fs::read(&input).expect("read input");
    assert_eq!(meta.inputs.len(), 1);
    assert_eq!(meta.inputs[0].path, input.display().to_string());
    assert_eq!(meta.inputs[0].format, "jsonl");
    assert_eq!(meta.inputs[0].bytes, input_bytes.len() as u64);
    assert_eq!(meta.inputs[0].hash, fnv(&input_bytes));
    assert_eq!(meta.options.field, "text");
    assert!(!meta.options.allow_special);
}

#[test]
fn directory_inputs_are_read_in_path_byte_order() {
    let dir = scratch_dir("tokenize-walk");
    let tok_path = write_byte_level_tokenizer(&dir, false);
    let corpus = dir.join("corpus");
    std::fs::create_dir_all(corpus.join("a")).expect("create dirs");
    let files = [
        ("b.txt", "bee"),
        ("a/c.txt", "sea"),
        ("A.txt", "ay"),
        ("skip.md", "no"),
    ];
    for (name, text) in files {
        std::fs::write(corpus.join(name), text).expect("write");
    }
    let output = dir.join("out.bin");
    run_tokenize(args(vec![corpus.clone()], &output, &tok_path)).expect("tokenize");

    let want = expected(
        &load(&tok_path),
        &["ay", "sea", "bee"],
        BYTE_LEVEL_EOS,
        SpecialMode::Ordinary,
    );
    assert_eq!(read_tokens_le(&output), want);
    let meta = read_token_meta(&output).expect("read").expect("sidecar");
    let order: Vec<String> = meta.inputs.iter().map(|i| i.path.clone()).collect();
    let want_order: Vec<String> = ["A.txt", "a/c.txt", "b.txt"]
        .iter()
        .map(|name| corpus.join(name).display().to_string())
        .collect();
    assert_eq!(order, want_order);
}

#[test]
fn blank_lines_split_text_documents() {
    let dir = scratch_dir("tokenize-blank-line");
    let tok_path = write_byte_level_tokenizer(&dir, false);
    let input = dir.join("notes.txt");
    std::fs::write(&input, "first doc\nline two\n\n\nsecond doc\n").expect("write");
    let output = dir.join("out.bin");
    let mut blank = args(vec![input], &output, &tok_path);
    blank.txt_docs = TxtDocs::BlankLine;
    run_tokenize(blank).expect("tokenize");
    let docs = ["first doc\nline two", "second doc"];
    let want = expected(
        &load(&tok_path),
        &docs,
        BYTE_LEVEL_EOS,
        SpecialMode::Ordinary,
    );
    assert_eq!(read_tokens_le(&output), want);
}

#[test]
fn a_template_bos_opens_every_document() {
    let dir = scratch_dir("tokenize-bos");
    let tok_path = write_byte_level_tokenizer(&dir, true);
    let docs = ["one", "two", "three"];
    let input = dir.join("docs.jsonl");
    write_jsonl(&input, &docs);
    let output = dir.join("out.bin");
    run_tokenize(args(vec![input], &output, &tok_path)).expect("tokenize");

    let ids = read_tokens_le(&output);
    assert_eq!(
        ids,
        expected(
            &load(&tok_path),
            &docs,
            BYTE_LEVEL_EOS,
            SpecialMode::Ordinary
        )
    );
    assert_bos_after_every_separator(&ids, BYTE_LEVEL_BOS, BYTE_LEVEL_EOS, docs.len());
    let meta = read_token_meta(&output).expect("read").expect("sidecar");
    assert_eq!(meta.bos_id, Some(BYTE_LEVEL_BOS));
}

/// `ids` opens with `bos`, ends with `eos`, and holds `docs` documents.
fn assert_bos_after_every_separator(ids: &[u32], bos: u32, eos: u32, docs: usize) {
    assert_eq!(ids.first(), Some(&bos));
    assert_eq!(ids.last(), Some(&eos));
    let separators: Vec<usize> = (0..ids.len()).filter(|&i| ids[i] == eos).collect();
    assert_eq!(separators.len(), docs);
    for &at in &separators[..docs - 1] {
        assert_eq!(ids[at + 1], bos, "no BOS after the separator at {at}");
    }
}

#[cfg(not(feature = "vocab-cl100k"))]
#[test]
fn an_unbundled_name_names_the_missing_feature() {
    let dir = scratch_dir("tokenize-unbundled");
    let input = dir.join("a.txt");
    std::fs::write(&input, "text").expect("write");
    let mut named = args(vec![input], &dir.join("out.bin"), Path::new("x"));
    named.tokenizer = "cl100k_base".to_owned();
    let message = run_tokenize(named)
        .expect_err("no bundled vocabulary")
        .to_string();
    assert!(message.contains("vocab-cl100k"), "{message}");
}

#[cfg(feature = "vocab-cl100k")]
mod cl100k {
    use super::*;

    const EOS: u32 = 100257;

    fn named_args(inputs: Vec<PathBuf>, output: &Path) -> TokenizeArgs {
        let mut named = args(inputs, output, Path::new("x"));
        named.tokenizer = "cl100k_base".to_owned();
        named
    }

    fn tokenizer() -> AnyTokenizer {
        splintr::from_pretrained("cl100k_base").expect("cl100k_base is bundled")
    }

    #[test]
    fn ids_are_encode_with_then_eos() {
        let dir = scratch_dir("tokenize-cl100k-ids");
        let input = dir.join("docs.jsonl");
        write_jsonl(&input, &DOCS);
        let output = dir.join("out.bin");
        run_tokenize(named_args(vec![input], &output)).expect("tokenize");
        let want = expected(&tokenizer(), &DOCS, EOS, SpecialMode::Ordinary);
        assert_eq!(read_tokens_le(&output), want);
        let meta = read_token_meta(&output).expect("read").expect("sidecar");
        assert_eq!(meta.separator_id, EOS);
        assert_eq!(meta.bos_id, None);
        let name = TokenizerRecord::Name {
            name: "cl100k_base".to_owned(),
        };
        assert_eq!(meta.tokenizer, name);
    }

    #[test]
    fn special_token_text_is_ordinary_unless_allowed() {
        let dir = scratch_dir("tokenize-cl100k-special");
        let docs = ["a <|endoftext|> b"];
        let input = dir.join("docs.jsonl");
        write_jsonl(&input, &docs);
        let ordinary = dir.join("ordinary.bin");
        run_tokenize(named_args(vec![input.clone()], &ordinary)).expect("tokenize");
        let ids = read_tokens_le(&ordinary);
        assert_eq!(ids.iter().filter(|&&id| id == EOS).count(), 1, "{ids:?}");
        assert_eq!(
            ids,
            expected(&tokenizer(), &docs, EOS, SpecialMode::Ordinary)
        );

        let allowed = dir.join("allowed.bin");
        let mut allow = named_args(vec![input], &allowed);
        allow.allow_special = true;
        run_tokenize(allow).expect("tokenize");
        let ids = read_tokens_le(&allowed);
        assert_eq!(ids.iter().filter(|&&id| id == EOS).count(), 2, "{ids:?}");
    }

    #[test]
    fn batch_bytes_does_not_change_the_output() {
        let dir = scratch_dir("tokenize-cl100k-batches");
        let docs: Vec<String> = (0..500)
            .map(|i| format!("Document {i} says {}.", i * 7))
            .collect();
        let doc_refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let input = dir.join("docs.jsonl");
        write_jsonl(&input, &doc_refs);
        let small = dir.join("small.bin");
        let mut small_args = named_args(vec![input.clone()], &small);
        small_args.batch_bytes = 7;
        run_tokenize(small_args).expect("tokenize");
        let big = dir.join("big.bin");
        run_tokenize(named_args(vec![input], &big)).expect("tokenize");
        assert!(std::fs::read(&small).expect("read") == std::fs::read(&big).expect("read"));
    }
}

#[test]
fn a_llama3_tokenizer_json_places_bos_and_eos() {
    let Some(models) = std::env::var_os("BOOSTR_MODELS_DIR") else {
        eprintln!("skipped: BOOSTR_MODELS_DIR is unset, so no llama-3.2-1b tokenizer.json");
        return;
    };
    let tok_path = PathBuf::from(models)
        .join("llama-3.2-1b")
        .join("tokenizer.json");
    if !tok_path.is_file() {
        eprintln!("skipped: {} is absent", tok_path.display());
        return;
    }
    let dir = scratch_dir("tokenize-llama3");
    let docs = [
        "Hello world.",
        "Second document.",
        "Third one, longer than the rest.",
    ];
    let input = dir.join("docs.jsonl");
    write_jsonl(&input, &docs);
    let output = dir.join("out.bin");
    run_tokenize(args(vec![input], &output, &tok_path)).expect("tokenize");

    let ids = read_tokens_le(&output);
    assert_eq!(
        ids,
        expected(&load(&tok_path), &docs, 128001, SpecialMode::Ordinary)
    );
    assert_bos_after_every_separator(&ids, 128000, 128001, docs.len());
    let meta = read_token_meta(&output).expect("read").expect("sidecar");
    assert_eq!(meta.bos_id, Some(128000));
    assert_eq!(meta.separator_id, 128001);
}
