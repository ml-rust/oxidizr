//! The `oxidizr tokenize` pipeline: read, encode, write, then the sidecar.
//!
//! A reader thread cuts documents into batches of at most `--batch-bytes` of
//! text. The calling thread encodes each batch and appends its ids. The
//! channel holds no batch, so at most two batches exist at once: the one the
//! reader fills and the one being encoded.

use super::args::TokenizeArgs;
use super::documents::{read_documents, ReadOptions};
use super::encode::DocEncoder;
use super::input::{discover, InputFile};
use super::output::TokenWriter;
use crate::data::{write_token_meta, InputRecord, TokenMeta, TokenOptions, TOKEN_META_VERSION};
use crate::data_hash::HashAlgo;
use crate::tokenizer::load_tokenizer;
use anyhow::{anyhow, Result};
use std::sync::mpsc::{sync_channel, SyncSender};

/// What the reader thread saw.
struct ReadSummary {
    inputs: Vec<InputRecord>,
    num_documents: u64,
    skipped_empty: u64,
}

/// Runs `oxidizr tokenize`.
///
/// # Errors
/// Returns the first error of tokenizer loading, input discovery, reading,
/// encoding, or writing. No output file or sidecar changes on an error
/// before the commit.
pub fn run_tokenize(args: TokenizeArgs) -> Result<()> {
    let loaded = load_tokenizer(&args.tokenizer)?;
    let files = discover(&args.inputs, args.format, &args.output)?;
    let options = ReadOptions {
        field: args.field.clone(),
        txt_docs: args.txt_docs,
    };
    let encoder = DocEncoder::new(&loaded, args.allow_special);
    let mut writer = TokenWriter::create(&args.output)?;

    let summary = std::thread::scope(|scope| -> Result<ReadSummary> {
        let (sender, receiver) = sync_channel::<Vec<String>>(0);
        let reader = scope.spawn(|| read_batches(&files, &options, args.batch_bytes, sender));
        let mut ids = Vec::new();
        let mut pipeline = Ok(());
        for batch in receiver {
            ids.clear();
            pipeline = encoder
                .encode_into(&batch, &mut ids)
                .and_then(|()| writer.append(&ids));
            if pipeline.is_err() {
                break;
            }
        }
        let read = reader
            .join()
            .map_err(|_| anyhow!("the input reader thread stopped. Run the command again."))?;
        pipeline?;
        read
    })?;

    let written = writer.commit()?;
    let meta = TokenMeta {
        format_version: TOKEN_META_VERSION,
        oxidizr_version: env!("CARGO_PKG_VERSION").to_owned(),
        splintr_version: env!("OXIDIZR_SPLINTR_VERSION").to_owned(),
        tokenizer: loaded.record.clone(),
        separator_id: loaded.eos_id,
        bos_id: loaded.bos_id,
        tokenizer_vocab_size: loaded.vocab_size,
        max_token_id: written.max_token_id,
        num_tokens: written.num_tokens,
        num_documents: summary.num_documents,
        skipped_empty_documents: summary.skipped_empty,
        hash_algo: HashAlgo::Fnv1a64,
        file_hash: written.file_hash,
        inputs: summary.inputs,
        options: TokenOptions {
            format: args.format.name().to_owned(),
            field: args.field.clone(),
            txt_docs: args.txt_docs.name().to_owned(),
            allow_special: args.allow_special,
        },
    };
    write_token_meta(&args.output, &meta)?;

    println!(
        "Wrote {} tokens from {} documents to {} ({} empty documents skipped).",
        meta.num_tokens,
        meta.num_documents,
        args.output.display(),
        meta.skipped_empty_documents
    );
    Ok(())
}

/// Reads every file into batches of at most `batch_bytes` of text.
///
/// A document above `batch_bytes` is a batch of its own.
///
/// A send error means the encoder stopped. Its own error is the one reported.
fn read_batches(
    files: &[InputFile],
    options: &ReadOptions,
    batch_bytes: usize,
    sender: SyncSender<Vec<String>>,
) -> Result<ReadSummary> {
    let mut summary = ReadSummary {
        inputs: Vec::with_capacity(files.len()),
        num_documents: 0,
        skipped_empty: 0,
    };
    let mut batch: Vec<String> = Vec::new();
    let mut filled = 0usize;
    let stopped = || anyhow!("the encoder stopped before the inputs ended.");
    for file in files {
        let record = read_documents(file, options, &mut |doc| {
            if doc.is_empty() {
                summary.skipped_empty += 1;
                return Ok(());
            }
            summary.num_documents += 1;
            if !batch.is_empty() && filled + doc.len() > batch_bytes {
                filled = 0;
                sender
                    .send(std::mem::take(&mut batch))
                    .map_err(|_| stopped())?;
            }
            filled += doc.len();
            batch.push(doc);
            Ok(())
        })?;
        summary.inputs.push(record);
    }
    if !batch.is_empty() {
        sender.send(batch).map_err(|_| stopped())?;
    }
    Ok(summary)
}
