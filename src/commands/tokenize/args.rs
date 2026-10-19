//! Arguments of `oxidizr tokenize`.

use clap::{Args, ValueEnum};
use std::path::PathBuf;

/// Default `--batch-bytes`: 64 MiB of document text per batch.
pub const DEFAULT_BATCH_BYTES: usize = 64 << 20;

/// How an input file is read.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum FormatArg {
    /// From the extension: `.txt` is text, `.jsonl` is JSON lines.
    Auto,
    /// Plain UTF-8 text, split per `--txt-docs`.
    Txt,
    /// One JSON object per line. `--field` holds the document text.
    Jsonl,
}

impl FormatArg {
    /// The value as the command line spells it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Txt => "txt",
            Self::Jsonl => "jsonl",
        }
    }
}

/// Where one text-file document ends.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum TxtDocs {
    /// The whole file is one document.
    File,
    /// A line of only whitespace ends a document.
    BlankLine,
}

impl TxtDocs {
    /// The value as the command line spells it.
    pub fn name(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::BlankLine => "blank-line",
        }
    }
}

/// `oxidizr tokenize` arguments.
#[derive(Args, Debug, Clone)]
#[command(
    long_about = "Tokenize text into a packed u32 token file for `oxidizr train -d`.

Each document becomes the tokenizer's BOS (when it has one), the content ids,
then the EOS id. Text that spells a special token stays ordinary text unless
--allow-special is set. Empty documents are skipped and counted.

The output is little-endian u32 ids with no header. A sidecar
<output>.tokens.json records the tokenizer, the ids, the counts and the hashes.

A vocabulary name needs its cargo feature, such as `--features vocab-cl100k`.
The default build bundles none. A tokenizer.json path needs no feature."
)]
pub struct TokenizeArgs {
    /// Input files or directories. A directory is walked for .txt and .jsonl files.
    #[arg(short = 'i', long = "input", required = true, num_args = 1..)]
    pub inputs: Vec<PathBuf>,

    /// Output token file, such as data/corpus.bin.
    #[arg(short = 'o', long = "output")]
    pub output: PathBuf,

    /// cl100k_base, o200k_base, llama3, deepseek_v3, or a path to a tokenizer.json.
    #[arg(long = "tokenizer")]
    pub tokenizer: String,

    /// Input format. `auto` decides per file from its extension.
    #[arg(long = "format", value_enum, default_value_t = FormatArg::Auto)]
    pub format: FormatArg,

    /// JSONL field that holds the document text.
    #[arg(long = "field", default_value = "text")]
    pub field: String,

    /// Where a text-file document ends.
    #[arg(long = "txt-docs", value_enum, default_value_t = TxtDocs::File)]
    pub txt_docs: TxtDocs,

    /// Encode text that spells a special token as that token's id.
    #[arg(long = "allow-special")]
    pub allow_special: bool,

    /// Document text per encode batch, in bytes. The output does not depend on it.
    #[arg(long = "batch-bytes", default_value_t = DEFAULT_BATCH_BYTES, value_parser = parse_batch_bytes)]
    pub batch_bytes: usize,
}

/// Parses `--batch-bytes`: a whole number of bytes above zero.
fn parse_batch_bytes(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(0) => Err("--batch-bytes must be above 0. Pass a byte count such as 67108864.".into()),
        Ok(bytes) => Ok(bytes),
        Err(e) => Err(format!(
            "--batch-bytes '{value}' is not a byte count: {e}. Pass a whole number such as 67108864."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct Cli {
        #[command(flatten)]
        args: TokenizeArgs,
    }

    fn parse(extra: &[&str]) -> Result<TokenizeArgs, clap::Error> {
        let mut argv = vec!["tokenize", "-i", "a.txt", "-o", "out.bin"];
        argv.extend_from_slice(extra);
        argv.extend_from_slice(&["--tokenizer", "cl100k_base"]);
        Cli::try_parse_from(argv).map(|cli| cli.args)
    }

    #[test]
    fn defaults_match_the_documented_values() {
        let args = parse(&[]).expect("parse");
        assert_eq!(args.inputs, vec![PathBuf::from("a.txt")]);
        assert_eq!(args.format, FormatArg::Auto);
        assert_eq!(args.field, "text");
        assert_eq!(args.txt_docs, TxtDocs::File);
        assert!(!args.allow_special);
        assert_eq!(args.batch_bytes, 67_108_864);
    }

    #[test]
    fn several_inputs_follow_one_flag() {
        let argv = [
            "tokenize",
            "-i",
            "a.txt",
            "b.jsonl",
            "dir",
            "-o",
            "out.bin",
            "--tokenizer",
            "cl100k_base",
        ];
        let args = Cli::try_parse_from(argv).expect("parse").args;
        let want: Vec<PathBuf> = ["a.txt", "b.jsonl", "dir"].map(PathBuf::from).into();
        assert_eq!(args.inputs, want);
    }

    #[test]
    fn value_enums_take_their_kebab_names() {
        let args = parse(&["--format", "jsonl", "--txt-docs", "blank-line"]).expect("parse");
        assert_eq!(args.format.name(), "jsonl");
        assert_eq!(args.txt_docs.name(), "blank-line");
    }

    #[test]
    fn zero_batch_bytes_is_refused() {
        let message = parse(&["--batch-bytes", "0"])
            .expect_err("0 must fail")
            .to_string();
        assert!(message.contains("above 0"), "{message}");
    }
}
