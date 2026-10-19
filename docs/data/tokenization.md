# Tokenization

`oxidizr tokenize` turns text files into the token file `oxidizr train -d` reads.
The output is little-endian `u32` ids with no header. See [Data Format](format.md).

```bash
cargo run --release --features vocab-cl100k -- tokenize \
  -i data/raw -o data/corpus.bin --tokenizer cl100k_base

cargo run --release -- train -f models/nano.yaml -d data/corpus.bin
```

## Tokenizers

`--tokenizer` takes a splintr vocabulary name or a `tokenizer.json` path.
A value is a path only when it ends in `.json`.

| Name          | Used by        | Feature          | BOS      | EOS (separator) |
| ------------- | -------------- | ---------------- | -------- | --------------- |
| `cl100k_base` | GPT-4, GPT-3.5 | `vocab-cl100k`   | none     | 100257          |
| `o200k_base`  | GPT-4o         | `vocab-o200k`    | none     | 199999          |
| `llama3`      | Llama 3        | `vocab-llama3`   | 128000   | 128001          |
| `deepseek_v3` | DeepSeek V3/R1 | `vocab-deepseek` | 0        | 1               |

- The default build bundles no vocabulary. `--tokenizer cl100k_base` then fails, and splintr's error names the missing feature.
- Rebuild with that feature, or with `--features vocabs` for every vocabulary.
- Any other splintr name works too when its `vocab-*` feature is on.
- A `tokenizer.json` needs no feature. Prefer it for a real checkpoint: it carries the checkpoint's added tokens.
- A `tokenizer.json` keeps its own post-processor template. Its BOS is the special token that template places first.
- A `tokenizer.json` must declare an EOS token: `</s>`, `<eos>`, `<|endoftext|>`, `<|end_of_text|>` or `[SEP]`.

The model's `vocab_size` must be at least the sidecar's `tokenizer_vocab_size`.

## Documents

Each document becomes the tokenizer's BOS (when it has one), the content ids, then the EOS id.
A template that already ends with EOS gets no second one.

| Input                              | One document is                                          |
| ---------------------------------- | -------------------------------------------------------- |
| `.txt`, `--txt-docs file`          | the whole file                                           |
| `.txt`, `--txt-docs blank-line`    | the lines between whitespace-only lines, minus the last line ending |
| `.jsonl`                           | the `--field` string of one line (default `text`)        |

- `--format auto` decides per file from its extension. `--format txt` or `--format jsonl` forces one format for every file.
- An empty document is skipped. The sidecar counts it in `skipped_empty_documents`.
- A blank JSONL line is skipped.
- Invalid UTF-8, invalid JSON, a line that is not an object, or a missing or non-string field stops the run. The error names `file:line`.
- One document must fit in memory. A document above 1 MiB is encoded across threads. Its ids stay the same.

## Special Tokens

By default, text that spells a special token stays ordinary text.
A document holding `<|endoftext|>` can never forge a separator.

`--allow-special` encodes such text as the special token's id.
Use it only for text that places control tokens on purpose, such as a chat template.

## Inputs

- A file named by `-i` is always read.
- A directory is walked recursively for `.txt` and `.jsonl` files. Symlinked directories are not entered.
- Files from one directory are read in path byte order. `-i` order holds between arguments.
- A file reached twice, or an output path that is also an input, stops the run.

## Output

- The ids go to `out.bin.partial`, then a rename makes it `out.bin`.
- An error before the rename removes the partial file. The old `out.bin` stays.
- The sidecar `out.bin.tokens.json` is written last, atomically.
- The output bytes depend only on the input bytes, the tokenizer and the options. `--batch-bytes` and the thread count never change them.

A reader thread fills batches of at most `--batch-bytes` of text (default 64 MiB). At most two batches exist at once. A document above `--batch-bytes` is a batch of its own.

## The Sidecar

`<output>.tokens.json` describes the token file.

| Field                     | Meaning                                                         |
| ------------------------- | --------------------------------------------------------------- |
| `format_version`          | Sidecar format version, `1`                                     |
| `oxidizr_version`         | The oxidizr that wrote the file                                 |
| `splintr_version`         | The splintr that encoded the text, or `unknown`                 |
| `tokenizer`               | `{"kind": "name", "name"}` or `{"kind": "file", "path", "hash"}` |
| `separator_id`            | The EOS id after every document. `trainer.document_mask` reads it |
| `bos_id`                  | The id before every document's content, or `null`               |
| `tokenizer_vocab_size`    | The tokenizer's highest id plus one                             |
| `max_token_id`            | The highest id in the file, or `null` for zero tokens           |
| `num_tokens`              | Tokens in the file. The file holds `4 × num_tokens` bytes       |
| `num_documents`           | Documents in the file                                           |
| `skipped_empty_documents` | Empty documents left out                                        |
| `hash_algo`               | `fnv1a64`: the algorithm of every hash here                     |
| `file_hash`               | Hash of every byte of the token file                            |
| `inputs`                  | Each input's `path`, `format`, `bytes` and `hash`, in read order |
| `options`                 | `format`, `field`, `txt_docs` and `allow_special`               |

Every hash is 64-bit FNV-1a as 16 hex digits.
The sidecar holds no document offsets.

## Pre-Tokenized Data

The nano-start dataset ships already tokenized:

```bash
hf download fs90/nano-start-data-bin --local-dir data/nano-start/tokenized --repo-type dataset
cargo run --release -- -f models/nano.yaml -d data/nano-start/tokenized/combined.bin
```

## Common Errors

- **Token ids out of range**: the model's `vocab_size` is below the tokenizer's. Set it to at least `tokenizer_vocab_size`.
- **Strange text at inference**: training and inference use different tokenizers. Use one tokenizer throughout.

## Learn More

- [Data Format](format.md): binary file layout and rows
- [nano-start Guide](nano-start.md): educational dataset
- [splintr](https://github.com/ml-rust/splintr): the tokenizer library
