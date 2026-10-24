# Data Format

Oxidizr accepts tokenized data in a simple binary format.

## Binary Format

Training data is stored as raw token IDs:

- **Type**: u32 (32-bit unsigned integer)
- **Byte order**: little-endian on every platform
- **Headers**: None (raw token stream)
- **Extension**: `.bin`
- **Sidecar**: `oxidizr tokenize` also writes `<file>.tokens.json`. See [Tokenization](tokenization.md#the-sidecar).

### Structure

```
[token_0][token_1][token_2]...[token_N]
   4B      4B       4B           4B
```

Each token is 4 bytes. File size = `num_tokens × 4` bytes.

### Example

A file with 6,379 tokens:

- Size: 6,379 × 4 = 25,516 bytes
- Contains: Raw u32 values in sequence

### Rows

The loader cuts the stream into rows of `max_seq_len + 1` tokens at fixed
offsets. Row `r` is tokens `[r × (max_seq_len + 1), (r + 1) × (max_seq_len + 1))`.

- Inputs are `row[0..max_seq_len]`. Targets are `row[1..max_seq_len + 1]`.
- A batch is `batch_size` rows. Each epoch visits every row once. A [weighted mixture](#weighted-mixture) sets its own epoch instead.
- Row order: a per-epoch permutation seeded from `trainer.seed` when
  `trainer.shuffle_rows` is `true` (the default), stream order when `false`.
- Steps per epoch: `floor(rows / batch_size) / gradient_accumulation`. The
  partial batch at an epoch's end is dropped.
- Tokens past the last whole row are never read.

On a little-endian machine the loader memory-maps each file and reads the ids
in place. On a big-endian machine it reads each file into memory once and
decodes every id from little-endian bytes. A mapped file must not shrink
while a run reads it.

A dataset of short, independent examples can be packed so no example
straddles a row: write rows of exactly `max_seq_len + 1` tokens and pad each
row's tail with an id the loss mask never scores.

## Reading Binary Data

### Python

```python
import struct

def read_tokens(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data)//4}I", data))

tokens = read_tokens("data.bin")
print(f"Total tokens: {len(tokens)}")
```

### NumPy

```python
import numpy as np

tokens = np.fromfile("data.bin", dtype="<u4")  # little-endian u32
print(f"Total tokens: {len(tokens)}")
```

### Rust

```rust
use std::fs::File;
use std::io::Read;

fn read_tokens(path: &str) -> Vec<u32> {
    let mut file = File::open(path).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    buffer
        .chunks(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
```

## Creating Binary Data

### From Python (NumPy)

```python
import numpy as np

# Your token IDs
tokens = [100, 200, 300, 400, 500]

# Save as binary: "<u4" is little-endian u32 on every platform
np.array(tokens, dtype="<u4").tofile("data.bin")
```

### From Python (struct)

```python
import struct

tokens = [100, 200, 300, 400, 500]

with open("data.bin", "wb") as f:
    for token in tokens:
        f.write(struct.pack("<I", token))  # Little-endian u32
```

## Tokenization

`oxidizr tokenize` turns `.txt` and `.jsonl` files into this format:

```bash
cargo run --release --features vocab-cl100k -- tokenize -i data/raw -o data/all_data.bin --tokenizer cl100k_base
```

Each document becomes its ids, then the tokenizer's EOS id. See [Tokenization](tokenization.md).

## Data Organization

### Single File

One file with all data:

```bash
cargo run --release -- -f config.yaml -d data/all_data.bin
```

### Several Files

Repeat `-d` to read several files as one stream, in flag order:

```bash
cargo run --release -- train -f config.yaml -d data/web.bin -d data/code.bin
```

`trainer.data.sources` names the same list in the config:

```yaml
trainer:
  data:
    sources:
      - path: data/web.bin
      - path: data/code.bin
```

- The stream is file 0's whole rows, then file 1's, and so on.
- Each file drops its own tail shorter than a row. No row spans two files.
- The row order shuffles the whole stream. Row `r` of the shuffle can come from any file.
- Setting `-d` and `trainer.data.sources` together is an error naming both.
- One source in `trainer.data.sources` trains bit-identical to `-d` with that file.
- `${VAR}` in a source path expands, as in `trainer.checkpoint_dir`.

#### Held-out rows

Under `trainer.eval.holdout_fraction`, each file holds out that fraction of
its own rows, rounded down per file. The held-out rows come off the end of
each file. The eval reads the held-out ranges in file order.

- Training never reads a held-out row.
- `trainer.eval.holdout_rows` names one range, so it needs one file. With two or more files it is an error naming `trainer.eval.holdout_fraction`.

#### Weighted mixture

A `weight` on every source turns the stream into a weighted mixture:

```yaml
trainer:
  data:
    rows_per_epoch: 100000   # optional
    sources:
      - path: data/web.bin
        weight: 0.7
      - path: data/code.bin
        weight: 0.3
```

- Set a `weight` on every source, or on none. A weight is a finite number above 0. Weights are relative: `0.7` and `0.3` mix as `7` and `3` do.
- Without weights the run reads the plain stream, bit-identical to a run before mixtures.
- An epoch holds `rows_per_epoch` rows. The default is every training row of every source.
- `rows_per_epoch` needs weights. A value of 0 is an error.
- One weighted source with the default `rows_per_epoch` trains bit-identical to that file without a weight.

Per-epoch quota: source `i` supplies `c_i` rows of each epoch of `L` rows.

- `c_i` is the largest-remainder rounding of `w_i / sum(w) * L`.
- Each source gets the floor of its share. The rows left go one each to the largest remainders.
- A tie goes to the lower source index. The quotas sum to `L` exactly.

Order of the slots:

- `trainer.shuffle_rows: true`: a seeded shuffle of the epoch's quotas. The seed mixes `trainer.seed` and the epoch, so each epoch draws another order.
- `trainer.shuffle_rows: false`: a smooth weighted round-robin. It spreads each source evenly, and it is the same every epoch.

Rows of a source:

- A source reads its training rows in passes. Each pass reads every row once.
- Under a shuffle, each pass is a permutation seeded from `trainer.seed`, the source and the pass. In stream order a pass reads the rows in order.
- A source whose quota exceeds its rows repeats them: an upsampled source. A source with fewer quota rows than training rows takes several epochs per pass.
- At startup the run prints each source's quota, its training rows, and the repeats per epoch, `c_i / n_i`.

Data parallelism: every rank builds the same schedule. Rank `k` reads its usual slots of each global batch.

Holdout: each source holds out its held-out rows first. The mixture covers the training rows only. The eval reads the held-out ranges in file order, unmixed.

#### Resume

A checkpoint meta lists every source under `data_sources`: its path, its hash
and its whole rows. A resume compares the sources by position.

- A changed file count, file order, hash or row count is an error. The error names the source.
- A moved file with the same contents still matches. The path alone is not compared.
- Resume with the original files in their original order, or start a fresh run.
- A meta written before `data_sources` reads as one source: its `data_file_path` and `data_file_hash`. Its rows are not compared.
- A mixture run's meta records `mixture`: the weights and `rows_per_epoch`. A resume with another weight, another `rows_per_epoch`, or without weights is an error naming the key.
- A meta without `mixture` resumes only a run without weights.
- The saved data position is exact under a mixture. The row at each slot depends only on the epoch, the slot, the seed and the sources.

### Sidecar Checks

A file with a `<file>.tokens.json` sidecar is checked when it opens:

- The sidecar's `num_tokens` must equal the file's token count.
- The sidecar's `max_token_id` must be below `model.vocab_size`. The error names the file, `max_token_id` and `vocab_size`.
- A file without a sidecar is not checked.

### Concatenated Examples

Multiple examples separated by the EOS id. `oxidizr tokenize` writes this layout:
one example per document (see [Tokenization](tokenization.md#documents)), each followed by EOS.

A file written by other tools needs the same layout:

```python
import numpy as np
from splintr import Tokenizer

tokenizer = Tokenizer("cl100k_base")
EOS = 100257  # cl100k_base EOS token

examples = [
    "First training example.",
    "Second training example.",
    "Third training example.",
]

all_tokens = []
for text in examples:
    tokens = tokenizer.encode(text)
    all_tokens.extend(tokens)
    all_tokens.append(EOS)  # Separator

np.array(all_tokens, dtype="<u4").tofile("data.bin")
```

### Document Boundaries

A row holds several documents back to back. Without `trainer.document_mask`, attention crosses them, as in standard packed LM training. With it, the separator id delimits documents. See [Document Masking](../configuration.md#document-masking) for the key.

Segment rules, over the input tokens of each row:

- A document ends at its separator. The separator belongs to the document it ends.
- Every row start begins a new document. A row never continues the document of the row before it.
- Two separators in a row make a one-token document.
- A row without a separator is one document.

Each token attends only to earlier tokens of its own document. RoPE needs no position reset: rotated scores depend only on the distance between two tokens.

Loss rule:

- A position whose input is the separator is unscored. Its target is the first token of the next document.
- The document mask combines with `trainer.target_mask` or `model.speech_layout` by AND.
- The scored count drops the separator positions. Step weighting uses it, see [loss weighting](../configuration.md#loss-weighting-under-gradient-accumulation).

Example, `seq_len` 8, separator `S`:

```text
inputs:    a0 a1 a2 S  b0 b1 b2 b3
documents: 0  0  0  0  1  1  1  1
scored:    1  1  1  0  1  1  1  1
```

Limits:

- Mamba, Mamba2, Mamba3 and Gated DeltaNet layers reset their conv window and state at a document start.
- No `model.sliding_window`.
- On CUDA a GQA model needs head dim 64, 128 or 256, and f32 or f16.
- An MLA model needs `model.dtype: f32`.

## Vocab Size

Your config's `vocab_size` must match or exceed your tokenizer's vocabulary:

| Tokenizer   | Vocab Size |
| ----------- | ---------- |
| cl100k_base | 100,331    |
| llama3      | ~128,000   |
| deepseek_v3 | ~128,000   |

```yaml
model:
  vocab_size: 128354 # Llama 3 + agent tokens
```

## Dummy Data

For testing, oxidizr can generate random tokens:

```bash
# Train with random data (no -d flag)
cargo run --release -- -f models/nano.yaml
```

Or in code:

```rust
use oxidizr::data::create_dummy_data;

let tokens = create_dummy_data(vocab_size, num_tokens);
```

## File Size Guidelines

| Tokens | File Size | Training Time (rough) |
| ------ | --------- | --------------------- |
| 10K    | 40 KB     | Minutes               |
| 100K   | 400 KB    | ~Hour                 |
| 1M     | 4 MB      | Hours                 |
| 10M    | 40 MB     | Day+                  |

Actual training time depends on model size, hardware, and settings.
