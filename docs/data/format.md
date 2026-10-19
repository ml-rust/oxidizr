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
- A batch is `batch_size` rows. Each epoch visits every row once.
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

#### Resume

A checkpoint meta lists every source under `data_sources`: its path, its hash
and its whole rows. A resume compares the sources by position.

- A changed file count, file order, hash or row count is an error. The error names the source.
- A moved file with the same contents still matches. The path alone is not compared.
- Resume with the original files in their original order, or start a fresh run.
- A meta written before `data_sources` reads as one source: its `data_file_path` and `data_file_hash`. Its rows are not compared.

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
