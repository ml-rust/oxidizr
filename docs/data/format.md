# Data Format

Oxidizr accepts tokenized data in a simple binary format.

## Binary Format

Training data is stored as raw token IDs:

- **Type**: u32 (32-bit unsigned integer)
- **Byte order**: Little-endian
- **Headers**: None (raw token stream)
- **Extension**: `.bin`

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

tokens = np.fromfile("data.bin", dtype=np.uint32)
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

# Save as binary
np.array(tokens, dtype=np.uint32).tofile("data.bin")
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

Before creating binary files, you need to tokenize your text. The tokenizer converts text to token IDs.

### Using splintr (Recommended)

[splintr](https://github.com/ml-rust/splintr) is our tokenizer library:

```python
from splintr import Tokenizer
import numpy as np

# Load tokenizer
tokenizer = Tokenizer("cl100k_base")  # Or "llama3", "deepseek_v3"

# Tokenize text
text = "Hello, world! This is training data."
tokens = tokenizer.encode(text)

# Save as binary
np.array(tokens, dtype=np.uint32).tofile("data.bin")
```

See [Tokenization](tokenization.md) for more details.

## Data Organization

### Single File

Simple approach - one file with all data:

```bash
cargo run --release -- -f config.yaml -d data/all_data.bin
```

### Concatenated Examples

Multiple examples separated by EOS token:

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

np.array(all_tokens, dtype=np.uint32).tofile("data.bin")
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
