# Tokenization

Language models don't process text directly - they work with numbers called **tokens**. Tokenization converts text into token IDs:

```
"Hello world" → [9906, 1917]
```

## What is a Token?

A token is a piece of text that the model treats as a single unit. Tokens can be:

- Whole words: `hello` → `[15339]`
- Parts of words: `tokenization` → `[5765, 2065]`
- Single characters: `!` → `[0]`
- Special markers: `<|endoftext|>` → `[100257]`

## Why Tokenize?

Neural networks work with numbers, not text. Tokenization:

1. Converts text to numbers the model can process
2. Reduces vocabulary size (compared to one ID per character)
3. Captures meaningful text chunks

## Pre-Tokenized Data

The easiest way to get started is using pre-tokenized data:

```bash
# Download nano-start (already tokenized)
hf download fs90/nano-start-data-bin --local-dir data/nano-start/tokenized --repo-type dataset

# Train immediately
cargo run --release -- -f models/nano.yaml -d data/nano-start/tokenized/combined.bin
```

## Tokenizing Your Own Data

To tokenize your own text, use [splintr](https://github.com/ml-rust/splintr):

```python
from splintr import Tokenizer
import numpy as np

# Load tokenizer
tokenizer = Tokenizer("cl100k_base")  # OpenAI GPT-4 tokenizer

# Tokenize text
text = "Your training data here..."
tokens = tokenizer.encode(text)

# Save for oxidizr
np.array(tokens, dtype=np.uint32).tofile("data.bin")
```

## Available Tokenizers

| Tokenizer     | Used By        | Vocab Size |
| ------------- | -------------- | ---------- |
| `cl100k_base` | GPT-4, GPT-3.5 | ~100K      |
| `o200k_base`  | GPT-4o         | ~200K      |
| `llama3`      | Llama 3        | ~128K      |
| `deepseek_v3` | DeepSeek V3/R1 | ~128K      |

## Tokenizer Selection

Choose based on your use case:

- **Learning/Experimenting**: `cl100k_base` (widely used, good docs)
- **Llama-style models**: `llama3`
- **DeepSeek-style**: `deepseek_v3`

Ensure your config's `vocab_size` matches:

```yaml
model:
  vocab_size: 100331 # For cl100k_base
  # vocab_size: 128354  # For llama3 with agent tokens
```

## Special Tokens

Tokenizers include special tokens for structure:

| Token             | Purpose             | Example ID |
| ----------------- | ------------------- | ---------- |
| `<\|endoftext\|>` | End of example      | 100257     |
| `<\|system\|>`    | System instructions | 100277     |
| `<\|user\|>`      | User message        | 100278     |
| `<\|assistant\|>` | Assistant response  | 100279     |

Use these to structure your training data:

```python
text = "<|user|>What is 2+2?<|assistant|>4<|endoftext|>"
tokens = tokenizer.encode(text)
```

## Batch Tokenization

For large datasets:

```python
from splintr import Tokenizer
import numpy as np

tokenizer = Tokenizer("cl100k_base")

# Read all your text files
texts = []
for path in text_files:
    with open(path) as f:
        texts.append(f.read())

# Tokenize all (uses parallel processing)
all_tokens = []
for text in texts:
    tokens = tokenizer.encode(text)
    all_tokens.extend(tokens)
    all_tokens.append(100257)  # EOS separator

# Save
np.array(all_tokens, dtype=np.uint32).tofile("combined.bin")
```

## Decoding Tokens

Convert tokens back to text:

```python
tokenizer = Tokenizer("cl100k_base")

tokens = [9906, 1917]
text = tokenizer.decode(tokens)
print(text)  # "Hello world"
```

## Common Issues

**Token IDs out of range?**

- Your data has token IDs larger than `vocab_size`
- Use matching tokenizer and config

**Strange characters in output?**

- Tokenizer mismatch between training and inference
- Use same tokenizer throughout

**Training on different language?**

- Most tokenizers work with any language
- Some may have better support for specific languages

## Learn More

- [splintr documentation](https://github.com/ml-rust/splintr)
- [Data Format](format.md) - Binary file format details
- [nano-start Guide](nano-start.md) - Educational dataset
