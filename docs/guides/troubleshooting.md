# Troubleshooting

Common issues and solutions when using Oxidizr.

## Build Issues

### CUDA not found

```
error: could not find CUDA installation
```

**Solution:**
- Install CUDA Toolkit 12.x
- Verify with `nvcc --version`
- Ensure CUDA is in PATH

Or build without CUDA:
```bash
cargo build --release  # CPU only
```

### CUDA version mismatch

```
error: CUDA 13.x not supported
```

**Solution:**
CUDA 13.x is not yet supported. Use CUDA 12.x.

### Compilation errors

```bash
# Clean build
cargo clean
cargo build --release
```

## Runtime Issues

### Progress bar doesn't appear

The TUI progress bar may not render in:
- Non-interactive terminals
- CI/CD environments
- Piped output
- Some remote shells

**Solution:**
```bash
cargo run --release -- -f config.yaml --headless
```

### Training seems stuck

If progress bar appears but doesn't update:

1. **Check if actually training:**
   ```bash
   cargo run --release -- -f config.yaml --headless
   ```

2. **Verify output:**
   Look for JSON lines like:
   ```json
   {"step": 1, "loss": 11.5, ...}
   ```

3. **Training is slow, not stuck:**
   CPU training is ~10-20x slower than GPU.

### Out of memory (OOM)

**GPU OOM:**
```
CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size
cargo run --release --features cuda -- -f config.yaml --batch-size 1

# Reduce sequence length
cargo run --release --features cuda -- -f config.yaml --seq-len 128

# Use lower precision
cargo run --release --features cuda -- -f config.yaml --dtype bf16
```

**CPU OOM:**
Same solutions apply, but memory pressure is usually on GPU first.

### Loss is NaN

**Causes:**
- Learning rate too high
- Numerical instability
- Bad data (token IDs out of range)

**Solutions:**
```yaml
trainer:
  learning_rate: 0.0001  # Lower learning rate
```

Check data vocab size matches config:
```yaml
model:
  vocab_size: 128354  # Must cover all token IDs in data
```

### Loss not decreasing

**Causes:**
- Learning rate too low
- Model too small for task
- Data issues

**Solutions:**
- Increase learning rate slightly
- Increase model size
- Verify data quality

## Data Issues

### File not found

```
Error: No such file or directory
```

**Solution:**
Use absolute path or verify relative path:
```bash
cargo run --release -- -f models/nano.yaml -d /full/path/to/data.bin
```

### Invalid data format

**Symptoms:**
- Strange loss values
- Immediate NaN

**Solution:**
Verify data is u32 little-endian:
```python
import numpy as np
tokens = np.fromfile("data.bin", dtype=np.uint32)
print(tokens[:10])  # Should be reasonable token IDs
```

### Token IDs out of range

**Symptom:**
```
index out of bounds: token 150000 > vocab_size 128354
```

**Solution:**
Increase vocab_size in config or use correct tokenizer.

## Checkpoint Issues

### Cannot resume

```
Error: Checkpoint not found
```

**Solution:**
```bash
# Check checkpoint exists
ls checkpoints/

# Use correct path
cargo run --release -- -f config.yaml --resume checkpoints/step_1000.safetensors

# Or use auto-find
cargo run --release -- -f config.yaml --resume auto
```

### Resume produces different results

**Causes:**
- The checkpoint has no `.optimizer.safetensors` file. The resume prints a
  warning and restarts the Adam moments.
- Different config file
- Different data

**Solution:**
Use exact same config and data as original run.

## Multi-GPU Issues

### GPUs not detected

```bash
# Check visible GPUs
nvidia-smi

# Verify GPU IDs
nvidia-smi -L
```

### NCCL errors

```
NCCL error: ...
```

**Solutions:**
- Ensure NCCL is installed
- Build with `--features cuda,nccl`
- Try CPU backend first: `--sync-backend cpu`

### Uneven GPU utilization

Normal for small datasets. Data sharding may not be perfectly even.

## Performance Issues

### Slow training

**CPU is slow (expected):**
- CPU is 10-20x slower than GPU
- Use GPU if available

**GPU slower than expected:**
```bash
# Check GPU utilization
nvidia-smi

# Enable prefetching
cargo run --release --features cuda -- -f config.yaml --prefetch 2

# Use BF16
cargo run --release --features cuda -- -f config.yaml --dtype bf16
```

### High memory usage

```yaml
trainer:
  batch_size: 1
  gradient_accumulation: 4  # Simulate batch=4
```

## Getting Help

1. Check this troubleshooting guide
2. Review [CLI Reference](../cli.md) for options
3. Examine config examples in `models/`
4. Open an issue on GitHub with:
   - Error message
   - Config file
   - Command used
   - System info (OS, GPU, CUDA version)
