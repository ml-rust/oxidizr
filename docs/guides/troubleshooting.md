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

### Error: `--gpus 0,1` names 2 GPUs, and this build has no NCCL support

Multi-GPU training needs a build with both features:

```bash
cargo build --release --features cuda,nccl
```

The message says "no CUDA support" for a build without `cuda`. It says "trains on the CPU" for `--target-device cpu`. Pass `--target-device gpu`, or remove `--gpus`. Check `trainer.distributed.gpu_ids` in the config too: it applies without `--gpus`.

### Error: names GPU 0 twice

Each rank needs its own GPU. Name each id once in `--gpus` or `trainer.distributed.gpu_ids`.

### Error: checkpoint meta records world size 2, and this run has world size 1

A data-parallel checkpoint resumes only on the same GPU count. Resume with `--gpus` naming that many GPUs, or start a fresh run.

### Error: sync backend `cpu` cannot run 2 ranks

NCCL is the only multi-GPU backend. Pass `--sync-backend nccl`, or remove the flag.

### Error: rank 1 of 2 failed first

One rank failed, and the launcher aborted the others. The text after this line is that rank's error. Fix it as for a one-GPU run. An out-of-memory error on one GPU fails the whole run.

### Error: NCCL init of 2 ranks failed

Check that every id in `--gpus` names a different GPU that `nvidia-smi` lists. `CUDA_VISIBLE_DEVICES` renumbers the visible GPUs from 0.

### GPUs not detected

```bash
nvidia-smi -L
```

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
