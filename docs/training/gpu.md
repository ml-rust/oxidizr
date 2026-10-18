# GPU Training

GPU acceleration significantly speeds up training. Oxidizr uses CUDA for GPU support.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x installed
- `nvidia-smi` should work

## Building with CUDA

```bash
cargo build --release --features cuda
```

## Running on GPU

```bash
# GPU is default when CUDA feature is enabled
cargo run --release --features cuda -- -f models/nano.yaml

# Explicit GPU target
cargo run --release --features cuda -- -f models/nano.yaml --target-device gpu
```

## Performance

GPU training is significantly faster:

| Model | CPU | GPU (RTX 3090) |
|-------|-----|----------------|
| nano (~83M) | ~3 it/s | ~50+ it/s |

## Memory Management

### Check GPU Memory

```bash
nvidia-smi
```

### Adjust for Your GPU

```bash
# If out of memory, reduce batch size
cargo run --release --features cuda -- -f models/nano.yaml --batch-size 2

# Or reduce sequence length
cargo run --release --features cuda -- -f models/nano.yaml --seq-len 256

# Or use gradient accumulation (config file)
```

### Memory Estimation

Oxidizr estimates memory requirements before training starts. Watch for warnings about VRAM usage.

## Precision Options

Use lower precision for faster training and less memory:

```bash
# BF16 (recommended for modern GPUs - Ampere+)
cargo run --release --features cuda -- -f models/nano.yaml --dtype bf16

# FP16
cargo run --release --features cuda -- -f models/nano.yaml --dtype f16

# FP32 (default, most precise)
cargo run --release --features cuda -- -f models/nano.yaml --dtype f32
```

**BF16** is recommended for RTX 30xx/40xx and A100/H100.

## Selecting GPU

Use specific GPU by ID:

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 cargo run --release --features cuda -- -f models/nano.yaml

# Use GPU 1
CUDA_VISIBLE_DEVICES=1 cargo run --release --features cuda -- -f models/nano.yaml
```

## Multi-GPU

Multi-GPU training is not available. Oxidizr trains on one GPU. Pass `--gpus 0` or use `CUDA_VISIBLE_DEVICES`.

## Troubleshooting

**CUDA not found?**
- Ensure CUDA 12.x is installed
- Check `nvcc --version`
- Verify `nvidia-smi` works

**Out of memory?**
- Reduce `--batch-size`
- Reduce `--seq-len`
- Use `--dtype bf16` or `--dtype f16`
- Use gradient accumulation

**Wrong GPU used?**
- Set `CUDA_VISIBLE_DEVICES=X` before command

**Build errors?**
- CUDA 13.x not yet supported (use CUDA 12.x)
- Ensure CUDA toolkit matches your driver
