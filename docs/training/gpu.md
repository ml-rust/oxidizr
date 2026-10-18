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

`--gpus <ID>` trains on that GPU. `CUDA_VISIBLE_DEVICES` limits which GPUs the process sees:

```bash
# Use GPU 1
cargo run --release --features cuda -- -f models/nano.yaml --gpus 1

# The same GPU, seen as GPU 0
CUDA_VISIBLE_DEVICES=1 cargo run --release --features cuda -- -f models/nano.yaml
```

## Multi-GPU

Oxidizr trains one model on several GPUs of one machine with data parallelism. It needs a build with `--features cuda,nccl`:

```bash
cargo build --release --features cuda,nccl
./target/release/oxidizr train -f models/nano.yaml --gpus 0,1
```

How a run works:

- One process starts one rank per GPU id, each on its own thread. Rank `i` uses the i-th id of `--gpus`, or of `trainer.distributed.gpu_ids` without the flag.
- Each rank builds its own model, data loader and trainer. Rank 0's initial weights reach every rank by broadcast.
- Each global batch is `batch_size x GPUs` rows. Rank `i` reads its own `batch_size` rows of it. Every rank sees the same row order and data position.
- After the backward passes, one NCCL all-reduce sums the gradients and the loss. Clipping, the loss scale and the optimizer step then run on the summed gradients, so every replica stays bit-equal.
- A masked run sums its scored token counts over every rank first. A mask on some ranks only is an error on every rank.
- Rank 0 alone prints, evaluates and writes files. Its early-stop decision reaches every rank at the step boundary.
- A failure or panic on one rank aborts every NCCL communicator. The run fails with the error of the rank that failed first, and `run.json` records it.

Batch semantics:

| Config | Rows per optimizer step |
|--------|-------------------------|
| 1 GPU, `batch_size: 8`, `gradient_accumulation: 2` | 16 |
| 2 GPUs, `batch_size: 4`, `gradient_accumulation: 2` | 16, the same rows as the line above |
| 2 GPUs, `batch_size: 8`, `gradient_accumulation: 2` | 32 |

The two 16-row runs match up to float rounding. Halve `batch_size` when you double the GPU count to keep the global batch.

Resume:

- The checkpoint meta records `world_size` and `global_batch_rows`. A resume with another GPU count is an error that names both counts. Resume with `--gpus` naming the original count.
- A meta without `world_size` comes from a one-GPU run, and resumes on one GPU.
- A resume with another global batch warns: the data stream continues at the saved row, but the run does not match an uninterrupted one.

Errors before training starts:

- An id named twice.
- More than one id with `--target-device cpu`.
- More than one id on a build without `cuda` or without `nccl`. The message names the feature to add.
- `--sync-backend cpu`, or an explicit `--sync-backend single`, with more than one id. NCCL is the only multi-GPU backend.

Not supported: FSDP, ZeRO sharding, tensor parallelism, pipeline parallelism, and more than one machine.

Test coverage: the CPU test suite runs the launcher, the gradient sum, resume and failure handling on in-process CPU ranks. The NCCL communicator, the NCCL abort, and the per-GPU device setup are untested on real GPUs.

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
- Pass `--gpus X`, or set `CUDA_VISIBLE_DEVICES=X` before the command

**Build errors?**
- CUDA 13.x not yet supported (use CUDA 12.x)
- Ensure CUDA toolkit matches your driver
