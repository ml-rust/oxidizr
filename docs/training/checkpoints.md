# Checkpoints

Oxidizr saves training progress to checkpoints, allowing you to resume training or use trained models.

## Checkpoint Format

Checkpoints are saved in SafeTensors format (`.safetensors`):
- Efficient binary format
- Safe to load (no arbitrary code execution)
- Compatible with many ML tools

## Automatic Saving

Configure checkpoint saving in your YAML config:

```yaml
trainer:
  checkpoint_dir: "./checkpoints"
  save_interval: 500  # Save every 500 steps
```

Checkpoints are saved as:
```
checkpoints/
├── <prefix>_step_500.safetensors
├── <prefix>_step_500.optimizer.safetensors
├── checkpoint_step_500.meta.json
├── <prefix>_step_1000.safetensors
├── <prefix>_step_1000.optimizer.safetensors
├── checkpoint_step_1000.meta.json
└── ...
```

## Checkpoint Management

### Maximum Checkpoints

Control how many checkpoints to keep:

```bash
# Keep only 5 most recent checkpoints
cargo run --release -- -f models/nano.yaml --max-checkpoints 5

# Keep all checkpoints
cargo run --release -- -f models/nano.yaml --max-checkpoints 0
```

Default: 10 checkpoints

### Manual Save Points

The save interval determines when checkpoints are written:

```yaml
trainer:
  save_interval: 1000  # Every 1000 steps
```

Smaller intervals = more recovery points, more disk usage.

## Resuming Training

### From Specific Checkpoint

```bash
cargo run --release -- -f models/nano.yaml --resume checkpoints/step_1000.safetensors
```

### From Latest Checkpoint

```bash
cargo run --release -- -f models/nano.yaml --resume auto
```

The `auto` option finds the most recent checkpoint in the config's `checkpoint_dir`.

## What's Saved

Each checkpoint is a set of three files in `checkpoint_dir`:

| File | Content |
|------|---------|
| `<prefix>_step_N.safetensors` | Model weights after N completed optimizer steps |
| `<prefix>_step_N.optimizer.safetensors` | AdamW state: `exp_avg`, `exp_avg_sq`, `master` per trainable tensor, and `__step` |
| `checkpoint_step_N.meta.json` | Step N, config and data-file hashes with their `hash_algo`, dtype, `data_position`, and `optimizer_file` |

The end of a run writes the same set as `<prefix>_final.safetensors`,
`<prefix>_final.optimizer.safetensors` and `checkpoint_final.meta.json`.

The optimizer file uses the weights file's tensor names. Frozen parameters
have no optimizer state. A tied `lm_head.weight` has no entry of its own.

Each file is written atomically: a temp file in the same directory, then a
rename. The trainer writes the optimizer file, then the weights, then the
meta. The meta marks the set as complete. `--resume auto` skips weights that
have an optimizer file and no meta, because that save stopped early.

`--max-checkpoints` removes all three files of an old set.

## Resume Behavior

When resuming:
1. Model weights load from the checkpoint.
2. The optimizer state loads onto the training device. The load is strict:
   an unknown tensor name or a shape or dtype mismatch stops the run.
3. Training continues at step N, with the learning-rate schedule at step N.
4. The data loader seeks to `data_position`, so the resumed run draws the
   same rows, in the same shuffled order, as an uninterrupted run.

With all three files, the resumed run matches an uninterrupted run exactly.
The training step uses no random state, so no RNG state is saved.

`--resume <dir>/<prefix>_final.safetensors` resumes at the step that
`checkpoint_final.meta.json` records.

A checkpoint written before optimizer state existed has no optimizer file.
The resume prints a warning, restarts the Adam moments and step count at
zero, and continues. That run does not match an uninterrupted run. A resume
at step 0, such as from `oxidizr import`, needs no optimizer state.

A meta that names an optimizer file that does not exist stops the run.
Restore the file, or remove `optimizer_file` from the meta to resume with
fresh Adam moments.

`--resume auto` with `--data` compares the model config hash and the
data-file hash. The data-file hash covers the first 1 MiB of the file and
its size. Both hashes use 64-bit FNV-1a (`hash_algo: fnv1a64`), so a Rust
upgrade cannot change them.

Metadata without `hash_algo` holds `legacy` hashes from Rust's
`DefaultHasher`. The resume recomputes them the legacy way. A legacy
data-file hash that differs prints a warning, and the resume continues
without a data check. A legacy model hash that differs reports a mismatch,
as before.

Metadata written before `data_position` existed has none. The loader
position then derives from N and the current `batch_size` and
`gradient_accumulation`, and a stderr line reports it.

## Best Practices

### Regular Checkpoints

For long training runs:
```yaml
trainer:
  save_interval: 500  # Save frequently
  max_checkpoints: 10  # But don't fill disk
```

### Before Risky Changes

Save a checkpoint before:
- Changing learning rate
- Modifying config
- System maintenance

### Backup Important Checkpoints

Copy important checkpoints outside `checkpoint_dir`. Copy all three files
of the set to resume from the copy exactly:
```bash
cp checkpoints/<prefix>_step_5000.safetensors \
   checkpoints/<prefix>_step_5000.optimizer.safetensors \
   checkpoints/checkpoint_step_5000.meta.json backups/
```

## Checkpoint Directory Structure

```
my_project/
├── models/
│   └── nano.yaml
├── checkpoints/          # Default location
│   ├── step_500.safetensors
│   ├── step_1000.safetensors
│   └── step_1500.safetensors
└── data/
    └── ...
```

Custom directory:
```yaml
trainer:
  checkpoint_dir: "/mnt/storage/checkpoints"
```

## Troubleshooting

**Checkpoint not found?**
- Check the path is correct
- Verify file exists: `ls checkpoints/`
- Use `--resume auto` to find latest

**Resume produces different results?**
- Check for the warning about a missing optimizer state file
- Ensure same config file
- Verify data is the same

**Out of disk space?**
- Reduce `save_interval`
- Lower `--max-checkpoints`
- Clean old checkpoints manually

**Checkpoint too large?**
- Normal for large models
- Use lower precision training (`--dtype bf16`)
- Checkpoints scale with model size
