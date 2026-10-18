# Output Modes

Oxidizr supports two output modes for training progress.

## Interactive Mode (Default)

Shows a TUI (Terminal User Interface) progress bar with live metrics:

```
━━━━━━━━━━━━━━━━━━━━──────────────────── 500/1000 [2m14s<2m11s] loss: 8.2340 | grad: 0.5012 | 12.50 samples/s | 6400 tok/s | MFU 38.4%
```

**Features:**

- Real-time loss tracking
- Training speed (samples/second and tokens/second, recent steps)
- MFU, under `trainer.metrics.peak_tflops`
- ETA to completion
- Visual progress bar

### When Interactive Mode Works

- Interactive terminal sessions
- Direct terminal access (not piped)
- Most local development environments

### When Interactive Mode May Fail

- Non-interactive terminals
- CI/CD environments
- Piped output (`cargo run ... | tee log.txt`)
- Some remote shells (tmux, screen with issues)
- Redirected output (`cargo run ... > output.txt`)

## Headless Mode

Outputs JSON metrics to stdout, one line per log interval:

```bash
cargo run --release -- -f models/nano.yaml --headless
```

Output:

```json
{"step": 1, "loss": 11.5161, "grad_norm": 0.2542, "learning_rate": 2.000000e-3, "epoch": 0.01, "samples/s": 3.24, "vram": "812", "tokens": 1536, "tokens_per_sec": 1658.88, "vram_device": "1024/12288"}
{"step": 21, "loss": 11.4948, "grad_norm": 0.1980, "learning_rate": 2.000000e-3, "epoch": 0.21, "samples/s": 3.09, "vram": "812", "tokens": 1536, "tokens_per_sec": 1582.08, "vram_device": "1024/12288", "mfu": 0.0123}
```

### When to Use Headless Mode

1. **TUI doesn't render** - Progress bar appears stuck or garbled
2. **CI/CD pipelines** - Automated training jobs
3. **Logging to file** - Want to capture metrics
4. **Parsing metrics** - Programmatic analysis
5. **Remote execution** - Scripts, cron jobs

### JSON Fields

| Field           | Description                 |
| --------------- | --------------------------- |
| `step`          | Current training step       |
| `loss`          | Training loss               |
| `grad_norm`     | Gradient norm               |
| `learning_rate` | Learning rate this step applied |
| `epoch`         | Epoch progress (fractional) |
| `samples/s`     | Samples per second          |
| `vram`          | GPU memory this process holds, in MiB, summed over the GPUs that list it. `"0"` when no GPU lists it. `"n/a"` when NVML is unavailable |
| `tokens`        | Tokens the step processed over every rank: rows times `max_seq_len` |
| `tokens_per_sec` | Session tokens per second over every rank |
| `vram_device`   | `"<used>/<total>"` MiB of the GPUs that list this process, every process included. `"n/a"` when NVML is unavailable or no GPU lists it |
| `scored_tokens` | Masked runs only: tokens the loss scored over every rank |
| `mfu`           | Under `trainer.metrics.peak_tflops` only: model FLOPs utilization, a fraction of 1 |

The keys up to `vram` keep their order. New keys go after them.

`vram` and `vram_device` come from NVML, which only a `--features cuda` build reads. It loads `libnvidia-ml` at run time. A missing library or driver reads as `"n/a"`. A CPU build always reports `"n/a"`. The tests run the query on a CPU build only. No test reads a GPU. NVML lists host PIDs. Inside a container without the host PID namespace, `vram` reads `"0"`.

Other headless lines wrap their event in one key:

| Line | When |
| ---- | ---- |
| `{"warmup": {"steps": 2, "base_lr": 3.000000e-3}}` | At start, when the run warms up |
| `{"checkpoint": {"step": 500}}` | After an interval checkpoint |
| `{"eval": {"step": 500, "loss": 2.3026, "perplexity": 10.0000, "batches": 50, "tokens": 51200, "best_loss": 2.3026, "since_best": 0}}` | After each eval under `trainer.eval` |
| `{"checkpoint": {"step": 500, "label": "best"}}` | After an eval writes the best checkpoint |
| `{"spike": {"step": 812, "loss": 9.1234, "mean": 3.2100, "std": 0.0500, "z": 118.2680, "action": "skip"}}` | Under `trainer.spike_guard`, before the step line, for every flagged step, logged or not |
| `{"status": "complete"}` | When the run ends |
| `{"status": "complete", "stopped_early": true, "reason": "..."}` | When early stopping ends the run |

An eval line writes a non-finite `loss` or `perplexity` as `null`. A spike line writes `z` as `null` when the EMA variance is 0. A flagged step also prints a `[SpikeGuard]` line to stderr in both modes. `best_loss` is `null` before the first finite eval loss. The final-save line, `Saving final checkpoint to <path>`, is plain text. Interactive mode prints one line per eval above the progress bar.

## Run Directory Files

Both modes write the same files to the run directory. Set it with `--run-dir`. The default is `$OXIDIZED_RUN_DIR`, else `./runs/<timestamp>`.

| File         | Content                                                                                       |
| ------------ | --------------------------------------------------------------------------------------------- |
| `steps.json` | One JSON line per log interval and for the first step of each session: `step`, `loss`, `grad_norm`, `learning_rate`, `epoch`, `tokens`, `tokens_per_sec`. A masked run adds `scored_tokens`. `trainer.metrics.peak_tflops` adds `mfu`. A non-finite value is `null` |
| `evals.json` | Under `trainer.eval` only. One JSON line per eval: `step`, `loss`, `perplexity`, `batches`, `tokens`, `best_loss`, `since_best` |
| `run.json`   | Written when the run ends: `status`, `error`, `final_step`, `last_loss`, `started_at`, `finished_at`, `device`, `dataset_size`, `config`. `tokens_per_sec` of the last step, when a step ran. Under `trainer.metrics.peak_tflops` also `mfu`. Under `trainer.spike_guard` also `spikes_detected` and `spikes_skipped`, resumed steps included. Under `trainer.eval` also `best_eval_loss` and `evals`, the eval count of this run directory. After an early stop also `stopped_early: true` and `stop_reason` |
| `events.out.tfevents.*` | A `--features tensorboard` build only. One file per session, rank 0 only. Scalars `train/loss`, `train/learning_rate`, `train/grad_norm`, `train/tokens_per_sec` and `train/mfu` at each log step. `eval/loss` and `eval/perplexity` at each eval |

`status` is `completed` or `failed`. A failed run, setup errors included, records the error text in `error`. The process then exits with code 1. An early stop is `completed` and exits with code 0. A run without `trainer.eval` writes no `evals.json` and none of the eval keys.

### TensorBoard

Build with `--features tensorboard` and point TensorBoard at the run directory:

```bash
cargo build --release --features tensorboard
tensorboard --logdir runs/
```

The writer is part of oxidizr and needs no TensorFlow install. Each event file is a TFRecord stream of `tensorflow.Event` records with scalar summaries. The tests decode the files with an in-repo reader. No test loads them in TensorBoard. A resumed session writes a new file. A write error prints one warning and stops the event file. Training continues.

## Choosing the Right Mode

| Situation                  | Mode                  |
| -------------------------- | --------------------- |
| Local development          | Interactive (default) |
| Progress bar not appearing | Headless              |
| CI/CD pipeline             | Headless              |
| Logging to file            | Headless              |
| Quick visual check         | Interactive           |
| Long unattended run        | Headless              |

## Parsing Headless Output

### Python

```python
import json

metrics = []
with open("training.log") as f:
    for line in f:
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue  # the final-save line is plain text
        if "step" in m and "loss" in m:  # step lines only
            metrics.append(m)

# Plot loss curve
import matplotlib.pyplot as plt
steps = [m['step'] for m in metrics]
losses = [m['loss'] for m in metrics]
plt.plot(steps, losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('loss_curve.png')
```

### Bash

```bash
# Run training, capture output
cargo run --release -- -f models/nano.yaml --headless > training.log 2>&1

# Monitor in real-time
tail -f training.log
```

## Troubleshooting

### Progress bar stuck?

```bash
# Switch to headless
cargo run --release -- -f models/nano.yaml --headless
```

### No output at all?

Check stderr for errors:

```bash
cargo run --release -- -f models/nano.yaml --headless 2>&1
```

### Want both TUI and log file?

Use `tee` but note TUI may not render properly:

```bash
# Better: use headless for logging
cargo run --release -- -f models/nano.yaml --headless 2>&1 | tee training.log
```

## Log Interval

Control how often metrics are printed:

```yaml
trainer:
  log_interval: 10 # Print every 10 steps
```

Smaller = more output, larger = less output.

Both modes also log the first step of each session, a resumed session included. `log_interval: 0` logs that first step only.
