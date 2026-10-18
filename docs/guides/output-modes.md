# Output Modes

Oxidizr supports two output modes for training progress.

## Interactive Mode (Default)

Shows a TUI (Terminal User Interface) progress bar with live metrics:

```
Training ████████████████░░░░░░░░░░░░░░░░ 50% | Step 500/1000 | Loss: 8.234 | 45.2 it/s | ETA: 11s
```

**Features:**

- Real-time loss tracking
- Training speed (iterations/second)
- ETA to completion
- Memory usage indicators
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
{"step": 1, "loss": 11.5161, "grad_norm": 0.2542, "learning_rate": 2.000000e-3, "epoch": 0.01, "samples/s": 3.24, "vram": "0"}
{"step": 21, "loss": 11.4948, "grad_norm": 0.1980, "learning_rate": 2.000000e-3, "epoch": 0.21, "samples/s": 3.09, "vram": "0"}
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
| `vram`          | Device memory in use (MiB), `"0"` when unknown |

## Run Directory Files

Both modes write the same two files to the run directory. Set it with `--run-dir`. The default is `$OXIDIZED_RUN_DIR`, else `./runs/<timestamp>`.

| File         | Content                                                                                       |
| ------------ | --------------------------------------------------------------------------------------------- |
| `steps.json` | One JSON line per log interval and for the first step of each session: `step`, `loss`, `grad_norm`, `learning_rate`, `epoch` |
| `run.json`   | Written when the run ends: `status`, `error`, `final_step`, `last_loss`, `started_at`, `finished_at`, `device`, `dataset_size`, `config` |

`status` is `completed` or `failed`. A failed run, setup errors included, records the error text in `error`. The process then exits with code 1.

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
        metrics.append(json.loads(line))

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
