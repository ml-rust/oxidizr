# Contributing to oxidizr

## Before You Start

Open an issue before you start on anything non-trivial, such as a new architecture, config key, or CLI flag. Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## Build and Test

Clone `boostr` into the same parent directory as `oxidizr`. The build reads `../boostr` as a path dependency.

```bash
git clone https://github.com/ml-rust/boostr
git clone https://github.com/ml-rust/oxidizr
cd oxidizr

cargo build --release
cargo build --release --features cuda   # CUDA build, needs the CUDA toolkit
cargo test
cargo clippy
cargo fmt --all --check
```

Run `cargo test --features cuda` when your change touches GPU code.

## Conventions

- Return `Result` from fallible functions. Use `?` in library code and avoid `unwrap()`.
- Keep functions focused. Use descriptive variable names.
- Add a comment to any non-obvious algorithm. Cite the paper or reference when one exists.
- Put unit tests in the same file as the code they test, in a `#[cfg(test)] mod tests` block.
- Put integration tests in `tests/`. Test the public API only.

## Project Layout

```
oxidizr/
├── src/
│   ├── main.rs         # CLI entry point
│   ├── lib.rs          # Library root
│   ├── config.rs       # YAML config types and validation
│   ├── model.rs        # Model construction and forward pass
│   ├── trainer.rs      # Training loop
│   ├── data/           # Token loading, batching, prefetch
│   ├── distributed/    # Multi-GPU gradient sync
│   └── commands/       # train, import, merge, pack, push, prep-corpus
├── models/             # Example configs
├── docs/               # Reference documentation
└── hf/                 # Packaged models (git-ignored, except the README)
```

## Pull Requests

A pull request must include:

- A description of the behavior change and the reason for it.
- Tests for the new behavior.
- Passing `cargo fmt --all --check`, `cargo clippy`, and `cargo test`.
- Updates to `docs/configuration.md` or `docs/cli.md` when the change adds a config key or flag.

Keep each pull request to one change. Split unrelated fixes into separate pull requests.
