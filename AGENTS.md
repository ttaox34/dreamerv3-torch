# Repository Guidelines

## Project Structure & Module Organization
DreamerV3 training orchestration lives in `dreamer.py`, with world-model components split across `models.py`, `networks.py`, and `vjepa_models.py`. Shared utilities and logging helpers sit in `tools.py`, while environment adapters and setup helpers are under `envs/`. The consolidated hyperparameter catalog is `configs.yaml`; override keys via CLI flags rather than editing defaults. Artifacts land in `logdir/` (training) and `eval_logdir/`; reuse `sample-logdir/` as a template. Pretrained checkpoints and assets reside in `pretrained-models/` and `imgs/`. The `batch_*` scripts and `collect_data.py` support large-scale evaluation and data harvesting.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: create an isolated Python 3.11 environment.
- `pip install -r requirements.txt`: install runtime dependencies such as PyTorch and dm-control extras.
- `python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk`: launch a training run with the bundled config stack.
- `python3 evaluate.py --configs dmc_vision --checkpoint ./logdir/dmc_walker_walk/latest.pt`: score a saved agent checkpoint.
- `python3 batch_collect.py --reward_mode L2 --logdir ./logdir --output_dir ./sample-logdir --steps 10000`: harvest retro rollouts from the newest checkpoints.
- `tensorboard --logdir ./logdir`: inspect metrics, reconstructions, and video predictions.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and trailing newlines. Keep modules and functions in snake_case, classes in PascalCase, and CLI or YAML keys lowercase with underscores to match `configs.yaml`. Prefer explicit type hints for public APIs and document non-obvious tensor shapes inline. Run `python3 -m black .` and `python3 -m isort . --profile black` to keep formatting consistent, and group imports as (stdlib, third-party, local).

## Testing Guidelines
There is no central pytest suite in the root project; validate changes with targeted smoke runs. For trainer or model edits, run a short job such as `python3 dreamer.py --configs dmc_proprio --steps 5000 --logdir ./logdir/debug_run` and confirm losses and checkpoint rotation. For evaluation or replay tweaks, run `python3 evaluate.py ...` against a recent checkpoint and diff the scalar outputs. Record TensorBoard screenshots or key scalar deltas when behavior changes.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects (e.g., `add trajectory collection`, `add vit-l support`) without trailing punctuation. Keep commits focused on one concern, update accompanying configs or docs when behavior shifts, and note dependency bumps explicitly. Pull requests should explain the motivation, list the commands used for validation, and attach logs or plots for training-visible adjustments. Reference related issues when available and flag any long-running jobs reviewers must reproduce.
