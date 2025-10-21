# Repository Guidelines

## Project Structure & Module Organization
`dreamer.py` orchestrates training loops, drawing models from `models.py` and `networks.py`, helpers from `tools.py`, and async utilities from `parallel.py`. Environment wrappers and setup helpers for Atari, Crafter, Minecraft, and Retro live in `envs/`; keep new integrations alongside existing modules. Preset stacks are defined in `configs.yaml` and shared by both `dreamer.py` and `model_evaluate_dreamer.py`. Research artifacts sit in `archives/`, visual assets in `imgs/`, and long-running jobs emit checkpoints and TensorBoard data under `./logdir/<task>` (created at runtime).

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: create an isolated Python 3.11 workspace before installing dependencies.
- `pip install -r requirements.txt`: pull core libraries (PyTorch, dm-control, minerl, ruamel.yaml, etc.).
- `python dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk`: launch a standard DMC Vision run; swap `--configs` for presets such as `retro` or `debug`.
- `python model_evaluate_dreamer.py --checkpoint ./logdir/dmc_walker_walk/latest.pt --configs retro --episodes 3`: run the deterministic evaluator against a saved policy for quick regression checks.
- `tensorboard --logdir ./logdir`: monitor rewards, reconstruction loss, and entropy metrics during training.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and ≤88-character lines. Use snake_case for functions, modules, and config keys (`visual_reward_weight`, `make_env`), and UpperSnakeCase for constants. Keep configuration additions declarative in `configs.yaml` and document overrides inline. Handle devices through `config.device` and helper utilities instead of hard-coded `.to(...)` calls. Aim for explicit imports from local modules to avoid circular dependencies.

## Testing Guidelines
No dedicated unit suite exists; rely on fast configs to validate changes. Use `--configs debug` for a short smoke run before scaling up. Reproduce environment integrations by running `model_evaluate_dreamer.py` with a known checkpoint and limited `--episodes`. When touching exploration, reward, or visualization code, confirm expected ranges by inspecting TensorBoard scalars (`actor_entropy`, `reward_pred`). Note any reproducibility flags or seeds in commit messages or PRs.

## Commit & Pull Request Guidelines
Commits follow the concise imperative style seen in history (`add stable-retro to envs`, `avoid ".to(device)"`). Group related edits, update config defaults in the same change, and keep commits focused. PRs should include: a summary of intent, the exact commands/configs used for validation, relevant before/after metrics or plots, and references to linked issues or papers. Flag breaking configuration shifts in the description and update README or inline comments as needed.

## Environment & Config Tips
Set `MUJOCO_GL=osmesa` and `SDL_VIDEODRIVER=dummy` when running headless—these are already enforced in entry scripts but required for new executables. Extend presets by appending top-level keys in `configs.yaml`, reusing defaults through the override pattern. Place supplementary visual-reward assets beside `envs/visual_reward_wrapper.py` and archive large checkpoints under `archives/` to keep runtime logs compact.
