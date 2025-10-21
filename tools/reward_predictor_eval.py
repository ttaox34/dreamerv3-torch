import argparse
import json
import math
import pathlib
from collections import defaultdict
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from ruamel import yaml

import models


class AttrDict(dict):
    """Dictionary with attribute-style access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def load_config(config_names: Sequence[str]) -> AttrDict:
    config_path = pathlib.Path(__file__).resolve().parent.parent / "configs.yaml"
    configs = yaml.safe_load(config_path.read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    merged: Dict[str, object] = {}
    name_list = ["defaults", *config_names] if config_names else ["defaults"]
    for name in name_list:
        if name not in configs:
            raise KeyError(f"Config '{name}' not found in configs.yaml")
        recursive_update(merged, configs[name])
    return AttrDict(merged)


def discover_episode_dirs(root: pathlib.Path) -> List[pathlib.Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if any(root.glob("step_*.json")):
        return [root]
    episodes = [
        path
        for path in root.rglob("*")
        if path.is_dir() and any(path.glob("step_*.json"))
    ]
    if not episodes:
        raise FileNotFoundError(f"No step_*.json files found under {root}")
    episodes.sort()
    return episodes


def load_step(step_path: pathlib.Path) -> Dict[str, object]:
    with step_path.open("r", encoding="utf-8") as fp:
        step = json.load(fp)
    step["__path__"] = step_path
    return step


def buttons_to_index(buttons: Sequence[int]) -> int:
    value = 0
    for idx, pressed in enumerate(buttons):
        if pressed:
            value |= (1 << idx)
    return value


def index_to_one_hot(index: int, dim: int) -> np.ndarray:
    one_hot = np.zeros(dim, dtype=np.float32)
    one_hot[index] = 1.0
    return one_hot


def load_episode(
    directory: pathlib.Path,
    target_size: Sequence[int],
    num_actions: int,
) -> Dict[str, np.ndarray]:
    steps = sorted(directory.glob("step_*.json"))
    if not steps:
        raise FileNotFoundError(f"No step_*.json files in {directory}")

    images: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    discounts: List[float] = []
    is_first: List[bool] = []
    is_terminal: List[bool] = []

    for idx, step_path in enumerate(sorted(steps)):
        step = load_step(step_path)
        image_path = directory / step["observation_image_path"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Failed to load image {image_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        width, height = int(target_size[0]), int(target_size[1])
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        images.append(frame.astype(np.uint8))

        buttons = step.get("action")
        if not isinstance(buttons, Sequence):
            raise TypeError(f"Expected action sequence in {step_path}")
        index = buttons_to_index(buttons)
        actions.append(index_to_one_hot(index, num_actions))

        rewards.append(float(step.get("reward", 0.0)))
        terminated = bool(step.get("terminated", False))
        discounts.append(0.0 if terminated else 1.0)
        is_terminal.append(terminated)
        is_first.append(idx == 0)

    data = {
        "image": np.stack(images)[None],  # (1, T, H, W, C)
        "action": np.stack(actions)[None],  # (1, T, num_actions)
        "reward": np.array(rewards, dtype=np.float32)[None],  # (1, T)
        "discount": np.array(discounts, dtype=np.float32)[None],  # (1, T)
        "is_terminal": np.array(is_terminal, dtype=np.float32)[None],  # (1, T)
        "is_first": np.array(is_first, dtype=np.float32)[None],  # (1, T)
    }
    return data


def build_spaces(image_shape: Sequence[int], num_actions: int):
    import gym

    height, width, channels = image_shape
    obs_space = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(
                low=0, high=255, shape=(height, width, channels), dtype=np.uint8
            )
        }
    )
    act_space = gym.spaces.Box(
        low=0.0, high=1.0, shape=(num_actions,), dtype=np.float32
    )
    return obs_space, act_space


def evaluate_episode(wm: models.WorldModel, data: Dict[str, np.ndarray]):
    with torch.no_grad():
        tensors = wm.preprocess(data)
        embed = wm.encoder(tensors)
        post, _ = wm.dynamics.observe(
            embed, tensors["action"], tensors["is_first"]
        )
        feat = wm.dynamics.get_feat(post)
        reward_dist = wm.heads["reward"](feat)
        reward_mean = reward_dist.mean().squeeze(-1)
        log_prob = reward_dist.log_prob(tensors["reward"])

    target = tensors["reward"]
    mae = torch.abs(reward_mean - target).sum().item()
    mse = torch.square(reward_mean - target).sum().item()
    nll = -log_prob.sum().item()
    steps = target.numel()
    return {
        "mae_sum": mae,
        "mse_sum": mse,
        "nll_sum": nll,
        "steps": steps,
        "pred": reward_mean.cpu().numpy(),
        "target": target.cpu().numpy(),
    }


def load_world_model(
    checkpoint_path: pathlib.Path,
    config: AttrDict,
    obs_space,
    act_space,
) -> models.WorldModel:
    wm = models.WorldModel(obs_space, act_space, step=0, config=config).to(
        config.device
    )
    state = torch.load(checkpoint_path, map_location=config.device)
    wm_state = {}
    for key, value in state["agent_state_dict"].items():
        if key.startswith("_wm."):
            wm_state[key[len("_wm.") :]] = value
    missing, unexpected = wm.load_state_dict(wm_state, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys in world model state: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in world model state: {unexpected}")
    wm.eval()
    return wm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Dreamer reward predictor on offline trajectories."
    )
    parser.add_argument(
        "--data-root",
        nargs="+",
        required=True,
        help="One or more directories containing step_*.json and corresponding images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Path to Dreamer checkpoint (latest.pt).",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=["retro"],
        help="Additional config overrides (default: retro).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of episodes to evaluate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "latest.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    config = load_config(args.configs)
    config.device = args.device
    size = config.size
    if isinstance(size, (list, tuple)):
        config.size = [int(size[0]), int(size[1])]
    else:
        config.size = [int(size), int(size)]

    episode_dirs: List[pathlib.Path] = []
    for root in args.data_root:
        episode_dirs.extend(discover_episode_dirs(pathlib.Path(root)))
    if args.limit:
        episode_dirs = episode_dirs[: args.limit]

    if not episode_dirs:
        raise RuntimeError("No episodes discovered.")

    # Peek first episode for shapes and action dimension.
    first_dir = episode_dirs[0]
    step_files = sorted(first_dir.glob("step_*.json"))
    if not step_files:
        raise RuntimeError(f"No steps found in {first_dir}")
    first_step = load_step(step_files[0])
    num_buttons = len(first_step.get("action", []))
    if num_buttons == 0:
        raise ValueError("Action field is empty in sample data.")
    config.num_actions = 1 << num_buttons
    image_path = first_dir / first_step["observation_image_path"]
    first_frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise ValueError(f"Failed to load reference image {image_path}")
    channels = first_frame.shape[2] if first_frame.ndim == 3 else 1

    target_size = config.size if isinstance(config.size, list) else list(config.size)
    obs_space, act_space = build_spaces(
        (int(target_size[1]), int(target_size[0]), channels), config.num_actions
    )
    world_model = load_world_model(checkpoint_path, config, obs_space, act_space)

    aggregate = defaultdict(float)
    per_episode = []
    for directory in episode_dirs:
        episode = load_episode(directory, target_size, config.num_actions)
        result = evaluate_episode(world_model, episode, config.device)
        for key in ("mae_sum", "mse_sum", "nll_sum", "steps"):
            aggregate[key] += result[key]
        per_episode.append(
            {
                "episode_dir": str(directory),
                "steps": int(result["steps"]),
                "mae": result["mae_sum"] / max(1, result["steps"]),
                "rmse": math.sqrt(result["mse_sum"] / max(1, result["steps"])),
                "nll": result["nll_sum"] / max(1, result["steps"]),
            }
        )

    total_steps = aggregate["steps"]
    metrics = {
        "episodes": len(per_episode),
        "steps": int(total_steps),
        "mae": aggregate["mae_sum"] / max(1, total_steps),
        "rmse": math.sqrt(aggregate["mse_sum"] / max(1, total_steps)),
        "nll": aggregate["nll_sum"] / max(1, total_steps),
        "per_episode": per_episode,
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
