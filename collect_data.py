#!/usr/bin/env python
import argparse
import functools
import os
import pathlib
import sys
import time
import json
from datetime import datetime

import numpy as np
import ruamel.yaml as yaml
import torch
import imageio

# Add project root to path to allow imports
sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
from dreamer import make_env, Dreamer

def discrete_to_multibinary(discrete_action, n_buttons):
    """Converts a discrete action index into a multi-binary button vector."""
    binary_action = np.zeros(n_buttons, dtype=np.int8)
    action_int = int(discrete_action)
    for i in range(n_buttons):
        binary_action[i] = (action_int >> i) & 1
    return binary_action

def main(config):
    # --- Setup ---
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    # Create a unique output directory for this collection run
    run_timestamp = time.strftime('%Y%m%d-%H%M%S')
    game_name_safe = config.task.replace('_', '-')
    output_dir = pathlib.Path(config.output_dir).expanduser() / f"{game_name_safe}-{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving data to: {output_dir}")

    # --- Environment ---
    env = make_env(config, "eval", 0)
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    n_buttons = env._n_buttons if hasattr(env, '_n_buttons') else 9

    # Traverse the wrapper stack to find the env with the get_raw_frame method
    unwrapped_env = env
    while not hasattr(unwrapped_env, 'get_raw_frame'):
        if not hasattr(unwrapped_env, 'env'):
            raise RuntimeError("Could not find an environment with get_raw_frame in the wrapper stack.")
        unwrapped_env = unwrapped_env.env

    # --- Agent ---
    config.compile = False
    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        logger=None,
        dataset=None,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    # --- Load Checkpoint ---
    print(f"Loading checkpoint from {config.checkpoint}")
    checkpoint = torch.load(config.checkpoint, map_location=config.device)
    cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["agent_state_dict"].items()}
    agent.load_state_dict(cleaned_state_dict)
    agent.eval()
    print("Checkpoint loaded successfully.")

    # --- Data Collection Loop ---
    total_steps_collected = 0
    episode_count = 0

    while total_steps_collected < config.steps:
        episode_count += 1
        print(f"Starting Episode {episode_count}... Steps collected: {total_steps_collected}/{config.steps}")
        obs = env.reset()
        done = False
        agent_state = None

        # Handle the very first frame of the episode from reset
        first_raw_frame = unwrapped_env.get_raw_frame()

        while not done and total_steps_collected < config.steps:
            # --- Prepare data for saving ---
            step_data = {}
            step_data['game_name'] = config.task.replace('retro_', '')
            step_data['rl_model'] = 'DreamerV3+VJEPA' # Or a more specific name
            step_data['reward_mode'] = config.reward_mode
            step_data['timestamp'] = datetime.utcnow().isoformat() + 'Z'
            step_data['step'] = total_steps_collected
            
            # --- Save raw image ---
            # Use the frame from the previous step/reset
            raw_frame = first_raw_frame if obs['is_first'] else info.get('raw_frame')
            image_path = f"step_{total_steps_collected:06d}.png"
            step_data['observation_image_path'] = image_path
            if raw_frame is not None:
                imageio.imwrite(output_dir / image_path, raw_frame)
            else:
                print(f"Warning: raw_frame at step {total_steps_collected} is None.")

            # --- Agent policy step ---
            obs_for_agent = {k: torch.from_numpy(v).unsqueeze(0).to(config.device) for k, v in obs.items() if k == 'image'}
            obs_for_agent['is_first'] = torch.tensor([obs['is_first']], device=config.device)
            obs_for_agent['is_terminal'] = torch.tensor([obs['is_terminal']], device=config.device)

            with torch.no_grad():
                policy_output, agent_state = agent._policy(obs_for_agent, agent_state, training=False)
            
            action_onehot = policy_output['action'].squeeze(0).cpu().numpy()
            discrete_action = np.argmax(action_onehot)
            multibinary_action = discrete_to_multibinary(discrete_action, n_buttons)
            step_data['action'] = multibinary_action.tolist()

            # --- Environment step ---
            obs, reward, done, info = env.step({'action': action_onehot})
            
            # --- Finalize and save JSON ---
            step_data['reward'] = float(reward)
            step_data['terminated'] = bool(done) # Simplified, assumes done is termination
            step_data['truncated'] = info.get('TimeLimit.truncated', False)
            step_data['info'] = {k: v.item() if hasattr(v, 'item') else v for k, v in info.items() if k != 'raw_frame'}

            json_path = f"step_{total_steps_collected:06d}.json"
            with open(output_dir / json_path, 'w') as f:
                json.dump(step_data, f, indent=4)

            total_steps_collected += 1

    print(f"\nCollection complete. Saved {total_steps_collected} steps to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--configs", nargs="+", required=True, help="List of config names (e.g., vjepa_retro_vitl)")
    parser.add_argument("--game", type=str, required=True, help="Full name of the retro game (e.g., retro_SuperMarioBros-Nes)")
    parser.add_argument("--reward_mode", type=str, default="L3", choices=["L1", "L2", "L3"], help="Reward mode to use for the agent")
    parser.add_argument("--output_dir", required=True, help="Root directory to save the output data")
    parser.add_argument("--steps", type=int, default=10000, help="Total number of steps to collect")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to run the agent on")
    
    args, remaining = parser.parse_known_args()

    configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text())
    
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    defaults['task'] = args.game
    defaults['reward_mode'] = args.reward_mode

    parser2 = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser2.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    
    final_config = parser2.parse_args(remaining)
    final_config.checkpoint = args.checkpoint
    final_config.output_dir = args.output_dir
    final_config.steps = args.steps
    final_config.device = args.device

    main(final_config)
