
#!/usr/bin/env python
import argparse
import functools
import os
import pathlib
import sys
import time

import numpy as np
import ruamel.yaml as yaml
import torch
import imageio

# Add project root to path to allow imports
sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
from dreamer import make_env, Dreamer, count_steps

def discrete_to_multibinary(discrete_action, n_buttons):
    """Converts a discrete action index into a multi-binary button vector."""
    binary_action = np.zeros(n_buttons, dtype=np.int8)
    action_int = int(discrete_action)
    for i in range(n_buttons):
        binary_action[i] = (action_int >> i) & 1
    return binary_action


def main(config):
    # --- Setup ---
    # Provide a default logdir if it's not set, to prevent pathlib errors.
    if config.logdir is None:
        config.logdir = './eval_logdir'

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    output_dir = pathlib.Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Logdir: {logdir}")
    print(f"Video Output Dir: {output_dir}")

    # --- Environment ---
    print(f"Creating environment for {config.task}")
    env = make_env(config, "eval", 0)

    # Set num_actions from env action space, which is required by the model
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    print(f"Action Space Size (num_actions): {config.num_actions}")
    
    # --- Agent ---
    # Force compile to be False for evaluation
    config.compile = False

    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        logger=None,  # No logger needed for inference
        dataset=None, # No dataset needed for inference
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    # --- Load Checkpoint ---
    if not pathlib.Path(config.checkpoint).exists():
        print(f"ERROR: Checkpoint not found at {config.checkpoint}")
        return

    print(f"Loading checkpoint from {config.checkpoint}")
    checkpoint = torch.load(config.checkpoint, map_location=config.device)

    # Clean state dict keys from `_orig_mod.` prefix added by torch.compile
    cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["agent_state_dict"].items()}

    agent.load_state_dict(cleaned_state_dict)
    agent.eval() # Set agent to evaluation mode
    print("Checkpoint loaded successfully.")

    # --- Simulation and Recording ---
    obs = env.reset()
    done = False
    agent_state = None
    frames = [obs['image']]
    actions_log = []
    
    print("Starting simulation...")
    while not done:
        # The agent policy needs the full observation dictionary, with tensors.
        obs_for_agent = {k: torch.from_numpy(v).unsqueeze(0).to(config.device) for k, v in obs.items() if k == 'image'}
        obs_for_agent['is_first'] = torch.tensor([obs['is_first']], device=config.device)
        obs_for_agent['is_terminal'] = torch.tensor([obs['is_terminal']], device=config.device)

        with torch.no_grad():
            policy_output, agent_state = agent._policy(obs_for_agent, agent_state, training=True)
        
        action_onehot = policy_output['action'].squeeze(0).cpu().numpy()
        logits = policy_output['logits'].squeeze(0).cpu().numpy()
        actions_log.append({'onehot': action_onehot, 'logits': logits})

        # The environment wrappers expect the action to be in a dictionary.
        obs, reward, done, info = env.step({'action': action_onehot})
        frames.append(obs['image'])

    print(f"Episode finished after {len(frames)} steps.")

    # --- Generate common filename ---
    base_filename = f"{config.task.replace('_', '-')}-{config.reward_mode}-{time.strftime('%Y%m%d-%H%M%S')}"

    # --- Save Actions ---
    actions_path = output_dir / f"{base_filename}_actions.txt"
    n_buttons = env._n_buttons if hasattr(env, '_n_buttons') else 9 # Get button count from env
    print(f"Saving actions to {actions_path}")
    with open(actions_path, 'w') as f:
        for i, log in enumerate(actions_log):
            discrete_action = np.argmax(log['onehot'])
            multibinary_action = discrete_to_multibinary(discrete_action, n_buttons)
            sorted_logits = np.sort(log['logits'])[::-1]
            
            f.write(f"Step {i+1}:\n")
            f.write(f"  - Buttons:       {np.array2string(multibinary_action)}\n")
            f.write(f"  - Action Index:  {discrete_action}\n")
            f.write(f"  - Logit for [0]:   {log['logits'][0]:.4f}\n")
            f.write(f"  - Top 5 Logits:  {np.array2string(sorted_logits[:5], precision=4, suppress_small=True)}\n\n")
    print("Actions saved successfully.")

    # --- Save Video ---
    video_path = output_dir / f"{base_filename}.mp4"
    print(f"Saving video to {video_path}")
    imageio.mimsave(video_path, frames, fps=30)
    print("Video saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--configs", nargs="+", required=True, help="List of config names (e.g., vjepa_retro_vitl)")
    parser.add_argument("--game", type=str, required=True, help="Full name of the retro game (e.g., retro_SuperMarioBros-Nes)")
    parser.add_argument("--reward_mode", type=str, default="L3", choices=["L1", "L2", "L3"], help="Reward mode to use for the agent")
    parser.add_argument("--output_dir", type=str, default="./eval_videos", help="Directory to save the output video")
    
    # Parse known args to get the script-specific ones
    args, remaining = parser.parse_known_args()

    # Load base configs from YAML
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    # Apply config hierarchy
    name_list = ["defaults", *args.configs]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    # Override specific parameters from command line
    defaults['task'] = args.game
    defaults['reward_mode'] = args.reward_mode

    # Create a new parser to parse all dreamer-related args from the config
    parser2 = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser2.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    
    # The final config is a combination of YAML, command-line overrides, and any other remaining args
    final_config = parser2.parse_args(remaining)
    final_config.checkpoint = args.checkpoint
    final_config.output_dir = args.output_dir

    main(final_config)
