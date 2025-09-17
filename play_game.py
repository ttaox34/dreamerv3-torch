#!/usr/bin/env python3
"""
Play games with trained DreamerV3 checkpoints and save gameplay as video
"""

import argparse
import functools
import os
import pathlib
import sys
import cv2
import numpy as np

import torch
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import tools
import envs.wrappers as wrappers
from parallel import Damy
import dreamer

to_np = lambda x: x.detach().cpu().numpy()

# Define make_dataset function locally since it's not available in tools
def make_dataset(episodes, config):
    import tools
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    """Create environment for gameplay"""
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc
        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari
        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze
        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter
        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft
        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == "retro":
        import envs.stable_retro as stable_retro
        env = stable_retro.StableRetro(
            game=task,
            action_repeat=config.action_repeat,
            size=config.size,
            grayscale=config.grayscale,
            seed=config.seed + id,
        )
        # For retro games, we need to set render_mode before wrapping
        if hasattr(env, '_env'):
            env._env.render_mode = 'rgb_array'
        # Don't use OneHotAction wrapper for retro games with continuous action spaces
        # Skip the OneHotAction wrapper and handle action conversion manually
    else:
        raise NotImplementedError(suite)

    # Set render mode for video recording before wrapping
    # Note: We'll handle rendering during gameplay step instead

    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)

    return env


def load_agent_from_checkpoint(checkpoint_path, config, env):
    """Load trained agent from checkpoint"""
    import models

    # Create agent
    agent = models.ImagBehavior(config, env._wm if hasattr(env, '_wm') else None)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    # Load agent state
    if 'agent_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['agent_state_dict'])
    elif 'state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['state_dict'])
    else:
        # Try to load world model directly
        if hasattr(env, '_wm'):
            env._wm.load_state_dict(checkpoint)

    agent.to(config.device)
    agent.eval()

    return agent


def save_video(frames, output_path, fps=30):
    """Save frames as MP4 video"""
    if len(frames) == 0:
        print("Warning: No frames to save")
        return

    # Get frame dimensions
    height, width = frames[0].shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        out.write(frame)

    out.release()
    print(f"Video saved to: {output_path}")


def play_game_with_checkpoint(checkpoint_path, config, output_video_path, max_steps=10000):
    """Play game using trained checkpoint and save as video"""
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Create environment
    env = make_env(config, "eval", 0)
    # Don't use Damy wrapper as it causes issues with function returns

  
    # Set num_actions for the config
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Load agent
    try:
        # Try to load complete agent first
        import models

        # Create minimal dataset for agent initialization
        # This is required for the Dreamer agent but won't be used during gameplay
        dummy_episodes = [{'image': np.zeros((64, 64, 3), dtype=np.uint8),
                          'action': np.zeros(1, dtype=np.int32),
                          'reward': 0.0,
                          'discount': 1.0,
                          'is_first': True,
                          'is_last': False,
                          'is_terminal': False}]
        dummy_dataset = make_dataset(dummy_episodes, config)

        # Create Dreamer agent
        logger = tools.Logger(pathlib.Path(config.logdir) / "play", 0)
        agent = dreamer.Dreamer(
            env.observation_space,
            env.action_space,
            config,
            logger,
            dummy_dataset,
        ).to(config.device)

        # Load checkpoint
        if 'agent_state_dict' in checkpoint:
            agent.load_state_dict(checkpoint['agent_state_dict'])
        else:
            agent.load_state_dict(checkpoint)

        agent.eval()
        agent.requires_grad_(False)

    except Exception as e:
        print(f"Failed to load complete agent: {e}")
        print("Trying to load world model only...")

        # Fallback: load world model only
        import models
        wm = models.WorldModel(env.observation_space, env.action_space, 0, config)

        # Handle nested checkpoint structure
        actual_checkpoint = checkpoint
        if 'agent_state_dict' in checkpoint:
            actual_checkpoint = checkpoint['agent_state_dict']

        # Load world model state
        wm.load_state_dict(actual_checkpoint)
        wm.to(config.device)
        wm.eval()

        # Create simple agent with world model
        agent = wm

    # Gameplay loop
    obs = env.reset()
    # Handle if obs is a tuple (new gym API)
    if isinstance(obs, tuple):
        obs = obs[0]
    # Handle if obs is a function (Damy wrapper)
    elif callable(obs):
        obs = obs()

    # Ensure obs is in the correct format for the agent
    if not isinstance(obs, dict):
        # Convert numpy array to dict format expected by Dreamer
        # Add batch dimension to image (1, H, W, C)
        image_tensor = torch.tensor(obs, dtype=torch.float32, device=config.device)
        if len(image_tensor.shape) == 3:  # H, W, C
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        obs = {
            'image': image_tensor,
            'is_first': torch.tensor([step == 0], dtype=torch.bool, device=config.device),
            'is_last': torch.tensor([False], dtype=torch.bool, device=config.device),
            'is_terminal': torch.tensor([False], dtype=torch.bool, device=config.device),
        }
    else:
        # Ensure required fields are present and are tensors
        if 'image' in obs:
            if not torch.is_tensor(obs['image']):
                obs['image'] = torch.tensor(obs['image'], dtype=torch.float32, device=config.device)
            if len(obs['image'].shape) == 3:  # H, W, C
                obs['image'] = obs['image'].unsqueeze(0)  # Add batch dimension

        if 'is_first' not in obs:
            obs['is_first'] = torch.tensor([step == 0], dtype=torch.bool, device=config.device)
        elif not torch.is_tensor(obs['is_first']):
            obs['is_first'] = torch.tensor([obs['is_first']], dtype=torch.bool, device=config.device)

        if 'is_last' not in obs:
            obs['is_last'] = torch.tensor([False], dtype=torch.bool, device=config.device)
        elif not torch.is_tensor(obs['is_last']):
            obs['is_last'] = torch.tensor([obs['is_last']], dtype=torch.bool, device=config.device)

        if 'is_terminal' not in obs:
            obs['is_terminal'] = torch.tensor([False], dtype=torch.bool, device=config.device)
        elif not torch.is_tensor(obs['is_terminal']):
            obs['is_terminal'] = torch.tensor([obs['is_terminal']], dtype=torch.bool, device=config.device)
    state = None
    frames = []
    total_reward = 0
    step = 0

    print("Starting gameplay...")

    while step < max_steps:
        # Get action from agent
        try:
            if hasattr(agent, '__call__'):
                policy_output, state = agent(obs, [False], state, training=False)
                # Handle different output formats
                if isinstance(policy_output, dict) and 'action' in policy_output:
                    action = policy_output['action']
                else:
                    action = policy_output
            else:
                # Simple policy for world model only
                raise NotImplementedError("World model-only policy is not implemented. The loaded checkpoint should contain a complete agent.")
        except Exception as e:
            raise RuntimeError(f"Failed to get action from agent: {e}. This indicates a problem with the agent or observation format.") from e

        # Make sure action is numpy array for environment
        if torch.is_tensor(action):
            action = action.cpu().numpy()
            if action.ndim == 0:
                action = action.item() if hasattr(env.action_space, 'n') else action.reshape(1)

        # Handle action format based on wrapper configuration
        current_env = env
        while hasattr(current_env, 'env'):
            if hasattr(current_env, '_key') and current_env._key:
                # Found the SelectAction wrapper - convert to discrete action
                if isinstance(action, np.ndarray):
                    if len(action.shape) > 0 and action.size > 1:
                        discrete_action = np.argmax(action)
                    elif len(action.shape) > 0:
                        discrete_action = action.item()
                    else:
                        discrete_action = int(action)
                elif isinstance(action, (int, np.integer)):
                    discrete_action = action
                elif torch.is_tensor(action):
                    if action.numel() > 1:
                        discrete_action = action.argmax().item()
                    else:
                        discrete_action = action.item()
                else:
                    raise ValueError(f"Cannot convert action format: {type(action)}. Expected tensor, numpy array, or integer. Action value: {action}")
                action = {current_env._key: discrete_action}
                break
            current_env = current_env.env

        # Take action in environment
        step_result = env.step(action)

        if len(step_result) == 5:
            # New gym API
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            # Old gym API
            obs, reward, done, info = step_result

        # Handle if obs is a tuple (new gym API)
        if isinstance(obs, tuple):
            obs = obs[0]
        # Handle if obs is a function (Damy wrapper)
        elif callable(obs):
            obs = obs()

        # Ensure obs is in the correct format for the agent
        if not isinstance(obs, dict):
            # Convert numpy array to dict format expected by Dreamer
            # Add batch dimension to image (1, H, W, C)
            image_tensor = torch.tensor(obs, dtype=torch.float32, device=config.device)
            if len(image_tensor.shape) == 3:  # H, W, C
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

            obs = {
                'image': image_tensor,
                'is_first': torch.tensor([step == 0], dtype=torch.bool, device=config.device),
                'is_last': torch.tensor([False], dtype=torch.bool, device=config.device),
                'is_terminal': torch.tensor([False], dtype=torch.bool, device=config.device),
            }
        else:
            # Ensure required fields are present and are tensors
            if 'image' in obs:
                if not torch.is_tensor(obs['image']):
                    obs['image'] = torch.tensor(obs['image'], dtype=torch.float32, device=config.device)
                if len(obs['image'].shape) == 3:  # H, W, C
                    obs['image'] = obs['image'].unsqueeze(0)  # Add batch dimension

            if 'is_first' not in obs:
                obs['is_first'] = torch.tensor([step == 0], dtype=torch.bool, device=config.device)
            elif not torch.is_tensor(obs['is_first']):
                obs['is_first'] = torch.tensor([obs['is_first']], dtype=torch.bool, device=config.device)

            if 'is_last' not in obs:
                obs['is_last'] = torch.tensor([False], dtype=torch.bool, device=config.device)
            elif not torch.is_tensor(obs['is_last']):
                obs['is_last'] = torch.tensor([obs['is_last']], dtype=torch.bool, device=config.device)

            if 'is_terminal' not in obs:
                obs['is_terminal'] = torch.tensor([False], dtype=torch.bool, device=config.device)
            elif not torch.is_tensor(obs['is_terminal']):
                obs['is_terminal'] = torch.tensor([obs['is_terminal']], dtype=torch.bool, device=config.device)

        total_reward += reward.item() if torch.is_tensor(reward) else reward

        # Render frame
        frame = None
        try:
            # Try different rendering approaches
            if hasattr(env, 'render'):
                try:
                    frame = env.render(mode='rgb_array')
                except TypeError:
                    # Some wrappers don't accept mode parameter
                    try:
                        frame = env.render()
                    except Exception:
                        frame = None
                except Exception:
                    frame = None

            # If still no frame, try accessing underlying environment
            if frame is None:
                # Navigate through wrapper chain to find the actual environment
                current_env = env
                while hasattr(current_env, 'env'):
                    current_env = current_env.env

                    # First try get_raw_frame() for StableRetro environments
                    if hasattr(current_env, 'get_raw_frame'):
                        frame = current_env.get_raw_frame()
                        if frame is not None:
                            break

                    # Then try render() methods
                    if hasattr(current_env, 'render'):
                        try:
                            frame = current_env.render(mode='rgb_array')
                        except TypeError:
                            try:
                                frame = current_env.render()
                            except Exception:
                                frame = None
                        except Exception:
                            frame = None
                        if frame is not None:
                            break

            if frame is not None:
                frames.append(frame)

        except Exception:
            # Continue without frame if rendering fails
            pass

        step += 1

        # Check if episode is done
        if done:
            print(f"Episode finished at step {step}, total reward: {total_reward}")
            break

        if step % 1000 == 0:
            print(f"Step {step}, Reward: {total_reward}")

    print(f"Gameplay finished. Total steps: {step}, Total reward: {total_reward}")

    # Save video
    if frames:
        save_video(frames, output_video_path, fps=30)
    else:
        print("No frames captured during gameplay")

    env.close()
    return total_reward, step


def main():
    parser = argparse.ArgumentParser(description="Play games with DreamerV3 checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--configs", nargs="+", default=["defaults"],
                       help="Configuration names")
    parser.add_argument("--task", type=str, required=True,
                       help="Task to play (e.g., dmc_walker_walk, atari_breakout)")
    parser.add_argument("--output", type=str, default="gameplay.mp4",
                       help="Output video path")
    parser.add_argument("--max_steps", type=int, default=10000,
                       help="Maximum steps to play")
    parser.add_argument("--logdir", type=str, default="./logdir",
                       help="Log directory")

    args = parser.parse_args()

    # Load configuration
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    defaults = {}
    for name in ["defaults", *args.configs]:
        recursive_update(defaults, configs[name])

    # Override task
    defaults["task"] = args.task
    defaults["logdir"] = args.logdir

    # Set device
    if torch.cuda.is_available():
        defaults["device"] = "cuda:0"
    else:
        defaults["device"] = "cpu"

    # Convert to namespace
    from types import SimpleNamespace
    config = SimpleNamespace(**defaults)

    print(f"Playing task: {config.task}")
    print(f"Using device: {config.device}")

    # Play game and save video
    total_reward, steps = play_game_with_checkpoint(
        args.checkpoint, config, args.output, args.max_steps
    )

    print(f"Gameplay complete!")
    print(f"Total reward: {total_reward}")
    print(f"Steps taken: {steps}")
    print(f"Video saved to: {args.output}")


if __name__ == "__main__":
    main()