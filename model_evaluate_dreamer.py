"""
Script to evaluate DreamerV3 model from a checkpoint on a specified game.
Can run inference and save gameplay videos.
"""
import argparse
import datetime
import os
import sys
import time
import pathlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import ruamel.yaml as yaml

import dreamer
import tools
import networks
import models


def make_env(config, mode, id=0, **kwargs):
    from envs import wrappers
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
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
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
        
        # Add reward mode wrapper if enabled
        if hasattr(config, 'reward_mode') and config.reward_mode in ["L1", "L2", "L3"]:
            try:
                from reward_manager import RewardManager, SharedVisualEncoder
                
                # Get game name from task
                game_name = task.replace("retro_", "")  # Remove "retro_" prefix to get game name
                
                # Create reward manager with appropriate settings
                reward_manager = RewardManager(
                    reward_mode=config.reward_mode,
                    games_to_train=[game_name],  # Single game context
                    visual_encoder=getattr(config, 'visual_encoder', 'CLIP'),
                    device=getattr(config, 'device', 'cuda')
                )
                
                from envs.wrappers import RewardModeWrapper
                env = RewardModeWrapper(env, reward_manager, game_name)
                print(f"Reward mode {config.reward_mode} enabled for retro environment: {game_name}")
            except Exception as e:
                print(f"Warning: Failed to enable reward mode {config.reward_mode}: {e}")
                print("Continuing with original reward...")
        
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.OBS.wrappers.UUID(env)
    return env


def main(config):
    device = torch.device(config.device)
    print(f"Loading checkpoint from: {config.ckpt_path}")
    print(f"Game to evaluate: {config.game}")
    print(f"Reward mode: {getattr(config, 'reward_mode', 'L3')}")
    
    # Load the checkpoint to get saved game information
    checkpoint = torch.load(config.ckpt_path, map_location=device)
    
    # Load default configuration from configs.yaml similar to dreamer.py
    configs_yaml_path = pathlib.Path(__file__).parent / "configs.yaml"
    configs = yaml.safe_load(configs_yaml_path.read_text())
    
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    # Load configs similar to dreamer.py
    # Get the retro config as base (since we're evaluating retro games)
    # We need to ensure retro configs properly override defaults
    defaults = {}
    # First load defaults
    recursive_update(defaults, configs["defaults"])
    # Then override with retro-specific settings
    recursive_update(defaults, configs["retro"])
    
    # Store command-line value for reward_mode to ensure it's preserved
    reward_mode_cli = getattr(config, 'reward_mode', 'L3')
    
    # Now update config with defaults using the same approach as dreamer.py
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    
    # Set the specific task to the game being evaluated
    config.task = f"retro_{config.game}"
    
    # Restore command-line reward mode to override config file setting
    config.reward_mode = reward_mode_cli
    
    # Manually ensure actor config is correct for discrete actions
    # Since retro games use discrete actions, ensure dist='onehot' and std='none'
    if hasattr(config, 'actor') and isinstance(config.actor, dict):
        config.actor['dist'] = 'onehot'
        config.actor['std'] = 'none'
    else:
        config.actor = {'dist': 'onehot', 'std': 'none'}
    
    # First, check if we have game_action_spaces saved in the checkpoint
    if 'game_action_spaces' in checkpoint:
        saved_game_action_spaces = checkpoint['game_action_spaces']
        print(f"Game action spaces from checkpoint: {saved_game_action_spaces}")
    else:
        print("Warning: No game_action_spaces found in checkpoint, creating for single game only")
        saved_game_action_spaces = {}
    
    # Determine the maximum action space across all games used in training
    max_action_size = config.num_actions if hasattr(config, 'num_actions') else 0
    if saved_game_action_spaces:
        max_action_size = max(max(saved_game_action_spaces.values()), max_action_size)
    
    # Create a temporary environment to get observation and action space for current game
    temp_config = type('Config', (), {
        'task': f"retro_{config.game}",
        'action_repeat': config.action_repeat,
        'size': config.size,
        'grayscale': getattr(config, 'grayscale', False),
        'time_limit': getattr(config, 'time_limit', 10000),
        'reward_norm': getattr(config, 'reward_norm', 1.0),
        'seed': getattr(config, 'seed', 0),
        'device': getattr(config, 'device', 'cuda'),
        'reward_mode': getattr(config, 'reward_mode', 'L3'),  # Add reward mode
        'visual_encoder': getattr(config, 'visual_encoder', 'CLIP')  # Add visual encoder
    })()
    
    temp_env = make_env(temp_config, "eval")
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()
    
    # If max_action_size is still 0 (not set), determine it from the current game
    if max_action_size == 0:
        max_action_size = act_space.n if hasattr(act_space, "n") else act_space.shape[0]
    
    # Make sure config.num_actions is set to the maximum across all games
    config.num_actions = max_action_size
    
    # Use the saved game_action_spaces if available, otherwise create for current game only
    if saved_game_action_spaces:
        game_action_spaces = saved_game_action_spaces
    else:
        game_action_spaces = {}
        temp_env = make_env(type('Config', (), {
            'task': f"retro_{config.game}",
            'action_repeat': config.action_repeat,
            'size': config.size,
            'grayscale': getattr(config, 'grayscale', False),
            'time_limit': getattr(config, 'time_limit', 1000),
            'reward_norm': getattr(config, 'reward_norm', 1.0),
            'seed': getattr(config, 'seed', 0),
            'device': getattr(config, 'device', 'cuda'),
            'reward_mode': getattr(config, 'reward_mode', 'L3'),  # Add reward mode
            'visual_encoder': getattr(config, 'visual_encoder', 'CLIP')  # Add visual encoder
        })(), "eval")
        game_action_spaces[config.game] = temp_env.action_space.n if hasattr(temp_env.action_space, "n") else temp_env.action_space.shape[0]
        temp_env.close()
    
    # Print debug info to check actor configuration
    print(f"Actor config after loading: {getattr(config, 'actor', 'NOT SET')}")
    if hasattr(config, 'actor') and isinstance(config.actor, dict):
        print(f"  dist: {config.actor.get('dist', 'NOT SET')}")
        print(f"  std: {config.actor.get('std', 'NOT SET')}")
    
    # Create model with same architecture as training
    print("Creating model...")
    
    # We need to create a mock logger and set the step counter
    import tools
    
    # Create a temporary logdir for the logger
    temp_logdir = pathlib.Path('./temp_eval_logdir')
    temp_logdir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger with a default step count
    logger = tools.Logger(temp_logdir, 0)  # Initialize with step 0
    
    agent = dreamer.Dreamer(
        obs_space,
        act_space,
        config,
        logger,  # Use the logger with step 0
        None,  # No training dataset
        config.game,  # Current game
        game_action_spaces  # Game action spaces
    ).to(config.device)
    
    # Load the checkpoint with strict=False to handle missing keys for other games
    try:
        agent.load_state_dict(checkpoint['agent_state_dict'], strict=False)
        print("Partially loaded state_dict. Some keys may have been ignored.")
    except Exception as e:
        print(f"Error during loading: {e}")
        print("This is likely due to architecture differences between training and evaluation.")
        print("Loading available weights while ignoring mismatched dimensions...")
        
        # Load available weights while skipping mismatched layers
        model_dict = agent.state_dict()
        pretrained_dict = checkpoint['agent_state_dict']
        
        # Filter out keys with size mismatches
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            elif k in model_dict:
                print(f"Skipping {k} due to size mismatch: checkpoint {v.shape} vs model {model_dict[k].shape}")
        
        model_dict.update(filtered_dict)
        agent.load_state_dict(model_dict)
        print(f"Loaded {len(filtered_dict)} parameters, skipped {len(pretrained_dict) - len(filtered_dict)} due to size mismatch.")
    
    agent.eval()
    
    print("Model loaded successfully!")
    
    # Create evaluation environment
    eval_env = make_env(config, "eval")
    
    print(f"Starting evaluation on {config.game}...")
    
    # Initialize environment
    reset_result = eval_env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result  # Handle gymnasium-style return
    else:
        obs = reset_result  # Handle old gym-style return
    done = False
    if "is_first" not in obs:
        obs["is_first"] = np.array([True], dtype=np.bool_)
    else:
        obs["is_first"] = np.asarray(obs["is_first"])
    
    # Ensure is_first is the proper shape (batch_size, ) for the model
    if not isinstance(obs["is_first"], np.ndarray):
        obs["is_first"] = np.array([obs["is_first"]], dtype=np.bool_)
    elif np.isscalar(obs["is_first"]):
        obs["is_first"] = np.array([obs["is_first"]], dtype=np.bool_)
    elif obs["is_first"].shape == ():
        obs["is_first"] = np.array([obs["is_first"].item()], dtype=np.bool_)
    
    total_reward = 0
    step_count = 0
    episode_count = 0
    agent_state = None
    
    # For video recording
    frames = []
    if config.save_video:
        print("Recording video...")
    
    # Evaluation loop
    with torch.no_grad():
        while episode_count < config.episodes:
            if done:
                print(f"Episode {episode_count + 1} completed. Total reward: {total_reward:.2f}, Steps: {step_count}")
                episode_count += 1
                total_reward = 0
                step_count = 0
                
                # Reset environment and agent state
                reset_result = eval_env.reset()
                if isinstance(reset_result, tuple):
                    obs, _ = reset_result  # Handle gymnasium-style return
                else:
                    obs = reset_result  # Handle old gym-style return
                done = False
                if "is_first" not in obs:
                    obs["is_first"] = np.array([True], dtype=np.bool_)
                else:
                    obs["is_first"] = np.asarray(obs["is_first"])
                
                # Ensure is_first is the proper shape (batch_size, ) for the model
                if not isinstance(obs["is_first"], np.ndarray):
                    obs["is_first"] = np.array([obs["is_first"]], dtype=np.bool_)
                elif np.isscalar(obs["is_first"]):
                    obs["is_first"] = np.array([obs["is_first"]], dtype=np.bool_)
                elif obs["is_first"].shape == ():
                    obs["is_first"] = np.array([obs["is_first"].item()], dtype=np.bool_)
                
                agent_state = None  # Reset agent state
                
                if config.save_video and episode_count < config.episodes:
                    # Save the current episode
                    video_filename = f"{config.game}_episode_{episode_count}.mp4"
                    save_frames_as_video(frames, video_filename, config.fps)
                    frames = []  # Clear frames for next episode

            # Prepare observation with proper batch dimension and all required keys
            # Convert to tensor format with batch dimension
            formatted_obs = {}
            for key, value in obs.items():
                # Convert to tensor first
                if not isinstance(value, torch.Tensor):
                    if isinstance(value, np.ndarray):
                        tensor_val = torch.from_numpy(value).to(config.device).float()
                    elif isinstance(value, (bool, np.bool_, int, float)):
                        # Convert scalars to arrays with batch dimension
                        value = np.array([value])
                        tensor_val = torch.from_numpy(value).to(config.device).float()
                    else:
                        tensor_val = torch.tensor(value, device=config.device).float()
                else:
                    tensor_val = value.to(config.device).float()
                
                # Ensure batch dimension exists
                # For image data (usually has 3+ dimensions: H, W, C or B, H, W, C)
                if key == 'image':
                    if tensor_val.dim() == 3:  # Missing batch dimension (H, W, C)
                        tensor_val = tensor_val.unsqueeze(0)  # Add batch dim -> (1, H, W, C)
                    elif tensor_val.dim() == 4:  # Has batch dimension (B, H, W, C)
                        pass  # Correct format
                # For boolean flags
                elif key in ['is_first', 'is_terminal']:
                    if tensor_val.dim() == 0:  # Scalar
                        tensor_val = tensor_val.unsqueeze(0)  # Add batch dim
                    elif tensor_val.dim() == 1:  # Already has batch dim
                        pass  # Correct format
                # For other data types
                else:
                    if tensor_val.dim() == 1:  # Missing batch dimension
                        tensor_val = tensor_val.unsqueeze(0)  # Add batch dim
                        
                formatted_obs[key] = tensor_val
            
            # Add missing keys that are expected by the model
            if "is_terminal" not in formatted_obs:
                formatted_obs["is_terminal"] = torch.zeros_like(formatted_obs["is_first"], dtype=torch.float32, device=config.device)
            
            if "reward" not in formatted_obs:
                formatted_obs["reward"] = torch.zeros((formatted_obs["image"].shape[0],), dtype=torch.float32, device=config.device)
            
            # Get action from the agent
            policy_output, agent_state = agent(formatted_obs, np.array([done]), agent_state, training=False)
            
            # We need to convert the tensor action to numpy and remove batch dimension
            action_tensor = policy_output["action"]
            if action_tensor.dim() > 1:
                action_for_env = action_tensor.cpu().numpy()[0]  # Remove batch dimension
            else:
                action_for_env = action_tensor.cpu().numpy()
                
            # The environment has SelectAction wrapper which expects a dict with 'action' key
            # So we should pass a dict containing the action
            action_dict_for_env = {"action": action_for_env}
            
            obs, reward, done, info = eval_env.step(action_dict_for_env)
            
            # Add is_terminal flag to obs after step
            if "is_terminal" not in obs:
                obs["is_terminal"] = np.array([done], dtype=np.float32)
            if "is_first" not in obs:
                obs["is_first"] = np.array([False], dtype=np.bool_)
                
            # Ensure is_first is the proper shape (batch_size, ) for the model
            if not isinstance(obs["is_first"], np.ndarray):
                obs["is_first"] = np.array([obs["is_first"]], dtype=np.bool_)
            elif np.isscalar(obs["is_first"]):
                obs["is_first"] = np.array([obs["is_first"]], dtype=np.bool_)
            elif obs["is_first"].shape == ():
                obs["is_first"] = np.array([obs["is_first"].item()], dtype=np.bool_)
            
            total_reward += reward
            step_count += 1
            
            # Record frame if saving video
            if config.save_video:
                # Convert observation to image format
                if isinstance(obs['image'], torch.Tensor):
                    frame = obs['image'].cpu().numpy()
                else:
                    frame = obs['image']
                
                # Convert to [H, W, C] format if needed
                if len(frame.shape) == 3 and frame.shape[-1] not in [3, 4]:
                    frame = frame.transpose(1, 2, 0)  # From [C, H, W] to [H, W, C]
                
                # Ensure values are in [0, 255] range
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                
                frames.append(frame)
    
    # Save final episode if needed
    if config.save_video and frames:
        video_filename = f"{config.game}_episode_{episode_count + 1}.mp4"
        save_frames_as_video(frames, video_filename, config.fps)
    
    eval_env.close()
    
    print(f"Evaluation completed. Evaluated {episode_count} episodes.")


def save_frames_as_video(frames, filename, fps=30):
    """Save frames as a video file using ffmpeg."""
    # Try to use imageio-ffmpeg to save video
    try:
        import imageio
        imageio.mimwrite(filename, frames, fps=fps)
        print(f"Video saved as {filename}")
    except ImportError:
        print("imageio not available, saving frames as individual images instead")
        os.makedirs(f"frames_{filename.split('.')[0]}", exist_ok=True)
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame.astype('uint8'))
            img.save(f"frames_{filename.split('.')[0]}/{i:06d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--game", type=str, required=True, help="Game name to evaluate")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to evaluate")
    parser.add_argument("--save_video", action="store_true", help="Save gameplay video")
    parser.add_argument("--fps", type=int, default=60, help="FPS for saved video")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--reward_mode", choices=["L1", "L2", "L3"], default="L3", 
                       help="Reward mode: L1 (survival), L2 (visual), L3 (original env) (default: L3)")
    
    # Add default configurations that match the training config
    parser.add_argument("--action_repeat", type=int, default=4, help="Action repeat")
    parser.add_argument("--size", type=int, nargs=2, default=[64, 64], help="Image size as H W")
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale images")
    parser.add_argument("--time_limit", type=int, default=10000, help="Time limit for episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    # Add other config parameters needed by the model
    parser.add_argument("--num_actions", type=int, default=4096, help="Number of actions (max)")  
    # Note: This should be set to the max action space across all games used in multi-game training
    
    args = parser.parse_args()
    
    main(args)