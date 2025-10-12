"""
Script to collect inference data from a trained DreamerV3 model checkpoint.
Saves each step's screenshot and information in the format:
- sample-logdir/{game_name}/step_00000.png
- sample-logdir/{game_name}/step_00000.json
"""
import argparse
import datetime
import os
import sys
import time
import pathlib
import json
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


def create_agent_and_config(game, args, checkpoint):
    """Create agent and config for a specific game"""
    device = torch.device(args.device)
    
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
    reward_mode_cli = getattr(args, 'reward_mode', 'L3')
    
    # Create a new config object for this game
    game_config = argparse.Namespace(**vars(args))
    
    # Now update config with defaults using the same approach as dreamer.py
    for key, value in defaults.items():
        if not hasattr(game_config, key):
            setattr(game_config, key, value)
    
    # Set the specific task to the game being evaluated
    game_config.task = f"retro_{game}"
    
    # Restore command-line reward mode to override config file setting
    game_config.reward_mode = reward_mode_cli
    
    # Manually ensure actor config is correct for discrete actions
    # Since retro games use discrete actions, ensure dist='onehot' and std='none'
    if hasattr(game_config, 'actor') and isinstance(game_config.actor, dict):
        game_config.actor['dist'] = 'onehot'
        game_config.actor['std'] = 'none'
    else:
        game_config.actor = {'dist': 'onehot', 'std': 'none'}
    
    # First, check if we have game_action_spaces saved in the checkpoint
    if 'game_action_spaces' in checkpoint:
        saved_game_action_spaces = checkpoint['game_action_spaces']
        print(f"Game action spaces from checkpoint: {saved_game_action_spaces}")
    else:
        print("Warning: No game_action_spaces found in checkpoint, creating for single game only")
        saved_game_action_spaces = {}
    
    # Determine the maximum action space across all games used in training
    max_action_size = game_config.num_actions if hasattr(game_config, 'num_actions') else 0
    if saved_game_action_spaces:
        max_action_size = max(max(saved_game_action_spaces.values()), max_action_size)
    
    # Create a temporary environment to get observation and action space for current game
    temp_config = type('Config', (), {
        'task': f"retro_{game}",
        'action_repeat': game_config.action_repeat,
        'size': game_config.size,
        'grayscale': getattr(game_config, 'grayscale', False),
        'time_limit': getattr(game_config, 'time_limit', 10000),
        'reward_norm': getattr(game_config, 'reward_norm', 1.0),
        'seed': getattr(game_config, 'seed', 0),
        'device': getattr(game_config, 'device', 'cuda'),
        'reward_mode': getattr(game_config, 'reward_mode', 'L3'),  # Add reward mode
        'visual_encoder': getattr(game_config, 'visual_encoder', 'CLIP')  # Add visual encoder
    })()
    
    temp_env = make_env(temp_config, "eval")
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()
    
    # If max_action_size is still 0 (not set), determine it from the current game
    if max_action_size == 0:
        max_action_size = act_space.n if hasattr(act_space, "n") else act_space.shape[0]
    
    # Make sure game_config.num_actions is set to the maximum across all games
    game_config.num_actions = max_action_size
    
    # Use the saved game_action_spaces if available, otherwise create for current game only
    if saved_game_action_spaces:
        game_action_spaces = saved_game_action_spaces
        # Get the number of buttons for this specific game 
        num_actions_for_game = game_action_spaces[game] if game in game_action_spaces else max(game_action_spaces.values())
    else:
        game_action_spaces = {}
        temp_env = make_env(type('Config', (), {
            'task': f"retro_{game}",
            'action_repeat': game_config.action_repeat,
            'size': game_config.size,
            'grayscale': getattr(game_config, 'grayscale', False),
            'time_limit': getattr(game_config, 'time_limit', 1000),
            'reward_norm': getattr(game_config, 'reward_norm', 1.0),
            'seed': getattr(game_config, 'seed', 0),
            'device': getattr(game_config, 'device', 'cuda'),
            'reward_mode': getattr(game_config, 'reward_mode', 'L3'),  # Add reward mode
            'visual_encoder': getattr(game_config, 'visual_encoder', 'CLIP')  # Add visual encoder
        })(), "eval")
        num_actions_for_game = temp_env.action_space.n if hasattr(temp_env.action_space, "n") else temp_env.action_space.shape[0]
        game_action_spaces[game] = num_actions_for_game
        temp_env.close()
    
    # Print debug info to check actor configuration
    print(f"Actor config after loading: {getattr(game_config, 'actor', 'NOT SET')}")
    if hasattr(game_config, 'actor') and isinstance(game_config.actor, dict):
        print(f"  dist: {game_config.actor.get('dist', 'NOT SET')}")
        print(f"  std: {game_config.actor.get('std', 'NOT SET')}")
    
    # Create model with same architecture as training
    print(f"Creating model for {game}...")
    
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
        game_config,
        logger,  # Use the logger with step 0
        None,  # No training dataset
        game,  # Current game
        game_action_spaces  # Game action spaces
    ).to(device)
    
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
    
    print(f"Model for {game} created successfully!")
    return agent, game_config, game_action_spaces


def collect_data_for_game(agent, game_config, game, game_action_spaces):
    """Collect data for a specific game"""
    device = torch.device(game_config.device)
    
    print(f"Starting data collection on {game} for {game_config.steps_to_collect} steps...")
    
    # Create data collection environment
    eval_env = make_env(game_config, "eval")
    
    # Create output directory for collected data
    base_dir = getattr(game_config, 'base_dir', 'sample-logdir')
    output_dir = pathlib.Path(f"{base_dir}/{game}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    total_collected_steps = 0
    agent_state = None
    
    # Data collection loop
    with torch.no_grad():
        while total_collected_steps < game_config.steps_to_collect:
            if done:
                print(f"Episode completed at step {step_count}. Total collected: {total_collected_steps}/{game_config.steps_to_collect}")
                
                # Reset environment and agent state for next episode
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
                step_count = 0  # Reset step count for the episode

            # Prepare observation with proper batch dimension and all required keys
            # Convert to tensor format with batch dimension
            formatted_obs = {}
            for key, value in obs.items():
                # Convert to tensor first
                if not isinstance(value, torch.Tensor):
                    if isinstance(value, np.ndarray):
                        tensor_val = torch.from_numpy(value).to(device).float()
                    elif isinstance(value, (bool, np.bool_, int, float)):
                        # Convert scalars to arrays with batch dimension
                        value = np.array([value])
                        tensor_val = torch.from_numpy(value).to(device).float()
                    else:
                        tensor_val = torch.tensor(value, device=device).float()
                else:
                    tensor_val = value.to(device).float()
                
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
                formatted_obs["is_terminal"] = torch.zeros_like(formatted_obs["is_first"], dtype=torch.float32, device=device)
            
            if "reward" not in formatted_obs:
                formatted_obs["reward"] = torch.zeros((formatted_obs["image"].shape[0],), dtype=torch.float32, device=device)
            
            # Get action from the agent
            policy_output, agent_state = agent(formatted_obs, np.array([done]), agent_state, training=False)
            
            # We need to convert the tensor action to numpy and remove batch dimension
            action_tensor = policy_output["action"]
            if action_tensor.dim() > 1:
                action_for_env = action_tensor.cpu().numpy()[0]  # Remove batch dimension
            else:
                action_for_env = action_tensor.cpu().numpy()
                
            # Convert discrete action to multibinary action (num_buttons dimension)
            # If we know the number of buttons for this game from the checkpoint, 
            # convert the 2^n dimensional one-hot action to n dimensional multi-hot action
            if game in game_action_spaces:
                total_actions = game_action_spaces[game]  # This is 2^n where n is the number of buttons
                # Calculate the actual number of buttons by taking log base 2
                import math
                n_buttons = int(math.log2(total_actions))
                print(f"Game: {game}, Total actions: {total_actions}, Calculated n_buttons: {n_buttons}, Action shape: {action_for_env.shape if hasattr(action_for_env, 'shape') else len(action_for_env)}")
                try:
                    # The action from the model should be a one-hot vector of size 2^n_buttons
                    # We need to find the index of the '1' value and convert it to binary
                    action_idx = np.argmax(action_for_env)  # Find the index of the '1' value in one-hot vector
                    print(f"Action index: {action_idx}")
                    # Convert the index to binary representation
                    binary_action = np.zeros(n_buttons, dtype=np.int8)
                    for i in range(n_buttons):
                        binary_action[i] = (action_idx >> i) & 1
                    print(f"Binary action shape: {binary_action.shape}, values: {binary_action}")
                    action_to_save = binary_action
                except Exception as e:
                    print(f"Action conversion failed: {e}")
                    action_to_save = action_for_env
            else:
                # If unable to determine number of buttons, use the original action
                action_to_save = action_for_env

            # The environment has SelectAction wrapper which expects a dict with 'action' key
            # So we should pass a dict containing the action
            action_dict_for_env = {"action": action_for_env}
            
            # Debug: Print environment structure
            print(f"Environment type: {type(eval_env)}")
            if hasattr(eval_env, 'env'):
                print(f"eval_env.env type: {type(eval_env.env)}")
                if hasattr(eval_env.env, 'env'):
                    print(f"eval_env.env.env type: {type(eval_env.env.env)}")
                    if hasattr(eval_env.env.env, 'env'):
                        print(f"eval_env.env.env.env type: {type(eval_env.env.env.env)}")
                        if hasattr(eval_env.env.env.env, 'env'):
                            print(f"eval_env.env.env.env.env type: {type(eval_env.env.env.env.env)}")
            
            # Try to access the stable retro environment through multiple unwraps
            raw_frame = None
            current_env = eval_env
            depth = 0
            max_depth = 10  # Increase max depth to ensure we reach the base environment
            env_types = []  # Track environment types to prevent loops
            
            while current_env is not None and depth < max_depth:
                current_type = type(current_env)
                env_types.append(current_type)
                
                if hasattr(current_env, 'get_raw_frame'):
                    raw_frame = current_env.get_raw_frame()
                    if raw_frame is not None:
                        print(f"Found raw frame at depth {depth} in {current_type}: {raw_frame.shape if hasattr(raw_frame, 'shape') else 'N/A'}")
                        break
                
                # Check for common access patterns in gym/wrappers
                if hasattr(current_env, 'env'):
                    next_env = current_env.env
                    # Prevent infinite loops by checking if we're going back to the same environment
                    if next_env is current_env or type(next_env) in env_types:
                        break
                    current_env = next_env
                elif hasattr(current_env, 'unwrapped') and current_env.unwrapped is not current_env:
                    current_env = current_env.unwrapped
                else:
                    break  # No more nested environments
                depth += 1
            
            if raw_frame is None:
                print(f"Could not access raw frame after checking {depth} levels, will use obs image after step")
                print(f"Visited environment types: {[str(t) for t in env_types]}")
            
            obs, reward, done, info = eval_env.step(action_dict_for_env)
            
            # If we couldn't get the raw frame before the step, try again after the step
            # This might get the frame from the current step (not previous) if implemented correctly
            if raw_frame is None:
                current_env = eval_env
                depth = 0
                env_types = []  # Reset for second search
                while current_env is not None and depth < max_depth:
                    current_type = type(current_env)
                    env_types.append(current_type)
                    
                    if hasattr(current_env, 'get_raw_frame'):
                        raw_frame = current_env.get_raw_frame()
                        if raw_frame is not None:
                            print(f"Found raw frame after step at depth {depth} in {current_type}: {raw_frame.shape if hasattr(raw_frame, 'shape') else 'N/A'}")
                            break
                    
                    # Check for common access patterns in gym/wrappers
                    if hasattr(current_env, 'env'):
                        next_env = current_env.env
                        if next_env is current_env or type(next_env) in env_types:
                            break
                        current_env = next_env
                    elif hasattr(current_env, 'unwrapped') and current_env.unwrapped is not current_env:
                        current_env = current_env.unwrapped
                    else:
                        break  # No more nested environments
                    depth += 1

            # If raw frame is still not available, use the current observation
            if raw_frame is None:
                if isinstance(obs, dict) and 'image' in obs:
                    raw_frame = obs['image'].copy()  # Store the original observation image
                    print(f"Using obs image, shape: {raw_frame.shape if hasattr(raw_frame, 'shape') else 'N/A'}")
                else:
                    raw_frame = obs.copy()
                    print(f"Using obs directly, type: {type(raw_frame)}")
            
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
            
            # Save the current step's data
            step_image_path = output_dir / f"step_{total_collected_steps:05d}.png"
            step_json_path = output_dir / f"step_{total_collected_steps:05d}.json"
            
            # Save the raw frame (original size) as image
            print(f"Saving image, raw_frame shape: {raw_frame.shape if hasattr(raw_frame, 'shape') else 'N/A'}, type: {type(raw_frame)}")
            if raw_frame is not None:
                if isinstance(raw_frame, torch.Tensor):
                    frame = raw_frame.cpu().numpy()
                    print(f"Converted from tensor to numpy, shape: {frame.shape}")
                else:
                    frame = raw_frame
                    print(f"Using directly, shape: {frame.shape if hasattr(frame, 'shape') else 'N/A'}")

                # Ensure values are in [0, 255] range
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                        print(f"Normalized to [0, 255] and converted to uint8")
                    else:
                        frame = frame.astype(np.uint8)
                        print(f"Converted to uint8")

                print(f"Final frame shape before saving: {frame.shape}")
                img = Image.fromarray(frame)
                img.save(step_image_path)
            else:
                # Fallback to the processed image if raw frame is not available
                current_obs_image = obs.get('image', None)
                if isinstance(current_obs_image, torch.Tensor):
                    frame = current_obs_image.cpu().numpy()
                else:
                    frame = current_obs_image

                # Convert to [H, W, C] format if needed
                if frame is not None and len(frame.shape) == 3 and frame.shape[-1] not in [3, 4]:
                    frame = frame.transpose(1, 2, 0)  # From [C, H, W] to [H, W, C]

                # Ensure values are in [0, 255] range
                if frame is not None and frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)

                if frame is not None:
                    img = Image.fromarray(frame)
                    img.save(step_image_path)

            # Save the JSON metadata
            step_metadata = {
                "game_name": game,
                "rl_model": "DreamerV3",
                "reward_mode": game_config.reward_mode,
                "timestamp": datetime.datetime.now().isoformat(),
                "step": total_collected_steps,
                "observation_image_path": f"step_{total_collected_steps:05d}.png",
                "action": action_to_save.tolist(),  # Convert numpy array to list (multibinary action)
                "reward": float(reward),
                "terminated": bool(done),
                "truncated": False,  # We're not using truncation in this implementation
                "info": info  # Include any additional environment info
            }
            
            with open(step_json_path, 'w') as f:
                json.dump(step_metadata, f, indent=2)
            
            total_reward += reward
            step_count += 1
            total_collected_steps += 1
            
            if total_collected_steps % 1000 == 0:
                print(f"Collected {total_collected_steps}/{game_config.steps_to_collect} steps for {game}")
    
    eval_env.close()
    
    print(f"Data collection completed for {game}. Collected {total_collected_steps} steps.")


def main(args):
    device = torch.device(args.device)
    print(f"Loading checkpoint from: {args.ckpt_path}")
    print(f"Games to collect data from: {args.game}")
    print(f"Reward mode: {getattr(args, 'reward_mode', 'L3')}")
    print(f"Target steps per game: {args.steps_to_collect}")
    
    # Load the checkpoint to get saved game information
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    
    # Get game_action_spaces from checkpoint if available
    saved_game_action_spaces = checkpoint.get('game_action_spaces', {}) if 'game_action_spaces' in checkpoint else {}
    
    # Process each game using the same loaded model
    for game in args.game:
        print(f"\nProcessing game: {game}")
        
        # Create agent and config for this specific game
        agent, game_config, game_action_spaces = create_agent_and_config(game, args, checkpoint)
        
        # Collect data for this game
        collect_data_for_game(agent, game_config, game, game_action_spaces)
        
        # Clean up the agent to free memory
        del agent
        torch.cuda.empty_cache()  # if using GPU


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--game", type=str, nargs='+', required=True, 
                       help="Game name(s) to collect data from. Can specify multiple games separated by spaces.")
    parser.add_argument("--steps_to_collect", type=int, default=10000, help="Number of steps to collect data for per game")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--reward_mode", choices=["L1", "L2", "L3"], default="L3", 
                       help="Reward mode: L1 (survival), L2 (visual), L3 (original env) (default: L3)")
    parser.add_argument("--base_dir", type=str, default="sample-logdir", 
                       help="Base directory to save collected data (default: sample-logdir)")
    
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