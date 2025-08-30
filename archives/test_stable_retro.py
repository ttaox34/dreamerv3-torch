#!/usr/bin/env python3
"""
Test script for stable-retro integration with DreamerV3
"""

import argparse
import sys
import os
import numpy as np

# Set up headless display for gym environments
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['DISPLAY'] = ':99'

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import envs.stable_retro as stable_retro
import envs.wrappers as wrappers


def test_stable_retro_env(game="Airstriker-Genesis", steps=100):
    """Test stable-retro environment creation and basic functionality"""
    print(f"Testing stable-retro environment with game: {game}")
    
    try:
        # Create environment
        env = stable_retro.StableRetro(
            game=game,
            action_repeat=4,
            size=(84, 84),
            grayscale=False,
            seed=42,
        )
        
        # Wrap with DreamerV3 wrappers
        env = wrappers.OneHotAction(env)
        env = wrappers.TimeLimit(env, 1000)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)
        
        print(f"Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test environment
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        total_reward = 0
        for step in range(steps):
            # Sample random action and convert to one-hot
            action_space = env.action_space
            if hasattr(action_space, 'discrete') and action_space.discrete:
                # One-hot action space
                action_idx = np.random.randint(0, action_space.shape[0])
                raw_action = np.zeros(action_space.shape[0], dtype=np.float32)
                raw_action[action_idx] = 1.0
            else:
                raw_action = env.action_space.sample()
            
            # Wrap in dict for SelectAction wrapper
            action = {"action": raw_action}
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if step % 20 == 0:
                print(f"Step {step}: reward={reward:.2f}, total_reward={total_reward:.2f}")
            
            if done:
                print(f"Episode finished at step {step}")
                obs = env.reset()
                total_reward = 0
        
        env.close()
        print("Test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure stable-retro is installed: pip install stable-retro")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dreamerv3_integration():
    """Test DreamerV3 integration with stable-retro"""
    print("Testing DreamerV3 integration...")
    
    try:
        # Import DreamerV3 modules
        import dreamer
        
        # Create a minimal config for testing
        class Config:
            def __init__(self):
                self.task = "retro_Airstriker-Genesis"
                self.action_repeat = 4
                self.size = (84, 84)
                self.grayscale = False
                self.seed = 42
                self.time_limit = 1000
        
        config = Config()
        
        # Test environment creation through DreamerV3
        env = dreamer.make_env(config, "train", 0)
        print("DreamerV3 environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test basic functionality
        obs = env.reset()
        action_space = env.action_space
        if hasattr(action_space, 'discrete') and action_space.discrete:
            # One-hot action space
            action_idx = np.random.randint(0, action_space.shape[0])
            raw_action = np.zeros(action_space.shape[0], dtype=np.float32)
            raw_action[action_idx] = 1.0
        else:
            raw_action = env.action_space.sample()
        
        action = {"action": raw_action}
        obs, reward, done, info = env.step(action)
        
        env.close()
        print("DreamerV3 integration test passed!")
        return True
        
    except Exception as e:
        print(f"DreamerV3 integration test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test stable-retro integration")
    parser.add_argument("--game", default="Airstriker-Genesis", help="Game to test")
    parser.add_argument("--steps", type=int, default=100, help="Number of test steps")
    parser.add_argument("--test-dreamer", action="store_true", help="Test DreamerV3 integration")
    
    args = parser.parse_args()
    
    print("Starting stable-retro integration tests...")
    
    # Test basic environment
    success1 = test_stable_retro_env(args.game, args.steps)
    
    # Test DreamerV3 integration if requested
    success2 = True
    if args.test_dreamer:
        success2 = test_dreamerv3_integration()
    
    if success1 and success2:
        print("\nAll tests passed! ✓")
        return 0
    else:
        print("\nSome tests failed! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
