#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from dreamer import make_retro_game_env, VecEnvWrapper, SingleVecEnvWrapper
try:
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    STABLE_BASELINES3_AVAILABLE = False

class SimpleConfig:
    def __init__(self):
        self.action_repeat = 4
        self.time_limit = 108000

def test_env_interface():
    """测试环境接口是否正确"""
    print("Testing environment interface...")
    
    if not STABLE_BASELINES3_AVAILABLE:
        print("stable-baselines3 not available")
        return False
    
    # 创建测试环境
    config = SimpleConfig()
    game = 'Battletoads-Nes'
    
    # Create environment function
    env_fn = make_retro_game_env(game, config, max_episode_steps=4500)
    
    # Create SubprocVecEnv
    print("Creating SubprocVecEnv...")
    single_vec_env = SubprocVecEnv([env_fn])
    
    # Wrap with VecFrameStack
    print("Adding VecFrameStack...")
    vec_env_with_stack = VecFrameStack(single_vec_env, n_stack=4)
    
    # Add image transpose
    print("Adding VecTransposeImage...")
    final_vec_env = VecTransposeImage(vec_env_with_stack)
    
    # Create wrapper for Dreamer
    print("Creating Dreamer wrapper...")
    env = SingleVecEnvWrapper(final_vec_env)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # 测试 reset
    print("Testing reset...")
    obs = env.reset()
    print(f"Reset returned: {type(obs)}")
    print(f"Reset observation keys: {obs.keys() if isinstance(obs, dict) else 'Not a dict'}")
        
    # 测试 step
    print("Testing step...")
    # Create a proper one-hot action
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1.0  # Set first action to 1
    result = env.step(action)
    print(f"Step result type: {type(result)}")
    print(f"Step result length: {len(result) if hasattr(result, '__len__') else 'No length'}")
    
    print("\n✅ Environment interface test passed!")
    return True

if __name__ == "__main__":
    test_env_interface()
