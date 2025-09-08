#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import retro
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from dreamer import StochasticFrameSkip, MultiBinaryToDiscreteWrapper
import gymnasium
from gymnasium.wrappers import TimeLimit

def make_simple_retro_env():
    """创建简单的retro环境用于测试"""
    game_name = 'Battletoads-Nes'
    
    def _init():
        # Create retro environment
        env = retro.make(game_name, state=retro.State.DEFAULT, render_mode='rgb_array')
        print(f"Original action space: {env.action_space}")
        
        # Add frame skip wrapper
        env = StochasticFrameSkip(env, n=4, stickprob=0.25)
        
        # Add time limit
        env = TimeLimit(env, max_episode_steps=4500)
        
        # Add PPO-style wrappers
        env = WarpFrame(env)
        print(f"After WarpFrame action space: {env.action_space}")
        
        env = ClipRewardEnv(env)
        print(f"After ClipRewardEnv action space: {env.action_space}")
        
        # Convert MultiBinary to Discrete
        if hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0:
            n_actions = min(env.action_space.shape[0], 12)
            env = MultiBinaryToDiscreteWrapper(env, n_actions)
            print(f"After MultiBinaryToDiscreteWrapper action space: {env.action_space}")
        
        return env
    
    return _init

def test_simple_env():
    """测试简单环境"""
    print("Testing simple retro environment...")
    
    # Create SubprocVecEnv
    env_fn = make_simple_retro_env()
    single_vec_env = SubprocVecEnv([env_fn])
    
    print("Created SubprocVecEnv")
    
    # Test reset
    obs = single_vec_env.reset()
    print(f"Reset observation shape: {obs.shape}")
    
    # Test step
    action_space = single_vec_env.action_space
    print(f"Action space: {action_space}")
    
    # Create a random action
    action = [single_vec_env.action_space.sample()]
    print(f"Action: {action}")
    
    try:
        result = single_vec_env.step(action)
        print(f"Step result length: {len(result)}")
        print("✅ Step successful!")
        
        obs, rewards, dones, infos = result
        print(f"Obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        print(f"Rewards: {rewards}")
        print(f"Dones: {dones}")
        print(f"Infos: {type(infos)}")
        
    except Exception as e:
        print(f"❌ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    single_vec_env.close()
    print("✅ Simple environment test passed!")
    return True

if __name__ == "__main__":
    test_simple_env()
