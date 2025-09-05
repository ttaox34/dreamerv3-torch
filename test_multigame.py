#!/usr/bin/env python3
"""
测试多游戏retro环境功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from pathlib import Path

def test_multigame_env():
    """测试多游戏环境创建"""
    print("Testing MultiGameRetroEnv...")
    
    try:
        from envs.multigame_retro import MultiGameRetroEnv
        
        # 测试游戏列表
        test_games = ["BalloonFight-Nes", "BomberRaid-Sms"]
        
        print(f"Creating multi-game environment with games: {test_games}")
        
        env = MultiGameRetroEnv(
            games=test_games,
            action_repeat=4,
            size=(64, 64),
            grayscale=False,
            seed=42,
            visual_reward=False  # 禁用视觉奖励以简化测试
        )
        
        print(f"Environment created successfully!")
        print(f"Number of games: {len(env)}")
        print(f"Action spaces: {env.num_actions_list}")
        print(f"Max actions: {env.max_num_actions}")
        
        # 测试每个游戏
        for i, game_name in enumerate(test_games):
            print(f"\nTesting game {i}: {game_name}")
            
            # 设置当前游戏
            env.set_current_game(i)
            
            # 重置环境
            obs = env.reset()
            print(f"Observation keys: {list(obs.keys())}")
            print(f"Game ID in observation: {obs['game_id']}")
            print(f"Image shape: {obs['image'].shape}")
            
            # 测试动作空间
            valid_actions = env.get_valid_actions()
            print(f"Valid actions: {len(valid_actions)} actions")
            
            # 执行几个随机动作
            for step in range(5):
                action = np.random.choice(valid_actions)
                obs, reward, done, info = env.step(action)
                print(f"Step {step}: action={action}, reward={reward}, done={done}")
                
                if done:
                    print("Episode finished, resetting...")
                    obs = env.reset()
        
        env.close()
        print("\n✅ MultiGameRetroEnv test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ MultiGameRetroEnv test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multigame_models():
    """测试多游戏模型"""
    print("\nTesting MultiGameImagBehavior...")
    
    try:
        from multigame_models import MultiGameImagBehavior
        
        # 创建模拟配置
        class MockConfig:
            def __init__(self):
                self.dyn_stoch = 32
                self.dyn_deter = 512
                self.dyn_discrete = 32
                self.precision = 32
                self.units = 512
                self.act = "SiLU"
                self.norm = True
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self.actor = {
                    "layers": 2,
                    "dist": "onehot",
                    "std": "none",
                    "min_std": 0.1,
                    "max_std": 1.0,
                    "temp": 0.1,
                    "unimix_ratio": 0.01,
                    "outscale": 1.0,
                    "lr": 0.0001,
                    "eps": 0.0001,
                    "grad_clip": 100.0,
                }
                
                self.critic = {
                    "layers": 2,
                    "dist": "normal",
                    "outscale": 1.0,
                    "lr": 0.0001,
                    "eps": 0.0001,
                    "grad_clip": 100.0,
                    "slow_target": False,
                }
                
                self.weight_decay = 0.0
                self.opt = "adam"
        
        config = MockConfig()
        
        # 创建游戏动作空间
        game_action_spaces = {
            "SonicTheHedgehog-Genesis": 12,
            "Airstriker-Genesis": 16,
        }
        
        print(f"Creating MultiGameImagBehavior with action spaces: {game_action_spaces}")
        
        # 创建模拟世界模型
        class MockWorldModel:
            def get_feat(self, state):
                batch_size = state['stoch'].shape[0]
                return torch.randn(batch_size, config.dyn_stoch + config.dyn_deter)
        
        world_model = MockWorldModel()
        
        # 创建多游戏行为模型
        behavior = MultiGameImagBehavior(config, world_model, game_action_spaces)
        
        print(f"Model created successfully!")
        print(f"Number of actor heads: {len(behavior.actor_heads)}")
        
        # 测试前向传播
        batch_size = 4
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        features = torch.randn(batch_size, feat_size)
        
                
        # 测试不同游戏ID
        game_ids = torch.tensor([0, 0, 1, 1])
        
        actions, logprobs = behavior.get_action(features, game_ids, sample=True)
        
        print(f"Actions shape: {actions.shape}")
        print(f"Logprobs shape: {logprobs.shape}")
        print(f"Actions: {actions}")
        print(f"Logprobs: {logprobs}")
        
        print("\n✅ MultiGameImagBehavior test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ MultiGameImagBehavior test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multigame_dreamer():
    """测试多游戏Dreamer"""
    print("\nTesting MultiGameDreamer...")
    
    try:
        from multigame_dreamer import MultiGameDreamer
        
        print("MultiGameDreamer class imported successfully!")
        
        # 这里只测试导入，因为完整的测试需要环境和数据
        print("\n✅ MultiGameDreamer test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ MultiGameDreamer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🎮 Starting Multi-Game Retro Environment Tests")
    print("=" * 50)
    
    results = []
    
    # 运行测试
    results.append(test_multigame_env())
    results.append(test_multigame_models())
    results.append(test_multigame_dreamer())
    
    # 总结结果
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! Multi-game retro environment is ready!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())