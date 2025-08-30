#!/usr/bin/env python3
"""
测试DreamerV3中的视觉奖励功能
"""
import os
import sys
import numpy as np
import torch

# 设置路径
sys.path.append('/home/zhuolifeng/dreamerv3-torch')

def test_visual_reward_wrapper():
    """测试视觉奖励包装器"""
    print("🧪 测试视觉奖励包装器...")
    
    try:
        # 创建基础环境
        import envs.stable_retro as stable_retro
        base_env = stable_retro.StableRetro(
            game="Airstriker-Genesis",
            size=(64, 64),
            action_repeat=1,
            seed=42
        )
        
        # 测试包装器
        from envs.visual_reward_wrapper import VisualRewardWrapper, DummyVisualRewardWrapper
        
        print("  ✅ 创建视觉奖励包装器...")
        visual_env = VisualRewardWrapper(
            base_env,
            visual_encoder="ResNet",  # 使用ResNet替代CLIP以避免PyTorch版本问题
            visual_reward_weight=0.1,
            episode_length=100,
            device="auto"
        )
        
        print("  🔄 测试环境重置...")
        obs = visual_env.reset()
        print(f"     观察空间: {type(obs)}")
        if isinstance(obs, dict):
            print(f"     包含键: {list(obs.keys())}")
            if 'image' in obs:
                print(f"     图像形状: {obs['image'].shape}")
        
        print("  🎮 测试环境步进...")
        for step in range(5):
            action = visual_env.action_space.sample()
            obs, reward, done, info = visual_env.step(action)
            
            print(f"     步数 {step+1}: reward={reward:.4f}")
            if isinstance(info, dict):
                env_reward = info.get('env_reward', 0.0)
                visual_reward = info.get('visual_reward', 0.0)
                print(f"       环境奖励: {env_reward:.4f}")
                print(f"       视觉奖励: {visual_reward:.4f}")
            
            if done:
                print("     Episode 结束")
                break
        
        visual_env.close()
        print("  ✅ 视觉奖励包装器测试完成")
        
    except Exception as e:
        print(f"  ❌ 视觉奖励包装器测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_dreamer_env_creation():
    """测试DreamerV3环境创建"""
    print("\n🧪 测试DreamerV3环境创建...")
    
    try:
        # 导入必要模块
        import dreamer
        import tools
        import argparse
        
        print("  📝 设置配置...")
        # 创建最小配置
        class Config:
            def __init__(self):
                self.task = "retro_Airstriker-Genesis"
                self.size = [64, 64]
                self.action_repeat = 4
                self.grayscale = False
                self.seed = 42
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.time_limit = 1000  # 添加缺失的time_limit属性
                
                # 视觉奖励配置
                self.visual_reward = True
                self.visual_encoder = "ResNet"  # 使用ResNet替代CLIP
                self.visual_reward_weight = 0.1
                self.visual_episode_length = 100
                self.visual_device = "auto"
        
        config = Config()
        
        print("  🏗️  创建环境...")
        env = dreamer.make_env(config, "train", 0)
        
        print("  🔄 测试环境重置...")
        obs = env.reset()
        print(f"     观察: {type(obs)}")
        if isinstance(obs, dict):
            print(f"     包含键: {list(obs.keys())}")
        
        print("  🎮 测试环境步进...")
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            print(f"     步数 {step+1}: reward={reward:.4f}")
            if hasattr(info, '__contains__') and 'visual_reward' in info:
                print(f"       包含视觉奖励信息")
            
            if done:
                print("     Episode 结束")
                break
        
        env.close()
        print("  ✅ DreamerV3环境创建测试完成")
        
    except Exception as e:
        print(f"  ❌ DreamerV3环境创建测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_reward_generator():
    """测试奖励生成器"""
    print("\n🧪 测试奖励生成器...")
    
    try:
        sys.path.append('/home/zhuolifeng/dreamerv3-torch/rl_game')
        from reward_generator.reward_generator import RewardGenerator
        
        print("  🏗️  创建奖励生成器...")
        reward_gen = RewardGenerator()
        
        print("  🧮 测试特征向量...")
        # 生成一些测试特征向量
        features = [
            np.random.randn(512),  # 模拟CLIP特征
            np.random.randn(512),
            np.random.randn(512) + 2,  # 不同的特征
        ]
        
        rewards = []
        for i, feat in enumerate(features):
            reward = reward_gen.generate_reward(feat)
            rewards.append(reward)
            print(f"     特征 {i+1}: 奖励 = {reward:.4f}")
        
        print(f"  📊 奖励范围: {min(rewards):.4f} - {max(rewards):.4f}")
        print("  ✅ 奖励生成器测试完成")
        
    except Exception as e:
        print(f"  ❌ 奖励生成器测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("🚀 DreamerV3 视觉奖励功能测试")
    print("="*60)
    
    # 测试奖励生成器
    test_reward_generator()
    
    # 测试视觉奖励包装器
    test_visual_reward_wrapper()
    
    # 测试DreamerV3环境创建
    test_dreamer_env_creation()
    
    print("\n🎉 所有测试完成!")


if __name__ == "__main__":
    # 设置环境变量
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["DISPLAY"] = ":99"
    
    main()
