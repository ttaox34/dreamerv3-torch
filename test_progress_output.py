#!/usr/bin/env python3
"""测试episode和step输出功能"""
import sys
sys.path.append('.')

from envs.multigame_retro import MultiGameRetroEnv

def test_progress_output():
    """测试进度输出功能"""
    games = ['SuperMarioBros-Nes', 'BalloonFight-Nes']
    
    print("Testing progress output functionality...")
    
    env = MultiGameRetroEnv(
        games=games,
        action_repeat=1,
        size=(64, 64),
        grayscale=False
    )
    
    # 设置输出频率为50步以便更频繁地测试时间戳
    env.set_output_frequency(50)
    
    # 测试第一个游戏
    print("\n=== Testing first game ===")
    obs = env.reset(game_idx=0)
    
    # 模拟300步
    for i in range(300):
        action = i % 512  # 循环使用不同的动作
        obs, reward, done, info = env.step(action)
        
        if done:
            print(f"Episode finished at step {env.current_step}")
            obs = env.reset(game_idx=0)
    
    # 测试游戏切换
    print("\n=== Testing game switch ===")
    obs = env.reset(game_idx=1)
    
    # 模拟200步
    for i in range(200):
        action = i % 512
        obs, reward, done, info = env.step(action)
        
        if done:
            print(f"Episode finished at step {env.current_step}")
            obs = env.reset(game_idx=1)
    
    print(f"\nFinal stats:")
    print(f"Total episodes: {env.episode_count}")
    print(f"Total steps: {env.total_steps}")
    
    env.close()

if __name__ == "__main__":
    test_progress_output()