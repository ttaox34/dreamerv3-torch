#!/usr/bin/env python3
"""测试多游戏动作空间获取"""
import sys
sys.path.append('.')

from envs.multigame_retro import MultiGameRetroEnv

def test_multigame_action_spaces():
    """测试多游戏环境的动作空间"""
    games = [
        'SuperMarioBros-Nes',
        'BalloonFight-Nes', 
        'MegaMan-Nes',
        'Battletoads-Nes'
    ]
    
    print("Testing MultiGameRetroEnv action space detection...")
    
    env = MultiGameRetroEnv(
        games=games,
        action_repeat=1,
        size=(64, 64),
        grayscale=False
    )
    
    print(f"\nMulti-game action spaces: {env.action_spaces}")
    print(f"Num actions list: {env.num_actions_list}")
    print(f"Max actions per game: {env.max_num_actions}")
    
    # Test each game
    for i, game in enumerate(games):
        print(f"\n=== {game} ===")
        print(f"Action space: {env.action_spaces[i]}")
        print(f"Num actions: {env.num_actions_list[i]}")
        
        # Test reset and step
        obs = env.reset(game_idx=i)
        print(f"Observation keys: {obs.keys()}")
        print(f"Game ID: {obs['game_id']}")
        
        # Test action conversion
        action = 42  # Test action
        converted = env.envs[i]._convert_action(action) if env.envs[i] is not None else "N/A"
        print(f"Test action {action} -> {converted}")
    
    env.close()

if __name__ == "__main__":
    test_multigame_action_spaces()