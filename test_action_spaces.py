#!/usr/bin/env python3
"""测试动作空间获取"""
import sys
sys.path.append('.')

from envs.stable_retro import StableRetro

def test_action_spaces():
    """测试不同游戏的动作空间"""
    games = [
        'SuperMarioBros-Nes',
        'BalloonFight-Nes', 
        'MegaMan-Nes',
        'Battletoads-Nes'
    ]
    
    for game in games:
        try:
            print(f"\n=== {game} ===")
            env = StableRetro(
                game=game,
                action_repeat=1,
                size=(64, 64),
                grayscale=False
            )
            
            print(f"Action space: {env.action_space}")
            print(f"Action space type: {type(env.action_space)}")
            
            if hasattr(env.action_space, 'n'):
                print(f"MultiBinary space with {env.action_space.n} buttons")
                print(f"Possible actions: {2 ** env.action_space.n}")
            else:
                print(f"Discrete space with {env.action_space.n} actions")
            
            print(f"_n_buttons: {env._n_buttons}")
            print(f"_n_actions: {env._n_actions}")
            print(f"_is_multibinary: {env._is_multibinary}")
            
            env.close()
            
        except Exception as e:
            print(f"Error with {game}: {e}")

if __name__ == "__main__":
    test_action_spaces()