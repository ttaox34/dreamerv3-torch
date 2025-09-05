#!/usr/bin/env python3
"""
多游戏retro训练示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    print("🎮 Multi-Game Retro Training Example")
    print("=" * 50)
    
    # 示例命令
    examples = [
        {
            "name": "双游戏训练 (Sonic + Airstriker)",
            "command": "python3 dreamer.py --configs retro_multigame --task retro_multigame --retro_games 'SonicTheHedgehog-Genesis,Airstriker-Genesis' --logdir ./logdir/multigame_sonic_airstriker"
        },
        {
            "name": "三游戏训练",
            "command": "python3 dreamer.py --configs retro_multigame --task retro_multigame --retro_games 'SonicTheHedgehog-Genesis,Airstriker-Genesis,AlteredBeast-Genesis' --logdir ./logdir/multigame_three_games"
        },
        {
            "name": "自定义游戏列表",
            "command": "python3 dreamer.py --configs retro_multigame --task retro_multigame --retro_games 'YourGame1-System,YourGame2-System' --logdir ./logdir/multigame_custom"
        }
    ]
    
    print("📋 Available training examples:")
    print()
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Command: {example['command']}")
        print()
    
    print("🔧 Configuration Options:")
    print("  --retro_games: Comma-separated list of game names")
    print("  --configs: Use 'retro_multigame' configuration")
    print("  --task: Should be 'retro_multigame'")
    print("  --logdir: Logging directory for the experiment")
    print()
    
    print("📝 Notes:")
    print("  - Game names must match the format 'GameName-System'")
    print("  - Use retro.data.list_games() to see available games")
    print("  - Each game will have its own action head in the neural network")
    print("  - The environment manager handles switching between games")
    print()
    
    print("🧪 Testing:")
    print("  Run the test script to verify functionality:")
    print("  python3 test_multigame.py")
    print()
    
    print("📊 Monitoring:")
    print("  Use TensorBoard to monitor training:")
    print("  tensorboard --logdir ./logdir/")
    print()
    
    # 可用游戏列表
    try:
        import retro
        print("🎮 Available retro games:")
        games = retro.data.list_games()
        print(f"  Total games available: {len(games)}")
        
        # 显示前10个游戏作为示例
        sample_games = games[:10]
        print("  Sample games:")
        for game in sample_games:
            print(f"    - {game}")
        
        if len(games) > 10:
            print(f"    ... and {len(games) - 10} more")
        
    except ImportError:
        print("⚠️  retro not installed. Install with: pip install stable-retro")
    except Exception as e:
        print(f"⚠️  Could not list games: {e}")
    
    print()
    print("🚀 Ready to train! Choose an example above and run the command.")

if __name__ == "__main__":
    main()