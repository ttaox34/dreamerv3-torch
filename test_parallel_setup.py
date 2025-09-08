#!/usr/bin/env python3
"""
测试并行retro环境设置的脚本
验证环境创建、动作空间兼容性等
"""
import sys
import os
import argparse
from pathlib import Path

# 添加dreamerv3路径
sys.path.append('/home/zhuolifeng/rl_game/dreamerv3-torch')

def test_single_env():
    """测试单个环境创建"""
    print("🧪 测试单个retro环境创建...")
    
    try:
        import envs.stable_retro as stable_retro
        
        # 测试创建单个环境
        env = stable_retro.StableRetro(
            game='SuperMarioBros-Nes',
            action_repeat=4,
            size=(64, 64),
            grayscale=False,
            seed=42,
        )
        
        print(f"✅ 环境创建成功")
        print(f"   动作空间: {env.action_space}")
        print(f"   观察空间: {env.observation_space}")
        
        # 测试reset和step
        obs = env.reset()
        print(f"   重置观察形状: {obs[0].shape if isinstance(obs, tuple) else obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"   步进后观察形状: {obs.shape}")
        print(f"   奖励: {reward}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 单个环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_envs():
    """测试多个不同游戏的环境创建"""
    print("\n🧪 测试多个retro环境创建...")
    
    games = ['SuperMarioBros-Nes', 'MegaMan-Nes', 'DonkeyKong-Nes']
    envs = []
    
    try:
        import envs.stable_retro as stable_retro
        
        for i, game in enumerate(games):
            print(f"   创建环境 {i+1}/{len(games)}: {game}")
            env = stable_retro.StableRetro(
                game=game,
                action_repeat=4,
                size=(64, 64),
                grayscale=False,
                seed=42 + i,
            )
            envs.append(env)
            print(f"     动作空间: {env.action_space}")
        
        # 检查动作空间兼容性
        action_spaces = [env.action_space for env in envs]
        all_same = all(
            space.n == action_spaces[0].n if hasattr(space, 'n') else space.shape == action_spaces[0].shape 
            for space in action_spaces
        )
        
        if all_same:
            print("✅ 所有游戏动作空间兼容")
        else:
            print("⚠️  游戏动作空间不同:")
            for game, space in zip(games, action_spaces):
                if hasattr(space, 'n'):
                    print(f"     {game}: {space.n} 离散动作")
                else:
                    print(f"     {game}: {space.shape} 连续动作")
        
        # 清理
        for env in envs:
            env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 多环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理已创建的环境
        for env in envs:
            try:
                env.close()
            except:
                pass
        
        return False

def test_config_loading():
    """测试配置加载"""
    print("\n🧪 测试配置加载...")
    
    try:
        import ruamel.yaml as yaml
        from pathlib import Path
        
        config_path = Path('/home/zhuolifeng/rl_game/dreamerv3-torch/configs.yaml')
        configs = yaml.safe_load(config_path.read_text())
        
        # 检查retro配置
        if 'retro' in configs:
            retro_config = configs['retro']
            print("✅ retro配置加载成功")
            print(f"   retro_parallel_games: {retro_config.get('retro_parallel_games', 'Not found')}")
            print(f"   retro_games_list: {retro_config.get('retro_games_list', 'Not found')}")
        
        # 检查retro_parallel配置
        if 'retro_parallel' in configs:
            parallel_config = configs['retro_parallel']
            print("✅ retro_parallel配置加载成功")
            print(f"   retro_parallel_games: {parallel_config.get('retro_parallel_games', 'Not found')}")
            print(f"   游戏列表: {parallel_config.get('retro_games_list', 'Not found')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dreamer_import():
    """测试Dreamer模块导入"""
    print("\n🧪 测试Dreamer模块导入...")
    
    try:
        # 测试主要模块导入
        import dreamer
        print("✅ dreamer模块导入成功")
        
        # 测试make_env函数
        if hasattr(dreamer, 'make_env'):
            print("✅ make_env函数存在")
        else:
            print("❌ make_env函数不存在")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Dreamer模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="测试并行retro环境设置")
    parser.add_argument("--skip-env", action="store_true", help="跳过环境创建测试（如果没有ROM）")
    args = parser.parse_args()
    
    print("🚀 开始测试并行retro环境设置...")
    print("=" * 60)
    
    # 测试配置加载
    config_ok = test_config_loading()
    
    # 测试Dreamer导入
    dreamer_ok = test_dreamer_import()
    
    # 测试环境创建（如果不跳过）
    single_env_ok = True
    multi_env_ok = True
    
    if not args.skip_env:
        single_env_ok = test_single_env()
        multi_env_ok = test_multiple_envs()
    else:
        print("\n⏭️ 跳过环境创建测试")
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    print(f"   配置加载: {'✅' if config_ok else '❌'}")
    print(f"   Dreamer导入: {'✅' if dreamer_ok else '❌'}")
    if not args.skip_env:
        print(f"   单环境创建: {'✅' if single_env_ok else '❌'}")
        print(f"   多环境创建: {'✅' if multi_env_ok else '❌'}")
    
    all_passed = config_ok and dreamer_ok and (args.skip_env or (single_env_ok and multi_env_ok))
    
    if all_passed:
        print("\n🎉 所有测试通过! 并行retro环境设置就绪")
        
        if not args.skip_env:
            print("\n📝 使用说明:")
            print("1. 单游戏训练（原始方式）:")
            print("   python dreamer.py --configs retro --task retro_SuperMarioBros-Nes")
            print()
            print("2. 多游戏并行训练:")
            print("   python dreamer.py --configs retro_parallel --task retro_multitask")
            print()
            print("3. 使用便捷脚本:")
            print("   python run_parallel_retro.py --games SuperMarioBros-Nes MegaMan-Nes")
    else:
        print("\n❌ 部分测试失败，请检查配置")
        sys.exit(1)

if __name__ == "__main__":
    main()
