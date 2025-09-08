#!/usr/bin/env python3
"""
Dreamer并行retro训练示例脚本

这个脚本展示了如何使用修改后的Dreamer进行多游戏并行训练
"""
import subprocess
import sys
import time
from pathlib import Path

def run_example(example_type="demo"):
    """运行示例训练"""
    
    if example_type == "demo":
        print("🎮 运行demo示例：3个经典游戏并行训练")
        cmd = [
            sys.executable, "run_parallel_retro.py",
            "--games", "SuperMarioBros-Nes", "MegaMan-Nes", "DonkeyKong-Nes",
            "--steps", "1e5",  # 短时间用于演示
            "--logdir", "./demo_logs",
            "--config", "retro_parallel"
        ]
        
    elif example_type == "large":
        print("🚀 运行大规模示例：5个游戏长时间训练")
        cmd = [
            sys.executable, "run_parallel_retro.py", 
            "--games", 
            "SuperMarioBros-Nes", "SuperMarioWorld-Snes", 
            "MegaMan-Nes", "MegaMan2-Nes", "DonkeyKong-Nes",
            "--steps", "2e6",
            "--logdir", "./large_scale_logs",
            "--visual-reward",
            "--visual-weight", "0.05"
        ]
        
    elif example_type == "single":
        print("🎯 运行单游戏示例（对比基线）")
        cmd = [
            sys.executable, "dreamer.py",
            "--configs", "retro",
            "--task", "retro_SuperMarioBros-Nes",
            "--steps", "1e5",
            "--logdir", "./single_game_logs"
        ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, cwd="/home/zhuolifeng/rl_game/dreamerv3-torch", check=True)
        end_time = time.time()
        
        print(f"\n✅ 训练完成! 耗时: {end_time - start_time:.2f}秒")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  训练被用户中断")
        return False

def main():
    print("🎮 Dreamer并行retro训练示例")
    print("=" * 60)
    print("选择要运行的示例:")
    print("1. demo   - 3个游戏短时间并行训练 (推荐用于测试)")
    print("2. large  - 5个游戏长时间训练 (需要较好硬件)")
    print("3. single - 单游戏训练 (对比基线)")
    print("4. exit   - 退出")
    
    while True:
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            success = run_example("demo")
            break
        elif choice == "2":
            success = run_example("large")
            break
        elif choice == "3":
            success = run_example("single")
            break
        elif choice == "4":
            print("👋 再见!")
            sys.exit(0)
        else:
            print("❌ 无效选择，请输入1-4")
    
    if success:
        print("\n📊 查看训练结果:")
        print("   - TensorBoard日志: tensorboard --logdir <logdir>")
        print("   - 检查保存的模型: ls <logdir>/*.pt")
        print("\n📚 更多用法请参考: README_PARALLEL_RETRO.md")
    else:
        print("\n🔧 如果遇到问题，请:")
        print("   1. 检查是否安装了stable-retro: pip install stable-retro")
        print("   2. 确保ROM文件在正确位置")
        print("   3. 运行测试脚本: python test_parallel_setup.py")

if __name__ == "__main__":
    main()
