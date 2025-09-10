#!/usr/bin/env python3
"""
启动并行retro游戏训练的便捷脚本
使用预配置的多游戏设置来运行Dreamer训练
"""
import argparse
import subprocess
import sys
from pathlib import Path

def format_game_list(games):
    """将游戏列表格式化为命令行参数"""
    return ",".join(games)

def main():
    parser = argparse.ArgumentParser(description="启动并行retro游戏训练")
    parser.add_argument("--games", nargs="+", 
                       default=['SuperMarioBros-Nes', 'SuperMarioWorld-Snes', 'MegaMan-Nes'],
                       help="要训练的游戏列表")
    parser.add_argument("--steps", type=str, default="2e6",
                       help="训练步数 (default: 2e6)")
    parser.add_argument("--logdir", default="./logdir_parallel_retro",
                       help="日志目录")
    parser.add_argument("--device", default="cuda:0",
                       help="设备 (default: cuda:0)")
    parser.add_argument("--visual-reward", action="store_true",
                       help="启用视觉奖励")
    parser.add_argument("--visual-weight", type=float, default=0.05,
                       help="视觉奖励权重 (default: 0.05)")
    parser.add_argument("--config", default="retro_parallel",
                       help="配置名称 (default: retro_parallel)")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="启用进程并行 (default: True)")
    args = parser.parse_args()
    
    # 构建命令行参数
    cmd = [
        sys.executable, "dreamer.py",
        f"--configs={args.config}",
        f"--task=retro_multitask",  # 使用占位符任务名
        f"--logdir={args.logdir}",
        f"--device={args.device}",
        f"--steps={args.steps}",
        "--retro_parallel_games=True",
        f"--retro_games_list={format_game_list(args.games)}",
        f"--visual_reward_weight={args.visual_weight}",
    ]
    
    # 添加并行设置
    if args.parallel:
        cmd.append("--parallel=True")
    
    # 添加视觉奖励设置
    if args.visual_reward:
        cmd.extend([
            "--visual_reward=True",
            "--visual_encoder=ResNet",
        ])
    else:
        cmd.append("--visual_reward=False")
    
    print("🚀 启动并行retro游戏训练...")
    print(f"📱 游戏列表: {', '.join(args.games)}")
    print(f"🎯 训练步数: {args.steps}")
    print(f"📁 日志目录: {args.logdir}")
    print(f"💾 设备: {args.device}")
    print(f"🔄 进程并行: {'启用' if args.parallel else '禁用'}")
    print(f"🎨 视觉奖励: {'启用' if args.visual_reward else '禁用'}")
    if args.visual_reward:
        print(f"   权重: {args.visual_weight}")
    print()
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    # 执行训练
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        print("\n🔧 故障排除建议:")
        print("1. 检查stable-retro是否已安装: pip install stable-retro")
        print("2. 确保ROM文件在正确位置")
        print("3. 检查CUDA设备是否可用 (如果使用GPU)")
        print("4. 运行测试脚本: python test_parallel_setup.py")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        sys.exit(0)

if __name__ == "__main__":
    main()
