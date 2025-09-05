#!/usr/bin/env python3
"""
多GPU训练示例脚本

演示如何使用多GPU训练功能
"""

import torch
import argparse
import sys
import pathlib

# 添加项目根目录到路径
sys.path.append(str(pathlib.Path(__file__).parent))


def run_multigpu_dmc_example():
    """运行多GPU DMC训练示例"""
    print("🎯 多GPU DMC 训练示例")
    print("=" * 50)
    
    # 检查GPU数量
    if torch.cuda.device_count() < 2:
        print("⚠️  需要至少2个GPU才能运行多GPU训练")
        print("💡 您可以使用单GPU配置: python dreamer.py --configs dmc_vision")
        return
    
    print(f"✅ 发现 {torch.cuda.device_count()} 个GPU")
    print("🚀 启动多GPU DMC 训练...")
    
    # 构建命令
    cmd = [
        "python", "dreamer.py",
        "--configs", "multigpu_dmc_vision",
        "--task", "dmc_walker_walk",
        "--logdir", "./logs/multigpu_dmc_example",
        "--steps", "100000"
    ]
    
    print("📋 执行命令:")
    print("   " + " ".join(cmd))
    print()
    
    # 注意：这里只是演示，实际运行时取消注释
    # import subprocess
    # subprocess.run(cmd)


def run_multigpu_retro_example():
    """运行多GPU Retro训练示例"""
    print("\n🎮 多GPU Retro 多游戏训练示例")
    print("=" * 50)
    
    # 检查GPU数量
    if torch.cuda.device_count() < 2:
        print("⚠️  需要至少2个GPU才能运行多GPU训练")
        print("💡 您可以使用单GPU配置: python dreamer.py --configs retro_multigame")
        return
    
    print(f"✅ 发现 {torch.cuda.device_count()} 个GPU")
    print("🚀 启动多GPU Retro 多游戏训练...")
    
    # 构建命令
    cmd = [
        "python", "dreamer.py",
        "--configs", "multigpu_retro",
        "--task", "retro_multigame",
        "--logdir", "./logs/multigpu_retro_example",
        "--steps", "100000"
    ]
    
    print("📋 执行命令:")
    print("   " + " ".join(cmd))
    print()
    
    # 注意：这里只是演示，实际运行时取消注释
    # import subprocess
    # subprocess.run(cmd)


def run_performance_comparison():
    """性能对比示例"""
    print("\n⚡ 性能对比示例")
    print("=" * 50)
    
    print("单GPU vs 多GPU 性能对比:")
    print()
    
    # 单GPU配置
    print("🔸 单GPU配置:")
    print("   python dreamer.py --configs dmc_vision --steps 100000")
    print("   - 使用1个GPU")
    print("   - 环境数量: 4")
    print("   - 批次大小: 16")
    print("   - 预期训练时间: ~2小时")
    print()
    
    # 多GPU配置
    print("🔸 多GPU配置:")
    print("   python dreamer.py --configs multigpu_dmc_vision --steps 100000")
    print(f"   - 使用{torch.cuda.device_count()}个GPU")
    print("   - 环境数量: 8 (每个GPU 2个)")
    print("   - 批次大小: 32")
    print("   - 预期训练时间: ~1小时")
    print()
    
    print("💡 性能提升:")
    print("   - 数据并行: 理论加速比 ≈ GPU数量")
    print("   - 环境并行: 更多环境并行收集数据")
    print("   - 更大批次: 更稳定的梯度更新")
    print("   - 总体加速: 2-4倍 (取决于GPU数量)")


def show_configuration_details():
    """显示配置详情"""
    print("\n⚙️ 配置详情")
    print("=" * 50)
    
    print("多GPU配置文件 (configs.yaml):")
    print()
    
    print("1. multigpu_dmc_vision:")
    print("   - multi_gpu: True")
    print("   - envs: 8 (总环境数)")
    print("   - batch_size: 32")
    print("   - multi_gpu_backend: 'nccl'")
    print("   - 其他参数与标准dmc_vision相同")
    print()
    
    print("2. multigpu_retro:")
    print("   - multi_gpu: True")
    print("   - envs: 4 (总环境数)")
    print("   - batch_size: 32")
    print("   - retro_games: 'BalloonFight-Nes,BomberRaid-Sms'")
    print("   - game_switch_strategy: 'random'")
    print("   - 支持视觉奖励和游戏切换")
    print()


def show_monitoring_commands():
    """显示监控命令"""
    print("\n📊 监控命令")
    print("=" * 50)
    
    print("训练过程中可以使用以下命令监控:")
    print()
    
    print("1. GPU使用情况:")
    print("   nvidia-smi")
    print("   watch -n 1 nvidia-smi")
    print()
    
    print("2. 训练进度:")
    print("   tensorboard --logdir ./logs/")
    print("   tail -f ./logs/*/metrics.jsonl")
    print()
    
    print("3. 系统资源:")
    print("   htop")
    print("   free -h")
    print("   df -h")
    print()


def show_troubleshooting():
    """显示故障排除"""
    print("\n🔧 故障排除")
    print("=" * 50)
    
    print("常见问题及解决方案:")
    print()
    
    print("1. CUDA out of memory:")
    print("   - 减少batch_size")
    print("   - 减少envs数量")
    print("   - 使用precision: 16 (混合精度)")
    print()
    
    print("2. 分布式训练失败:")
    print("   - 检查NCCL安装: python -c 'import torch.distributed as dist; print(dist.is_nccl_available())'")
    print("   - 检查GPU间通信: nvidia-smi topo -m")
    print("   - 确保所有GPU空闲")
    print()
    
    print("3. 环境创建失败:")
    print("   - 检查环境依赖")
    print("   - 减少并行环境数量")
    print("   - 使用parallel: False")
    print()
    
    print("4. 性能不佳:")
    print("   - 增加envs数量")
    print("   - 调整batch_size")
    print("   - 使用更大的imag_horizon")
    print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多GPU训练示例")
    parser.add_argument("--example", type=str, choices=['dmc', 'retro', 'all'], 
                       default='all', help="运行特定示例")
    args = parser.parse_args()
    
    print("🚀 DreamerV3 多GPU训练示例")
    print("=" * 60)
    print()
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法运行多GPU训练")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"🎯 系统信息:")
    print(f"   GPU数量: {num_gpus}")
    print(f"   CUDA版本: {torch.version.cuda}")
    print(f"   PyTorch版本: {torch.__version__}")
    print()
    
    # 运行示例
    if args.example in ['dmc', 'all']:
        run_multigpu_dmc_example()
    
    if args.example in ['retro', 'all']:
        run_multigpu_retro_example()
    
    if args.example == 'all':
        run_performance_comparison()
        show_configuration_details()
        show_monitoring_commands()
        show_troubleshooting()
    
    print("\n🎉 示例演示完成！")
    print()
    print("💡 下一步:")
    print("   1. 选择合适的配置文件")
    print("   2. 调整参数以适应您的硬件")
    print("   3. 运行训练命令")
    print("   4. 监控训练进度")
    print("   5. 分析训练结果")


if __name__ == "__main__":
    main()