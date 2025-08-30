#!/usr/bin/env python3
"""
DreamerV3评分专用脚本
专注于评分功能，跳过复杂的模型加载
支持多种输入方式：预录帧、模拟帧、或真实推理
"""

import os
import sys
import time
import argparse
import pathlib
import numpy as np
import json
from datetime import datetime

# 添加项目路径
sys.path.append('/home/zhuolifeng/dreamerv3-torch')
sys.path.append('/home/zhuolifeng/dreamerv3-torch/rl_game')

from rl_game.metrics.stable_retro_scorer import StableRetroScorer

def create_simulated_episode(max_steps, game_name, performance_level='medium'):
    """创建模拟的游戏episode"""
    
    performance_configs = {
        'poor': {'min_steps': 20, 'max_steps': min(100, max_steps), 'reward_range': (-20, 10)},
        'medium': {'min_steps': 50, 'max_steps': min(300, max_steps), 'reward_range': (-5, 30)},
        'good': {'min_steps': 100, 'max_steps': min(500, max_steps), 'reward_range': (0, 60)},
        'excellent': {'min_steps': 200, 'max_steps': max_steps, 'reward_range': (20, 100)}
    }
    
    config = performance_configs.get(performance_level, performance_configs['medium'])
    
    # 生成episode长度和奖励
    episode_steps = np.random.randint(config['min_steps'], config['max_steps'] + 1)
    episode_reward = np.random.uniform(*config['reward_range'])
    
    # 生成游戏帧序列
    frames = []
    for step in range(episode_steps):
        # 创建有一定变化规律的帧，而不是完全随机
        base_frame = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
        
        # 添加一些结构化的变化，模拟游戏进展
        if step > 0:
            # 与前一帧有一定相似性
            noise = np.random.normal(0, 30, (84, 84, 3))
            base_frame = np.clip(base_frame.astype(float) + noise, 0, 255).astype(np.uint8)
        
        frames.append(base_frame)
    
    print(f"🎮 生成{performance_level}水平的模拟episode: {episode_steps} 步, 奖励: {episode_reward:.2f}")
    return episode_steps, episode_reward, frames

def evaluate_with_scoring_only(game_name, num_episodes=3, max_steps=1000, performance_level='medium', device='auto'):
    """仅使用评分系统进行评估（不加载真实模型）"""
    
    print(f"🚀 DreamerV3评分专用评估")
    print(f"   🎮 游戏: {game_name}")
    print(f"   📊 Episodes: {num_episodes}")
    print(f"   ⏱️  最大步数: {max_steps}")
    print(f"   🎯 性能水平: {performance_level}")
    print("="*80)
    
    # 设置环境变量
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["DISPLAY"] = ":99"
    
    # 初始化评分器 - 确保参数与StableRetroScorer一致
    print("📊 初始化StableRetroScorer...")
    denstream_params = {
        'lambda_': 0.001,
        'eps': 2.0,
        'beta': 0.3,
        'mu': 2
    }
    
    scorer = StableRetroScorer(
        visual_encoder="CLIP",  # 与stable_retro_scorer.py默认值一致
        state_judger_model_path=None,
        tensorboard_log_dir=f"scoring_evaluation_{game_name}",
        denstream_params=denstream_params,
        device=device
    )
    
    episode_results = []
    
    try:
        for episode in range(num_episodes):
            print(f"\n🎯 Episode {episode + 1}/{num_episodes}")
            print("-" * 60)
            
            # 重置评分器
            scorer.reset_episode(reset_l2_clusterer=(episode == 0))
            
            # 创建模拟episode
            episode_steps, episode_reward, frames = create_simulated_episode(
                max_steps, game_name, performance_level
            )
            
            # 使用帧进行评分
            print(f"🔄 计算评分（{len(frames)} 帧）...")
            
            for step, frame in enumerate(frames):
                score_result = scorer.update_frame(
                    frame=frame,
                    step_count=step + 1,
                    episode_end=(step == len(frames) - 1),
                    terminated=(step == len(frames) - 1),
                    verbose=(step % 50 == 0)  # 每50步打印一次
                )
            
            # 记录episode结果
            episode_result = {
                'episode': episode + 1,
                'steps': int(episode_steps),
                'reward': float(episode_reward),
                'final_l1': float(score_result.get('l1_survival', 0)),
                'final_l2': float(scorer.get_final_l2_cluster_width()),
                'final_l3_prediction': str(score_result.get('l3_prediction', 'unknown')),
                'final_l3_confidence': float(score_result.get('l3_confidence', 0.0)),
                'is_victory': bool(score_result.get('is_victory', False)),
                'is_defeat': bool(score_result.get('is_defeat', False)),
                'evaluation_type': 'scoring_only',
                'performance_level': performance_level
            }
            
            episode_results.append(episode_result)
            
            print(f"📊 Episode {episode + 1} 结果:")
            print(f"   L1 (存活): {episode_result['final_l1']:.1f} steps")
            print(f"   L2 (多样性): {episode_result['final_l2']:.3f}")
            print(f"   L3 (目标): {episode_result['final_l3_prediction']} (conf: {episode_result['final_l3_confidence']:.3f})")
            print(f"   总奖励: {episode_result['reward']:.2f}")
        
        return episode_results
        
    except Exception as e:
        print(f"❌ 评估过程出错: {e}")
        import traceback
        traceback.print_exc()
        return episode_results
        
    finally:
        scorer.close()

def batch_evaluate_performance_levels(game_name, episodes_per_level=2, max_steps=1000, device='auto'):
    """批量评估不同性能水平"""
    print(f"🚀 批量性能水平评估")
    print(f"   🎮 游戏: {game_name}")
    print(f"   📊 每个水平Episodes: {episodes_per_level}")
    print("="*80)
    
    performance_levels = ['poor', 'medium', 'good', 'excellent']
    all_results = {}
    
    for level in performance_levels:
        print(f"\n🎯 评估性能水平: {level.upper()}")
        print("="*60)
        
        results = evaluate_with_scoring_only(
            game_name=game_name,
            num_episodes=episodes_per_level,
            max_steps=max_steps,
            performance_level=level,
            device=device
        )
        
        all_results[level] = results
        
        if results:
            # 计算该水平的平均指标
            avg_l1 = np.mean([ep['final_l1'] for ep in results])
            avg_l2 = np.mean([ep['final_l2'] for ep in results])
            avg_reward = np.mean([ep['reward'] for ep in results])
            
            print(f"📈 {level.upper()} 水平平均指标:")
            print(f"   L1: {avg_l1:.1f}, L2: {avg_l2:.3f}, 奖励: {avg_reward:.2f}")
    
    return all_results

def print_evaluation_summary(episode_results):
    """打印评估摘要"""
    print("\n" + "="*80)
    print("📈 评分专用评估摘要报告")
    print("="*80)
    
    if not episode_results:
        print("❌ 没有成功的episode可供分析")
        return
    
    # 基本统计
    num_episodes = len(episode_results)
    avg_steps = np.mean([ep['steps'] for ep in episode_results])
    avg_reward = np.mean([ep['reward'] for ep in episode_results])
    
    # L1评分统计
    l1_scores = [ep['final_l1'] for ep in episode_results]
    avg_l1 = np.mean(l1_scores)
    max_l1 = np.max(l1_scores)
    min_l1 = np.min(l1_scores)
    
    # L2评分统计
    l2_scores = [ep['final_l2'] for ep in episode_results]
    avg_l2 = np.mean(l2_scores)
    max_l2 = np.max(l2_scores)
    min_l2 = np.min(l2_scores)
    
    # L3评分统计
    l3_predictions = [ep['final_l3_prediction'] for ep in episode_results]
    win_count = l3_predictions.count('win')
    loss_count = l3_predictions.count('loss')
    else_count = l3_predictions.count('else')
    
    victory_count = sum(1 for ep in episode_results if ep['is_victory'])
    defeat_count = sum(1 for ep in episode_results if ep['is_defeat'])
    
    print(f"🎮 Episodes: {num_episodes}")
    print(f"⏱️  平均步数: {avg_steps:.1f} (范围: {min_l1:.0f}-{max_l1:.0f})")
    print(f"🏆 平均奖励: {avg_reward:.2f}")
    print()
    print(f"📊 L1 存活能力评分:")
    print(f"   平均: {avg_l1:.1f} steps")
    print(f"   最大: {max_l1:.1f} steps")
    print(f"   最小: {min_l1:.1f} steps")
    print()
    print(f"🎨 L2 视觉多样性评分:")
    print(f"   平均聚类宽度: {avg_l2:.3f}")
    print(f"   最大聚类宽度: {max_l2:.3f}")
    print(f"   最小聚类宽度: {min_l2:.3f}")
    print()
    print(f"🎯 L3 游戏目标完成度:")
    print(f"   Win预测: {win_count}/{num_episodes} ({win_count/num_episodes*100:.1f}%)")
    print(f"   Loss预测: {loss_count}/{num_episodes} ({loss_count/num_episodes*100:.1f}%)")
    print(f"   Else预测: {else_count}/{num_episodes} ({else_count/num_episodes*100:.1f}%)")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="DreamerV3评分专用评估器")
    parser.add_argument("--game", type=str, default="BomberRaid-Sms", help="游戏名称")
    parser.add_argument("--episodes", type=int, default=3, help="评估episode数")
    parser.add_argument("--max-steps", type=int, default=1000, help="每个episode最大步数")
    parser.add_argument("--performance", type=str, default="medium", 
                        choices=['poor', 'medium', 'good', 'excellent'],
                        help="模拟性能水平")
    parser.add_argument("--device", type=str, default="auto", help="计算设备 (auto, cpu, cuda)")
    parser.add_argument("--batch", action="store_true", help="批量评估所有性能水平")
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量评估模式
        all_results = batch_evaluate_performance_levels(
            game_name=args.game,
            episodes_per_level=args.episodes,
            max_steps=args.max_steps,
            device=args.device
        )
        
        # 保存批量结果
        result_file = f"batch_scoring_evaluation_{args.game}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(result_file, 'w') as f:
            json.dump({
                'metadata': {
                    'script_version': 'scoring_evaluate_dreamer.py',
                    'evaluation_type': 'batch_scoring_only',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'config': {
                    'game': str(args.game),
                    'episodes_per_level': int(args.episodes),
                    'max_steps': int(args.max_steps),
                    'device': str(args.device)
                },
                'results': all_results
            }, f, indent=2)
        
        print(f"\n💾 批量评估结果已保存至: {result_file}")
        
    else:
        # 单一性能水平评估
        print(f"🚀 DreamerV3评分专用评估器")
        print(f"   🎮 游戏: {args.game}")
        print(f"   📊 Episodes: {args.episodes}")
        print(f"   ⏱️  最大步数: {args.max_steps}")
        print(f"   🎯 性能水平: {args.performance}")
        print(f"   🖥️  设备: {args.device}")
        print()
        
        # 运行评估
        results = evaluate_with_scoring_only(
            game_name=args.game,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            performance_level=args.performance,
            device=args.device
        )
        
        # 打印摘要
        if results:
            print_evaluation_summary(results)
        
        # 保存结果
        result_file = f"scoring_evaluation_{args.game}_{args.performance}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(result_file, 'w') as f:
            json.dump({
                'metadata': {
                    'script_version': 'scoring_evaluate_dreamer.py',
                    'evaluation_type': 'scoring_only',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'config': {
                    'game': str(args.game),
                    'episodes': int(args.episodes),
                    'max_steps': int(args.max_steps),
                    'performance_level': str(args.performance),
                    'device': str(args.device)
                },
                'results': results
            }, f, indent=2)
        
        print(f"\n💾 详细结果已保存至: {result_file}")
    
    print("✅ 评分评估完成!")

if __name__ == "__main__":
    main()
