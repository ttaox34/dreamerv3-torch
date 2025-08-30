#!/usr/bin/env python3
"""
DreamerV3模型性能评估脚本（集成StableRetroScorer）
使用StableRetroScorer对DreamerV3训练的智能体进行三级评分
"""

import os
import sys
import time
import argparse
import pathlib
import numpy as np
import torch
from collections import defaultdict
import ruamel.yaml as yaml

# 添加项目路径
sys.path.append('/home/zhuolifeng/dreamerv3-torch')
sys.path.append('/home/zhuolifeng/dreamerv3-torch/rl_game')

# DreamerV3相关导入
import tools
import models
import dreamer
from rl_game.metrics.stable_retro_scorer import StableRetroScorer

def load_dreamer_agent(checkpoint_path, game_name, device):
    """加载DreamerV3智能体"""
    print(f"🔄 加载DreamerV3模型: {checkpoint_path}")
    
    # 从配置文件加载完整配置
    configs = yaml.safe_load(
        (pathlib.Path(__file__).parent / "configs.yaml").read_text()
    )
    
    # 合并配置
    config_dict = {}
    config_dict.update(configs['defaults'])
    config_dict.update(configs['retro'])
    
    # 更新特定参数
    config_dict['task'] = f"retro_{game_name}"
    config_dict['device'] = device
    config_dict['visual_reward'] = False  # 评估时禁用视觉奖励
    config_dict['visual_reward_weight'] = 0.0
    
    # 转换为对象
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    config = Config(config_dict)
    
    # 创建环境来获取正确的观察和动作空间
    temp_env = dreamer.make_env(config, "eval", 0)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    
    # 添加缺少的配置
    if hasattr(act_space, 'n'):
        config.num_actions = act_space.n
    else:
        config.num_actions = act_space.shape[0] if hasattr(act_space, 'shape') else 1
    
    temp_env.close()
    
    # 创建虚拟数据集和日志器
    class DummyDataset:
        pass
    
    class DummyLogger:
        def __init__(self):
            self.step = 0
        def video(self, name, video):
            pass
        def write(self, **kwargs):
            pass
    
    logger = DummyLogger()
    dataset = DummyDataset()
    
    # 初始化Dreamer智能体
    agent = dreamer.Dreamer(obs_space, act_space, config, logger, dataset)
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.eval()
    
    print("✅ DreamerV3模型加载完成")
    return agent, config

def evaluate_agent_with_scoring(
    agent, 
    env, 
    scorer, 
    num_episodes=5, 
    max_steps_per_episode=1000,
    config=None
):
    """使用StableRetroScorer评估智能体性能"""
    
    print(f"🎮 开始评估智能体 (Episodes: {num_episodes}, Max Steps: {max_steps_per_episode})")
    print("="*80)
    
    episode_results = []
    
    for episode in range(num_episodes):
        print(f"\n🎯 Episode {episode + 1}/{num_episodes}")
        
        # 重置环境和评分器
        obs = env.reset()
        scorer.reset_episode(reset_l2_clusterer=(episode == 0))  # 只在第一个episode重置聚类器
        
        # 初始化智能体状态
        agent_state = None
        action = torch.zeros(1, env.action_space.n, dtype=torch.float32, device=config.device)
        
        episode_reward = 0.0
        episode_steps = 0
        episode_scores = []
        
        for step in range(max_steps_per_episode):
            # 准备观察数据
            obs_tensor = {}
            for key, value in obs.items():
                if key == 'image':
                    # 转换图像数据
                    image = torch.FloatTensor(value).unsqueeze(0).to(config.device)
                    if image.max() > 1.0:
                        image = image / 255.0
                    obs_tensor[key] = image
                else:
                    obs_tensor[key] = torch.FloatTensor([value]).to(config.device)
            
            # 智能体选择动作
            with torch.no_grad():
                action, agent_state = agent.act(obs_tensor, agent_state, mode='eval')
                action_idx = torch.argmax(action, dim=-1).item()
            
            # 环境步进
            next_obs, reward, done, info = env.step(action_idx)
            episode_reward += reward
            episode_steps += 1
            
            # 获取原始帧进行评分
            if hasattr(env, 'get_raw_frame'):
                raw_frame = env.get_raw_frame()
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'get_raw_frame'):
                raw_frame = env.unwrapped.get_raw_frame()
            elif isinstance(obs, dict) and 'image' in obs:
                raw_frame = obs['image']
            else:
                raw_frame = obs
            
            # 评分器更新
            score_result = scorer.update_frame(
                frame=raw_frame,
                step_count=step + 1,
                episode_end=done,
                terminated=done,
                verbose=(step % 50 == 0)  # 每50步打印一次详细信息
            )
            
            episode_scores.append(score_result)
            
            obs = next_obs
            
            # 检查episode结束条件
            if done:
                print(f"   Episode结束 - Steps: {episode_steps}, Reward: {episode_reward:.2f}")
                break
            
            # 失败检测（基于评分器判断）
            if score_result.get('is_defeat', False):
                print(f"   检测到失败状态 - 提前结束episode")
                break
        
        # 记录episode结果
        episode_result = {
            'episode': episode + 1,
            'steps': episode_steps,
            'reward': episode_reward,
            'scores': episode_scores,
            'final_l1': score_result.get('l1_survival', 0),
            'final_l2': scorer.get_final_l2_cluster_width(),
            'final_l3_prediction': score_result.get('l3_prediction', 'unknown'),
            'final_l3_confidence': score_result.get('l3_confidence', 0.0),
            'is_victory': score_result.get('is_victory', False),
            'is_defeat': score_result.get('is_defeat', False)
        }
        
        episode_results.append(episode_result)
        
        print(f"   📊 Episode {episode + 1} 结果:")
        print(f"      L1 (存活): {episode_result['final_l1']:.1f} steps")
        print(f"      L2 (多样性): {episode_result['final_l2']:.3f}")
        print(f"      L3 (目标): {episode_result['final_l3_prediction']} (conf: {episode_result['final_l3_confidence']:.3f})")
        print(f"      总奖励: {episode_result['reward']:.2f}")
    
    return episode_results

def print_evaluation_summary(episode_results):
    """打印评估摘要"""
    print("\n" + "="*80)
    print("📈 评估摘要报告")
    print("="*80)
    
    # 基本统计
    num_episodes = len(episode_results)
    avg_steps = np.mean([ep['steps'] for ep in episode_results])
    avg_reward = np.mean([ep['reward'] for ep in episode_results])
    
    # L1评分统计
    l1_scores = [ep['final_l1'] for ep in episode_results]
    avg_l1 = np.mean(l1_scores)
    max_l1 = np.max(l1_scores)
    
    # L2评分统计
    l2_scores = [ep['final_l2'] for ep in episode_results]
    avg_l2 = np.mean(l2_scores)
    max_l2 = np.max(l2_scores)
    
    # L3评分统计
    l3_predictions = [ep['final_l3_prediction'] for ep in episode_results]
    win_count = l3_predictions.count('win')
    loss_count = l3_predictions.count('loss')
    else_count = l3_predictions.count('else')
    
    victory_count = sum(1 for ep in episode_results if ep['is_victory'])
    defeat_count = sum(1 for ep in episode_results if ep['is_defeat'])
    
    print(f"🎮 总Episodes: {num_episodes}")
    print(f"⏱️  平均步数: {avg_steps:.1f}")
    print(f"🏆 平均奖励: {avg_reward:.2f}")
    print()
    print(f"📊 L1 存活能力评分:")
    print(f"   平均: {avg_l1:.1f} steps")
    print(f"   最大: {max_l1:.1f} steps")
    print()
    print(f"🎨 L2 视觉多样性评分:")
    print(f"   平均聚类宽度: {avg_l2:.3f}")
    print(f"   最大聚类宽度: {max_l2:.3f}")
    print()
    print(f"🎯 L3 游戏目标完成度:")
    print(f"   Win预测: {win_count}/{num_episodes} ({win_count/num_episodes*100:.1f}%)")
    print(f"   Loss预测: {loss_count}/{num_episodes} ({loss_count/num_episodes*100:.1f}%)")
    print(f"   Else预测: {else_count}/{num_episodes} ({else_count/num_episodes*100:.1f}%)")
    print()
    print(f"🏅 真实结果:")
    print(f"   胜利: {victory_count}/{num_episodes} ({victory_count/num_episodes*100:.1f}%)")
    print(f"   失败: {defeat_count}/{num_episodes} ({defeat_count/num_episodes*100:.1f}%)")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="DreamerV3模型性能评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--game", type=str, default="BomberRaid-Sms", help="游戏名称")
    parser.add_argument("--episodes", type=int, default=5, help="评估episode数")
    parser.add_argument("--max-steps", type=int, default=1000, help="每个episode最大步数")
    parser.add_argument("--visual-encoder", type=str, default="ResNet", 
                       choices=["CLIP", "ResNet", "ViT", "DINO"], help="视觉编码器类型")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"🚀 DreamerV3模型性能评估")
    print(f"   📁 Checkpoint: {args.checkpoint}")
    print(f"   🎮 游戏: {args.game}")
    print(f"   📊 Episodes: {args.episodes}")
    print(f"   🖥️  设备: {device}")
    print()
    
    # 设置环境变量
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["DISPLAY"] = ":99"
    
    try:
        # 设置环境变量
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["DISPLAY"] = ":99"
        
        # 创建评分器
        print("📊 初始化StableRetroScorer...")
        scorer = StableRetroScorer(
            visual_encoder=args.visual_encoder,
            tensorboard_log_dir=f"evaluation_scores_{args.game}",
            device=device
        )
        
        # 加载模型（包含配置）
        agent, config = load_dreamer_agent(args.checkpoint, args.game, device)
        
        # 创建环境
        print("🌐 创建环境...")
        env = dreamer.make_env(config, "eval", 0)
        
        # 开始评估
        episode_results = evaluate_agent_with_scoring(
            agent=agent,
            env=env,
            scorer=scorer,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            config=config
        )
        
        # 打印摘要
        print_evaluation_summary(episode_results)
        
        # 保存详细结果
        import json
        result_file = f"evaluation_results_{args.game}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 清理结果中的复杂对象
        cleaned_results = []
        for ep_result in episode_results:
            cleaned_ep = {
                'episode': ep_result['episode'],
                'steps': convert_numpy(ep_result['steps']),
                'reward': convert_numpy(ep_result['reward']),
                'final_l1': convert_numpy(ep_result['final_l1']),
                'final_l2': convert_numpy(ep_result['final_l2']),
                'final_l3_prediction': ep_result['final_l3_prediction'],
                'final_l3_confidence': convert_numpy(ep_result['final_l3_confidence']),
                'is_victory': ep_result['is_victory'],
                'is_defeat': ep_result['is_defeat']
            }
            cleaned_results.append(cleaned_ep)
        
        with open(result_file, 'w') as f:
            json.dump({
                'config': {
                    'checkpoint': args.checkpoint,
                    'game': args.game,
                    'episodes': args.episodes,
                    'max_steps': args.max_steps,
                    'visual_encoder': args.visual_encoder
                },
                'results': cleaned_results
            }, f, indent=2)
        
        print(f"\n💾 详细结果已保存至: {result_file}")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if 'env' in locals():
            env.close()
        if 'scorer' in locals():
            scorer.close()

if __name__ == "__main__":
    main()
