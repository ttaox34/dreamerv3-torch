#!/usr/bin/env python3
"""
真实DreamerV3模型评估脚本
完全独立的模型加载和推理版本

功能：
- 真实加载DreamerV3模型并进行推理
- 使用StableRetroScorer进行三级评分
- 参数与stable_retro_scorer.py完全一致，确保横向比较有效性
"""

import os
import sys
import time
import argparse
import pathlib
import numpy as np
import torch
import json
from collections import defaultdict

# 添加项目路径
sys.path.append('/home/zhuolifeng/dreamerv3-torch')
sys.path.append('/home/zhuolifeng/dreamerv3-torch/rl_game')

# DreamerV3相关导入
import tools
import dreamer
import models
from rl_game.metrics.stable_retro_scorer import StableRetroScorer

class AttrDict(dict):
    """字典的属性访问版本"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

class DreamerV3ModelLoader:
    """DreamerV3模型加载器"""
    
    def __init__(self, checkpoint_path, game_name, device='auto'):
        self.checkpoint_path = checkpoint_path
        self.game_name = game_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        
        self.config = None
        self.agent = None
        self.env = None
        
    def load_config(self):
        """加载配置"""
        import ruamel.yaml as yaml
        
        # 加载基础配置
        config_file = pathlib.Path(__file__).parent / "configs.yaml"
        configs = yaml.safe_load(config_file.read_text())
        
        # 合并默认配置和retro配置
        config_dict = {}
        config_dict.update(configs['defaults'])
        
        # 安全地更新retro配置
        if 'retro' in configs:
            retro_config = configs['retro']
            for key, value in retro_config.items():
                config_dict[key] = value
        
        # 设置任务相关参数
        config_dict['task'] = f"retro_{self.game_name}"
        config_dict['device'] = str(self.device)
        config_dict['visual_reward'] = False
        config_dict['visual_reward_weight'] = 0.0
        
        # 验证并修复嵌套配置
        config_dict = self._validate_config(config_dict)
        
        # 创建配置对象
        self.config = AttrDict(config_dict)
        
        print(f"✅ 配置加载完成")
        print(f"   🎮 任务: {self.config.task}")
        print(f"   🖥️  设备: {self.config.device}")
    
    def _validate_config(self, config_dict):
        """验证并修复配置"""
        # 确保actor配置存在且格式正确
        if 'actor' not in config_dict or not isinstance(config_dict['actor'], dict):
            config_dict['actor'] = {
                'layers': 2, 
                'dist': 'onehot', 
                'std': 'none',
                'entropy': 3e-4,
                'unimix_ratio': 0.01,
                'min_std': 0.1,
                'max_std': 1.0,
                'temp': 0.1,
                'lr': 3e-5,
                'eps': 1e-5,
                'grad_clip': 100.0,
                'outscale': 1.0
            }
        
        # 确保critic配置存在
        if 'critic' not in config_dict or not isinstance(config_dict['critic'], dict):
            config_dict['critic'] = {
                'layers': 2,
                'dist': 'symlog_disc',
                'slow_target': True,
                'slow_target_update': 1,
                'slow_target_fraction': 0.02,
                'lr': 3e-5,
                'eps': 1e-5,
                'grad_clip': 100.0,
                'outscale': 0.0
            }
        
        # 确保encoder配置存在
        if 'encoder' not in config_dict or not isinstance(config_dict['encoder'], dict):
            config_dict['encoder'] = {
                'mlp_keys': '$^',
                'cnn_keys': 'image',
                'act': 'SiLU',
                'norm': True,
                'cnn_depth': 32,
                'kernel_size': 4,
                'minres': 4,
                'mlp_layers': 5,
                'mlp_units': 1024,
                'symlog_inputs': True
            }
        
        # 确保decoder配置存在
        if 'decoder' not in config_dict or not isinstance(config_dict['decoder'], dict):
            config_dict['decoder'] = {
                'mlp_keys': '$^',
                'cnn_keys': 'image',
                'act': 'SiLU',
                'norm': True,
                'cnn_depth': 32,
                'kernel_size': 4,
                'minres': 4,
                'mlp_layers': 5,
                'mlp_units': 1024,
                'cnn_sigmoid': False,
                'image_dist': 'mse',
                'vector_dist': 'symlog_mse',
                'outscale': 1.0
            }
        
        # 确保其他必要配置
        required_configs = {
            'reward_head': {'layers': 2, 'dist': 'symlog_disc', 'loss_scale': 1.0, 'outscale': 0.0},
            'cont_head': {'layers': 2, 'loss_scale': 1.0, 'outscale': 1.0}
        }
        
        for key, default_value in required_configs.items():
            if key not in config_dict or not isinstance(config_dict[key], dict):
                config_dict[key] = default_value
        
        print("✅ 配置验证完成，所有必要的嵌套配置已确保存在")
        return config_dict
        
    def create_environment(self):
        """创建环境"""
        try:
            self.env = dreamer.make_env(self.config, 'eval', 0)
            
            # 获取动作空间信息并添加到配置中
            if hasattr(self.env.action_space, 'n'):
                self.config.num_actions = self.env.action_space.n
            else:
                self.config.num_actions = self.env.action_space.shape[0] if hasattr(self.env.action_space, 'shape') else 1
            
            print(f"✅ 环境创建成功")
            print(f"   🎯 观察空间: {self.env.observation_space}")
            print(f"   🎮 动作空间: {self.env.action_space}")
            print(f"   🔢 动作数量: {self.config.num_actions}")
            return True
        except Exception as e:
            print(f"❌ 环境创建失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_model(self):
        """加载模型"""
        try:
            print("🔄 正在加载DreamerV3模型...")
            
            # 调试配置内容
            print("🔍 调试配置内容:")
            print(f"   actor存在: {hasattr(self.config, 'actor')}")
            if hasattr(self.config, 'actor'):
                print(f"   actor类型: {type(self.config.actor)}")
                if isinstance(self.config.actor, dict):
                    print(f"   actor keys: {list(self.config.actor.keys())}")
                    print(f"   actor有layers: {'layers' in self.config.actor}")
            
            # 创建虚拟logger和dataset
            class DummyLogger:
                def __init__(self):
                    self.step = 0
                def write(self):
                    pass
            
            class DummyDataset:
                pass
            
            logger = DummyLogger()
            dataset = DummyDataset()
            
            # 创建agent
            self.agent = dreamer.Dreamer(
                self.env.observation_space,
                self.env.action_space,
                self.config,
                logger,
                dataset
            ).to(self.device)
            
            # 加载checkpoint
            if os.path.exists(self.checkpoint_path):
                print(f"🔄 加载checkpoint: {self.checkpoint_path}")
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                
                if 'agent_state_dict' in checkpoint:
                    self.agent.load_state_dict(checkpoint['agent_state_dict'])
                    print("✅ 成功加载agent状态")
                else:
                    print("⚠️  checkpoint格式可能不正确，尝试直接加载")
                    self.agent.load_state_dict(checkpoint)
                
                self.agent.eval()
                print("✅ 模型加载完成并设置为评估模式")
                return True
            else:
                print(f"❌ Checkpoint文件不存在: {self.checkpoint_path}")
                return False
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_episode(self, max_steps=1000, verbose=True):
        """运行一个episode"""
        if self.agent is None or self.env is None:
            raise RuntimeError("模型或环境未正确加载")
        
        # 重置环境
        obs = self.env.reset()
        if verbose:
            print(f"🎮 Episode开始，最大步数: {max_steps}")
        
        # 初始化状态
        total_reward = 0.0
        episode_steps = 0
        frames = []
        done = False
        agent_state = None
        
        while not done and episode_steps < max_steps:
            # 获取动作
            with torch.no_grad():
                action, agent_state = self.agent(obs, episode_steps == 0, agent_state, training=False)
            
            # 执行动作
            obs, reward, done, info = self.env.step(action)
            
            # 记录结果
            total_reward += reward
            episode_steps += 1
            
            # 保存帧用于评分
            frame = self._extract_frame(obs)
            frames.append(frame)
            
            if verbose and episode_steps % 100 == 0:
                print(f"   步数: {episode_steps:>4d}, 累计奖励: {total_reward:>7.2f}")
        
        if verbose:
            print(f"✅ Episode完成: {episode_steps} 步, 总奖励: {total_reward:.2f}")
        
        return episode_steps, total_reward, frames
    
    def _extract_frame(self, obs):
        """从观察中提取帧"""
        # 尝试不同的帧提取方法
        if 'image' in obs:
            frame = obs['image']
        elif hasattr(self.env, 'get_raw_frame'):
            frame = self.env.get_raw_frame()
        elif hasattr(self.env, 'render'):
            frame = self.env.render(mode='rgb_array')
        else:
            # 创建默认帧
            frame = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
        
        # 确保帧格式正确
        if isinstance(frame, np.ndarray):
            if frame.shape != (84, 84, 3):
                # 调整尺寸
                from PIL import Image
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                if len(frame.shape) == 2:  # 灰度图
                    frame = np.stack([frame] * 3, axis=-1)
                
                frame_pil = Image.fromarray(frame)
                frame_resized = frame_pil.resize((84, 84))
                frame = np.array(frame_resized)
                
                if len(frame.shape) == 2:  # 确保是RGB
                    frame = np.stack([frame] * 3, axis=-1)
        
        return frame
    
    def close(self):
        """清理资源"""
        if self.env is not None:
            self.env.close()

def evaluate_model_with_real_inference(checkpoint_path, game_name, num_episodes=3, max_steps=1000, device='auto'):
    """使用真实模型推理进行评估"""
    
    print(f"🚀 开始真实模型评估")
    print(f"   📁 Checkpoint: {checkpoint_path}")
    print(f"   🎮 游戏: {game_name}")
    print(f"   📊 Episodes: {num_episodes}")
    print(f"   ⏱️  最大步数: {max_steps}")
    print("="*80)
    
    # 设置环境变量
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["DISPLAY"] = ":99"
    
    # 初始化评分器
    print("📊 初始化StableRetroScorer...")
    denstream_params = {
        'lambda_': 0.001,
        'eps': 2.0,
        'beta': 0.3,
        'mu': 2
    }
    
    scorer = StableRetroScorer(
        visual_encoder="CLIP",
        state_judger_model_path=None,
        tensorboard_log_dir=f"real_evaluation_scores_{game_name}",
        denstream_params=denstream_params,
        device=device
    )
    
    # 初始化模型加载器
    model_loader = DreamerV3ModelLoader(checkpoint_path, game_name, device)
    
    try:
        # 加载配置、环境和模型
        model_loader.load_config()
        
        if not model_loader.create_environment():
            raise RuntimeError("环境创建失败")
        
        if not model_loader.load_model():
            raise RuntimeError("模型加载失败")
        
        # 运行评估episodes
        episode_results = []
        
        for episode in range(num_episodes):
            print(f"\n🎯 Episode {episode + 1}/{num_episodes}")
            print("-" * 60)
            
            # 重置评分器
            scorer.reset_episode(reset_l2_clusterer=(episode == 0))
            
            try:
                # 运行真实模型推理
                episode_steps, episode_reward, frames = model_loader.run_episode(max_steps, verbose=True)
                
                # 使用真实帧进行评分
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
                    'real_inference': True  # 标记这是真实推理结果
                }
                
                episode_results.append(episode_result)
                
                print(f"📊 Episode {episode + 1} 结果:")
                print(f"   L1 (存活): {episode_result['final_l1']:.1f} steps")
                print(f"   L2 (多样性): {episode_result['final_l2']:.3f}")
                print(f"   L3 (目标): {episode_result['final_l3_prediction']} (conf: {episode_result['final_l3_confidence']:.3f})")
                print(f"   总奖励: {episode_result['reward']:.2f}")
                
            except Exception as e:
                print(f"❌ Episode {episode + 1} 运行失败: {e}")
                # 记录失败的episode
                episode_results.append({
                    'episode': episode + 1,
                    'error': str(e),
                    'real_inference': False
                })
        
        return episode_results
        
    except Exception as e:
        print(f"❌ 评估过程出错: {e}")
        import traceback
        traceback.print_exc()
        return []
        
    finally:
        scorer.close()
        model_loader.close()

def print_evaluation_summary(episode_results):
    """打印评估摘要"""
    print("\n" + "="*80)
    print("📈 真实模型评估摘要报告")
    print("="*80)
    
    # 过滤成功的episodes
    successful_episodes = [ep for ep in episode_results if ep.get('real_inference', False)]
    failed_episodes = [ep for ep in episode_results if not ep.get('real_inference', False)]
    
    if failed_episodes:
        print(f"⚠️  失败的Episodes: {len(failed_episodes)}/{len(episode_results)}")
    
    if not successful_episodes:
        print("❌ 没有成功的episode可供分析")
        return
    
    # 基本统计
    num_episodes = len(successful_episodes)
    avg_steps = np.mean([ep['steps'] for ep in successful_episodes])
    avg_reward = np.mean([ep['reward'] for ep in successful_episodes])
    
    # L1评分统计
    l1_scores = [ep['final_l1'] for ep in successful_episodes]
    avg_l1 = np.mean(l1_scores)
    max_l1 = np.max(l1_scores)
    min_l1 = np.min(l1_scores)
    
    # L2评分统计
    l2_scores = [ep['final_l2'] for ep in successful_episodes]
    avg_l2 = np.mean(l2_scores)
    max_l2 = np.max(l2_scores)
    min_l2 = np.min(l2_scores)
    
    # L3评分统计
    l3_predictions = [ep['final_l3_prediction'] for ep in successful_episodes]
    win_count = l3_predictions.count('win')
    loss_count = l3_predictions.count('loss')
    else_count = l3_predictions.count('else')
    
    victory_count = sum(1 for ep in successful_episodes if ep['is_victory'])
    defeat_count = sum(1 for ep in successful_episodes if ep['is_defeat'])
    
    print(f"🎮 成功Episodes: {num_episodes}")
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
    print(f"   实际胜利: {victory_count}/{num_episodes} ({victory_count/num_episodes*100:.1f}%)")
    print(f"   实际失败: {defeat_count}/{num_episodes} ({defeat_count/num_episodes*100:.1f}%)")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="真实DreamerV3模型评估器")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--game", type=str, default="BomberRaid-Sms", help="游戏名称")
    parser.add_argument("--episodes", type=int, default=3, help="评估episode数")
    parser.add_argument("--max-steps", type=int, default=1000, help="每个episode最大步数")
    parser.add_argument("--device", type=str, default="auto", help="计算设备 (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    print(f"🚀 真实DreamerV3模型评估器")
    print(f"   📁 Checkpoint: {args.checkpoint}")
    print(f"   🎮 游戏: {args.game}")
    print(f"   📊 Episodes: {args.episodes}")
    print(f"   ⏱️  最大步数: {args.max_steps}")
    print(f"   🖥️  设备: {args.device}")
    print()
    
    # 检查checkpoint文件
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint文件不存在: {args.checkpoint}")
        return
    
    # 运行评估
    try:
        results = evaluate_model_with_real_inference(
            checkpoint_path=args.checkpoint,
            game_name=args.game,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            device=args.device
        )
        
        # 打印摘要
        if results:
            print_evaluation_summary(results)
        
        # 保存结果
        result_file = f"real_evaluation_results_{args.game}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(result_file, 'w') as f:
            json.dump({
                'metadata': {
                    'script_version': 'real_model_evaluate_dreamer.py',
                    'evaluation_type': 'real_inference',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'config': {
                    'checkpoint': str(args.checkpoint),
                    'game': str(args.game),
                    'episodes': int(args.episodes),
                    'max_steps': int(args.max_steps),
                    'device': str(args.device)
                },
                'results': results
            }, f, indent=2)
        
        print(f"\n💾 详细结果已保存至: {result_file}")
        print("✅ 真实模型评估完成!")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
