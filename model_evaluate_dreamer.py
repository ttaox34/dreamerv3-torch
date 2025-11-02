#!/usr/bin/env python3
"""
DreamerV3真实模型评估脚本
使用与dreamer.py相同的配置加载方式，确保配置一致性
"""

import argparse
import functools
import os
import pathlib
import sys
import time
import json
import numpy as np
import torch
from datetime import datetime

# 设置环境变量
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["DISPLAY"] = ":99"

# 添加项目路径
sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append('/home/zhuolifeng/dreamerv3-torch')
sys.path.append('/home/zhuolifeng/dreamerv3-torch/rl_game')

import ruamel.yaml as yaml
import tools
import models
import envs.wrappers as wrappers
import exploration as expl
from parallel import Parallel, Damy
from rl_game.metrics.stable_retro_scorer import StableRetroScorer

to_np = lambda x: x.detach().cpu().numpy()

class Dreamer(torch.nn.Module):
    """DreamerV3 Agent - 与原始dreamer.py完全一致"""
    
    def __init__(self, obs_space, act_space, config, logger=None, dataset=None):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = 0  # logger.step // config.action_repeat if logger else 0
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        
        # 添加探索行为（即使在推理中也需要，因为检查点包含它）
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)
        
        # 模型编译（与原始版本一致）
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        
    def _policy(self, obs, state, training=False):
        """推理策略"""
        if state is None:
            latent = action = None
        else:
            latent, action = state
        
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
            
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        
        policy_output = {"action": action, "logits": actor.logits}
        state = (latent, action)
        return policy_output, state

    def __call__(self, obs, reset, state=None, training=False):
        """前向传播"""
        return self._policy(obs, state, training)

def load_config_like_dreamer(configs_list=None):
    """使用与dreamer.py相同的方式加载配置"""
    
    # 读取配置文件
    config_path = pathlib.Path(__file__).parent / "configs.yaml"
    configs = yaml.safe_load(config_path.read_text())
    
    def recursive_update(base, update):
        """递归更新配置"""
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value
    
    # 构建配置名称列表
    name_list = ["defaults"]
    if configs_list:
        name_list.extend(configs_list)
    
    # 递归合并配置
    defaults = {}
    for name in name_list:
        if name in configs:
            recursive_update(defaults, configs[name])
        else:
            print(f"Warning: Config '{name}' not found in configs.yaml")
    
    return defaults

def make_env_like_dreamer(config, mode, id):
    """使用与dreamer.py相同的方式创建环境"""
    suite, task = config["task"].split("_", 1)
    
    if suite == "retro":
        import envs.stable_retro as stable_retro
        
        env = stable_retro.StableRetro(
            game=task,
            action_repeat=config["action_repeat"],
            size=config["size"],
            grayscale=config["grayscale"],
            seed=config["seed"] + id,
        )
        
        # Add visual reward wrapper if enabled
        if config.get('visual_reward', False) and config.get('visual_reward_weight', 0.0) > 0.0:
            try:
                from envs.visual_reward_wrapper import VisualRewardWrapper
                
                visual_device = config.get('visual_device', 'auto')
                if visual_device == 'auto':
                    visual_device = config.get('device', 'auto')
                
                env = VisualRewardWrapper(
                    env,
                    visual_encoder=config.get('visual_encoder', 'CLIP'),
                    visual_reward_weight=config.get('visual_reward_weight', 0.1),
                    episode_length=config.get('visual_episode_length', 1000),
                    device=visual_device
                )
                print(f"Visual reward enabled for retro environment (weight: {config['visual_reward_weight']})")
            except Exception as e:
                print(f"Warning: Failed to enable visual reward: {e}")
                print("Continuing without visual reward...")
        
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(f"Suite '{suite}' not implemented in this evaluation script")
    
    env = wrappers.TimeLimit(env, config["time_limit"])
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    
    return env

def create_attr_dict(d):
    """将普通字典转换为AttrDict（支持点号访问）"""
    class AttrDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(f"'AttrDict' object has no attribute '{key}'")
                
        def __setattr__(self, key, value):
            self[key] = value
            
        def __contains__(self, key):
            return dict.__contains__(self, key)
            
        def __iter__(self):
            return dict.__iter__(self)
            
        def __len__(self):
            return dict.__len__(self)
            
        def keys(self):
            return dict.keys(self)
            
        def values(self):
            return dict.values(self)
            
        def items(self):
            return dict.items(self)
    
    if isinstance(d, dict):
        attr_dict = AttrDict()
        for key, value in d.items():
            if isinstance(value, dict):
                attr_dict[key] = create_attr_dict(value)
            else:
                attr_dict[key] = value
        return attr_dict
    else:
        return d

def load_agent_checkpoint(logdir, config, env):
    """加载DreamerV3检查点"""
    print(f"🔄 加载检查点从: {logdir}")
    
    checkpoint_path = pathlib.Path(logdir) / "latest.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 创建agent
    print("🤖 创建DreamerV3 agent...")
    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config
    ).to(config.device)
    
    # 加载检查点
    print("📥 加载检查点状态...")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    # 处理编译后的模型 - 使用strict=False并尝试加载兼容的权重
    try:
        agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
    except RuntimeError as e:
        print(f"⚠️  严格模式加载失败，尝试灵活加载: {e}")
        
        # 尝试处理编译模型的key映射
        state_dict = checkpoint["agent_state_dict"]
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # 移除 _orig_mod 前缀（torch.compile产生的）
            new_key = key.replace("._orig_mod", "")
            new_state_dict[new_key] = value
        
        # 尝试使用处理后的state dict
        try:
            missing_keys, unexpected_keys = agent.load_state_dict(new_state_dict, strict=False)
            print(f"📋 灵活加载完成:")
            print(f"   缺失的键数量: {len(missing_keys)}")
            print(f"   意外的键数量: {len(unexpected_keys)}")
            if len(missing_keys) > 0:
                print(f"   示例缺失键: {missing_keys[:3]}")
            if len(unexpected_keys) > 0:
                print(f"   示例意外键: {unexpected_keys[:3]}")
        except Exception as e2:
            print(f"❌ 灵活加载也失败: {e2}")
            raise e2
    
    agent.eval()  # 设置为评估模式
    
    print("✅ 检查点加载成功!")
    return agent

def evaluate_with_real_model(
    checkpoint_dir,
    game_name,
    num_episodes=3,
    max_steps=1000,
    configs_list=None,
    device='auto',
    visual_reward_weight=0.0
):
    """使用真实DreamerV3模型进行评估"""
    
    print(f"🚀 DreamerV3真实模型评估")
    print(f"   📁 检查点目录: {checkpoint_dir}")
    print(f"   🎮 游戏: {game_name}")
    print(f"   📊 Episodes: {num_episodes}")
    print(f"   ⏱️  最大步数: {max_steps}")
    print(f"   🖥️  设备: {device}")
    print("="*80)
    
    # 1. 加载配置（使用dreamer.py相同方式）
    print("⚙️  加载配置...")
    config_dict = load_config_like_dreamer(configs_list)
    
    # 设置游戏任务
    config_dict["task"] = f"retro_{game_name}"
    
    # 设置设备
    if device != 'auto':
        config_dict["device"] = device
    elif device == 'auto':
        config_dict["device"] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 可选：覆盖visual reward设置
    if visual_reward_weight > 0.0:
        config_dict["visual_reward"] = True
        config_dict["visual_reward_weight"] = visual_reward_weight
    
    # 转换为AttrDict以支持点号访问
    config = create_attr_dict(config_dict)
    
    print(f"   任务: {config.task}")
    print(f"   设备: {config.device}")
    print(f"   动作重复: {config.action_repeat}")
    print(f"   尺寸: {config.size}")
    
    # 2. 创建环境
    print("🌍 创建环境...")
    env = make_env_like_dreamer(config_dict, "eval", 0)
    env = Damy(env)  # 不使用并行
    
    # 设置动作数量
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    print(f"   动作空间: {acts}")
    print(f"   动作数量: {config.num_actions}")
    
    # 3. 加载模型
    agent = load_agent_checkpoint(checkpoint_dir, config, env)
    
    # 4. 初始化评分器
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
        tensorboard_log_dir=f"real_model_evaluation_{game_name}",
        denstream_params=denstream_params,
        device=config.device
    )
    
    episode_results = []
    
    try:
        for episode in range(num_episodes):
            print(f"\n🎯 Episode {episode + 1}/{num_episodes}")
            print("-" * 60)
            
            # 重置评分器
            scorer.reset_episode(reset_l2_clusterer=(episode == 0))
            
            # 重置环境
            reset_fn = env.reset()
            if callable(reset_fn):
                obs_result = reset_fn()  # Damy包装器返回函数
            else:
                obs_result = reset_fn
                
            if isinstance(obs_result, tuple):
                obs = obs_result[0]  # 新版本gym返回(obs, info)
            else:
                obs = obs_result
            
            # 处理字典格式的观察值（StableRetro返回字典）
            if isinstance(obs, dict) and "image" in obs:
                current_image = obs["image"]
            elif isinstance(obs, np.ndarray):
                current_image = obs
            else:
                print(f"⚠️  意外的观察值类型: {type(obs)}")
                current_image = np.zeros((64, 64, 3), dtype=np.uint8)
            
            state = None
            done = False
            step = 0
            episode_reward = 0.0
            frames = []
            
            print(f"🔄 运行episode（最大 {max_steps} 步）...")
            
            while not done and step < max_steps:
                # 使用DreamerV3进行推理
                with torch.no_grad():
                    # 准备模型输入 - 使用当前观察值
                    if isinstance(obs, dict):
                        model_obs = {}
                        for key, value in obs.items():
                            if isinstance(value, np.ndarray):
                                model_obs[key] = torch.from_numpy(value).unsqueeze(0).to(config.device)
                            elif torch.is_tensor(value):
                                model_obs[key] = value.unsqueeze(0).to(config.device)
                            else:
                                model_obs[key] = value
                    else:
                        model_obs = {"image": torch.from_numpy(obs).unsqueeze(0).to(config.device)}
                    
                    # 归一化图像数据
                    for key, value in model_obs.items():
                        if key == "image" and torch.is_tensor(value):
                            if value.dtype == torch.uint8:
                                model_obs[key] = value.float() / 255.0
                            elif value.max() > 1.0:
                                model_obs[key] = value.float() / 255.0
                    
                    # 添加is_first标志
                    model_obs["is_first"] = torch.tensor([step == 0], dtype=torch.bool, device=config.device)
                    
                    # 模型推理
                    policy_output, state = agent(model_obs, torch.tensor([False]), state, training=False)
                    action = policy_output["action"]
                    
                    # 转换动作为环境格式
                    if len(action.shape) > 1:
                        action = action.squeeze(0)
                    action_array = to_np(action)
                    
                    # 确保动作格式正确（对于SelectAction包装器）
                    # SelectAction包装器需要包含"action"键的字典
                    action_dict = {"action": action_array}
                
                # 执行动作
                step_fn = env.step(action_dict)
                if callable(step_fn):
                    step_result = step_fn()  # Damy包装器返回函数
                else:
                    step_result = step_fn
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                elif len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    raise ValueError(f"Unexpected step result length: {len(step_result)}")
                
                episode_reward += reward
                step += 1
                
                # 收集帧用于评分（从观察值中提取图像）
                if isinstance(obs, dict) and "image" in obs:
                    current_image = obs["image"]
                elif isinstance(obs, np.ndarray) and len(obs.shape) == 3:
                    current_image = obs
                else:
                    # 创建虚拟帧
                    current_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                
                frames.append(current_image)
                
                # 更新评分器
                score_result = scorer.update_frame(
                    frame=current_image,
                    step_count=step,
                    episode_end=done,
                    terminated=done,
                    verbose=(step % 100 == 0)
                )
                
                if step % 100 == 0:
                    print(f"   步骤 {step}: 奖励={episode_reward:.2f}")
            
            # Episode结束，最终评分更新
            if frames:
                score_result = scorer.update_frame(
                    frame=frames[-1],
                    step_count=step,
                    episode_end=True,
                    terminated=True,
                    verbose=True
                )
            
            # 记录episode结果
            episode_result = {
                'episode': episode + 1,
                'steps': int(step),
                'reward': float(episode_reward),
                'final_l1': float(score_result.get('l1_survival', 0)),
                'final_l2': float(scorer.get_final_l2_cluster_width()),
                'final_l3_prediction': str(score_result.get('l3_prediction', 'unknown')),
                'final_l3_confidence': float(score_result.get('l3_confidence', 0.0)),
                'is_victory': bool(score_result.get('is_victory', False)),
                'is_defeat': bool(score_result.get('is_defeat', False)),
                'evaluation_type': 'real_model',
                'checkpoint_dir': str(checkpoint_dir)
            }
            
            episode_results.append(episode_result)
            
            print(f"📊 Episode {episode + 1} 结果:")
            print(f"   步数: {episode_result['steps']}")
            print(f"   总奖励: {episode_result['reward']:.2f}")
            print(f"   L1 (存活): {episode_result['final_l1']:.1f} steps")
            print(f"   L2 (多样性): {episode_result['final_l2']:.3f}")
            print(f"   L3 (目标): {episode_result['final_l3_prediction']} (conf: {episode_result['final_l3_confidence']:.3f})")
        
        return episode_results
        
    except Exception as e:
        print(f"❌ 评估过程出错: {e}")
        import traceback
        traceback.print_exc()
        return episode_results
        
    finally:
        scorer.close()
        env.close()

def print_evaluation_summary(episode_results):
    """打印评估摘要"""
    print("\n" + "="*80)
    print("📈 真实模型评估摘要报告")
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
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="DreamerV3真实模型评估器")
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点目录路径")
    parser.add_argument("--game", type=str, default="BomberRaid-Sms", help="游戏名称")
    parser.add_argument("--episodes", type=int, default=3, help="评估episode数")
    parser.add_argument("--max-steps", type=int, default=1000, help="每个episode最大步数")
    parser.add_argument("--configs", nargs="+", help="额外的配置名称（如retro, debug等）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备 (auto, cpu, cuda, cuda:0)")
    parser.add_argument("--visual-reward-weight", type=float, default=0.0, help="视觉奖励权重")
    
    args = parser.parse_args()
    
    print(f"🚀 DreamerV3真实模型评估器")
    print(f"   📁 检查点: {args.checkpoint}")
    print(f"   🎮 游戏: {args.game}")
    print(f"   📊 Episodes: {args.episodes}")
    print(f"   ⏱️  最大步数: {args.max_steps}")
    print(f"   ⚙️  配置: {args.configs}")
    print(f"   🖥️  设备: {args.device}")
    print()
    
    # 运行评估
    results = evaluate_with_real_model(
        checkpoint_dir=args.checkpoint,
        game_name=args.game,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        configs_list=args.configs,
        device=args.device,
        visual_reward_weight=args.visual_reward_weight
    )
    
    # 打印摘要
    if results:
        print_evaluation_summary(results)
    
    # 保存结果
    result_file = f"real_model_evaluation_{args.game}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(result_file, 'w') as f:
        json.dump({
            'metadata': {
                'script_version': 'model_evaluate_dreamer.py',
                'evaluation_type': 'real_model',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'config': {
                'checkpoint_dir': str(args.checkpoint),
                'game': str(args.game),
                'episodes': int(args.episodes),
                'max_steps': int(args.max_steps),
                'configs': args.configs,
                'device': str(args.device),
                'visual_reward_weight': float(args.visual_reward_weight)
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 详细结果已保存至: {result_file}")
    print("✅ 真实模型评估完成!")

if __name__ == "__main__":
    main()
