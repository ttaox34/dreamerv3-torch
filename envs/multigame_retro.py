import numpy as np
try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        gym = None
import torch
from collections import defaultdict
import threading
import time
from typing import List, Dict, Any, Optional, Tuple

from .stable_retro import StableRetro


# 全局锁确保同一时间只有一个模拟器实例
_emulator_lock = threading.Lock()
_current_emulator = None


class MultiGameRetroEnv(gym.Env):
    """多游戏Retro环境管理器，支持多个不同的Retro游戏并行训练"""
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        games: List[str],
        action_repeat: int = 4,
        size: Tuple[int, int] = (64, 64),
        grayscale: bool = False,
        seed: Optional[int] = None,
        visual_reward: bool = False,
        visual_reward_weight: float = 0.1,
        visual_encoder: str = 'CLIP',
        visual_episode_length: int = 1000,
        visual_device: str = 'auto'
    ):
        """
        初始化多游戏环境
        
        Args:
            games: 游戏名称列表
            action_repeat: 动作重复次数
            size: 图像尺寸
            grayscale: 是否使用灰度图
            seed: 随机种子
            visual_reward: 是否启用视觉奖励
            visual_reward_weight: 视觉奖励权重
            visual_encoder: 视觉编码器类型
            visual_episode_length: 视觉奖励回合长度
            visual_device: 视觉编码器设备
        """
        self.games = games
        self.num_games = len(games)
        self.action_repeat = action_repeat
        self.size = size
        self.grayscale = grayscale
        self.seed = seed
        self.visual_reward = visual_reward
        self.visual_reward_weight = visual_reward_weight
        self.visual_encoder = visual_encoder
        self.visual_episode_length = visual_episode_length
        self.visual_device = visual_device
        
        # 为每个游戏创建环境（每个游戏在独立的进程中）
        self.envs = []
        self.game_to_idx = {game: idx for idx, game in enumerate(games)}
        
        # 初始化动作空间相关属性
        self.action_spaces = []
        self.num_actions_list = []
        self.max_num_actions = 0
        
        # 创建环境
        self._create_envs()
        
        # 设置动作空间
        self._setup_action_spaces()
        
        # 设置观察空间
        self._setup_observation_spaces()
        
        # 当前活跃的游戏索引
        self.current_game_idx = 0
        
        # 游戏切换策略
        self.game_switch_strategy = 'random'
        self.game_switch_freq = 1000
        self.episode_count = 0
        
    def _create_envs(self):
        """为每个游戏创建环境"""
        print(f"Creating environments for games: {self.games}")
        
        # 由于stable-retro的限制，我们需要延迟创建环境
        # 我们将在第一次使用时创建环境，或者每次切换游戏时重新创建
        self.envs = [None] * len(self.games)  # 延迟初始化
        self._current_env = None
        self._current_game_idx = 0
        
        # 预先计算动作空间信息
        self._precompute_action_spaces()
        
        print(f"MultiGameRetroEnv initialized with {len(self.games)} games")
        print(f"Action spaces will be created on-demand")
    
    def _precompute_action_spaces(self):
        """预计算动作空间信息"""
        # 为了避免同时创建多个模拟器，我们创建临时环境来获取动作空间信息
        temp_envs = []
        
        try:
            for i, game in enumerate(self.games):
                print(f"Getting action space for {game}...")
                
                # 创建临时环境
                temp_env = StableRetro(
                    game=game,
                    action_repeat=self.action_repeat,
                    size=self.size,
                    grayscale=self.grayscale,
                    seed=self.seed + i if self.seed is not None else None
                )
                
                # 获取动作空间信息
                action_space = temp_env.action_space
                if hasattr(action_space, 'n'):
                    num_actions = action_space.n
                else:
                    num_actions = action_space.shape[0] if hasattr(action_space, 'shape') else 1
                
                self.action_spaces.append(action_space)
                self.num_actions_list.append(num_actions)
                self.max_num_actions = max(self.max_num_actions, num_actions)
                
                # 关闭临时环境
                temp_env.close()
                temp_envs.append(None)
                
        except Exception as e:
            print(f"Error during action space precomputation: {e}")
            # 清理已创建的环境
            for env in temp_envs:
                if env is not None:
                    try:
                        env.close()
                    except:
                        pass
            raise
    
    def _get_or_create_env(self, game_idx):
        """获取或创建指定游戏的环境"""
        global _emulator_lock, _current_emulator
        
        if self.envs[game_idx] is None:
            with _emulator_lock:
                # 关闭当前环境（如果存在）
                if _current_emulator is not None:
                    try:
                        _current_emulator.close()
                    except:
                        pass
                    _current_emulator = None
                
                # 创建新环境
                game = self.games[game_idx]
                print(f"Creating environment for {game}...")
                
                env = StableRetro(
                    game=game,
                    action_repeat=self.action_repeat,
                    size=self.size,
                    grayscale=self.grayscale,
                    seed=self.seed + game_idx if self.seed is not None else None
                )
                
                # 添加视觉奖励包装器
                if self.visual_reward and self.visual_reward_weight > 0.0:
                    try:
                        from .visual_reward_wrapper import VisualRewardWrapper
                        
                        visual_device = self.visual_device
                        if visual_device == 'auto':
                            visual_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                        
                        env = VisualRewardWrapper(
                            env,
                            visual_encoder=self.visual_encoder,
                            visual_reward_weight=self.visual_reward_weight,
                            episode_length=self.visual_episode_length,
                            device=visual_device
                        )
                        print(f"Visual reward enabled for {game} (weight: {self.visual_reward_weight})")
                    except Exception as e:
                        print(f"Warning: Failed to enable visual reward for {game}: {e}")
                        print("Continuing without visual reward...")
                
                self.envs[game_idx] = env
                self._current_env = env
                _current_emulator = env
                self._current_game_idx = game_idx
        
        return self.envs[game_idx]
    
    def _setup_action_spaces(self):
        """设置动作空间信息"""
        # 这些已经在_precompute_action_spaces中设置了
        # 创建统一的动作空间（使用最大动作数）
        self.action_space = gym.spaces.Discrete(self.max_num_actions)
        
        print(f"Multi-game action spaces: {dict(zip(self.games, self.num_actions_list))}")
        print(f"Max actions per game: {self.max_num_actions}")
    
    def _setup_observation_spaces(self):
        """设置观察空间"""
        # 创建一个模板观察空间
        # 假设所有游戏都有相同的观察空间结构
        image_shape = self.size + (3,) if not self.grayscale else self.size + (1,)
        
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "game_id": gym.spaces.Box(low=0, high=self.num_games - 1, shape=(), dtype=np.int32)
        })
    
    def reset(self, game_idx: Optional[int] = None, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        重置指定游戏的环境
        
        Args:
            game_idx: 游戏索引，如果为None则根据策略选择游戏
            seed: 随机种子
            options: 重置选项
        
        Returns:
            观察字典
        """
        old_game_idx = self.current_game_idx
        
        # 游戏切换逻辑
        if game_idx is None:
            # 根据策略选择游戏
            if self.game_switch_strategy == 'random':
                self.current_game_idx = np.random.randint(0, self.num_games)
            elif self.game_switch_strategy == 'round_robin':
                self.current_game_idx = (self.current_game_idx + 1) % self.num_games
            elif self.game_switch_strategy == 'episode_based':
                # 每N个回合切换一次游戏
                if self.episode_count % self.game_switch_freq == 0:
                    self.current_game_idx = (self.current_game_idx + 1) % self.num_games
            # 否则保持当前游戏
        else:
            self.current_game_idx = game_idx
            
        self.episode_count += 1
        
        # 打印游戏切换信息
        if old_game_idx != self.current_game_idx:
            print(f"Switching from game {self.games[old_game_idx]} to game {self.games[self.current_game_idx]} (episode {self.episode_count})")
        
        # 获取或创建环境
        env = self._get_or_create_env(self.current_game_idx)
        
        # 重置环境
        if seed is not None and hasattr(env, 'reset'):
            obs = env.reset(seed=seed, options=options)
        else:
            obs = env.reset()
        
        if isinstance(obs, tuple):
            obs = obs[0]  # 处理新的gym API
        
        # 添加游戏ID到观察中
        obs['game_id'] = np.array(self.current_game_idx, dtype=np.int32)
        
        return obs
    
    def step(self, action: int, game_idx: Optional[int] = None):
        """
        在指定游戏中执行动作
        
        Args:
            action: 要执行的动作
            game_idx: 游戏索引，如果为None则使用当前游戏
        
        Returns:
            (观察, 奖励, 是否结束, 信息)
        """
        if game_idx is not None:
            self.current_game_idx = game_idx
        
        # 获取或创建环境
        env = self._get_or_create_env(self.current_game_idx)
        
        # 确保动作在有效范围内
        max_action = self.num_actions_list[self.current_game_idx] - 1
        if isinstance(action, np.ndarray):
            action = action.item() if action.size == 1 else action[0]
        elif hasattr(action, '__len__') and len(action) == 1:
            action = action[0]
        action = min(int(action), max_action)
        
        # 执行动作
        step_result = env.step(action)
        
        # 处理不同的gym API返回格式
        if len(step_result) == 5:
            # 新的gym API: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            # 旧的gym API: (obs, reward, done, info)
            obs, reward, done, info = step_result
        
        # 添加游戏ID到观察中
        obs['game_id'] = np.array(self.current_game_idx, dtype=np.int32)
        
        return obs, reward, done, info
    
    def get_game_info(self, game_idx: Optional[int] = None) -> Dict[str, Any]:
        """获取游戏信息"""
        if game_idx is None:
            game_idx = self.current_game_idx
        
        return {
            'game_name': self.games[game_idx],
            'game_idx': game_idx,
            'num_actions': self.num_actions_list[game_idx],
            'action_space': self.action_spaces[game_idx]
        }
    
    def set_current_game(self, game_idx: int):
        """设置当前游戏"""
        if 0 <= game_idx < self.num_games:
            self.current_game_idx = game_idx
        else:
            raise ValueError(f"Game index {game_idx} out of range [0, {self.num_games - 1}]")
    
    def set_game_switch_strategy(self, strategy: str, freq: int = 1000):
        """设置游戏切换策略
        
        Args:
            strategy: 切换策略 ('random', 'round_robin', 'episode_based', 'none')
            freq: 切换频率（对于episode_based策略）
        """
        if strategy in ['random', 'round_robin', 'episode_based', 'none']:
            self.game_switch_strategy = strategy
            self.game_switch_freq = freq
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def get_valid_actions(self, game_idx: Optional[int] = None) -> List[int]:
        """获取指定游戏的有效动作列表"""
        if game_idx is None:
            game_idx = self.current_game_idx
        
        return list(range(self.num_actions_list[game_idx]))
    
    def close(self):
        """关闭所有环境"""
        global _emulator_lock, _current_emulator
        
        with _emulator_lock:
            for env in self.envs:
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass
            _current_emulator = None
    
    def __len__(self):
        """返回游戏数量"""
        return self.num_games
    
    def __getitem__(self, game_idx: int):
        """获取指定游戏的环境"""
        return self.envs[game_idx]
    
    def __iter__(self):
        """迭代所有游戏环境"""
        return iter(self.envs)
    
    def render(self, mode='human'):
        """渲染当前游戏环境"""
        if self._current_env is not None:
            return self._current_env.render(mode)
        else:
            return None


def create_multigame_retro_env(
    games: List[str],
    **kwargs
) -> MultiGameRetroEnv:
    """
    创建多游戏Retro环境的便捷函数
    
    Args:
        games: 游戏名称列表
        **kwargs: 其他参数传递给MultiGameRetroEnv
    
    Returns:
        MultiGameRetroEnv实例
    """
    return MultiGameRetroEnv(games=games, **kwargs)