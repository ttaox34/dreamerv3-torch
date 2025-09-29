"""
奖励管理器，处理三种不同的奖励模式
- L1: 生存奖励模式
- L2: 视觉奖励模式  
- L3: 环境原始奖励模式
"""

import numpy as np
import torch
import retro
import gymnasium as gym
from typing import Dict, Any, Optional


class RewardManager:
    """奖励管理器，处理三种不同的奖励模式"""
    def __init__(self, reward_mode: str, games_to_train: list, visual_encoder: str = "CLIP", device: str = "cuda"):
        self.reward_mode = reward_mode
        self.games_to_train = games_to_train
        self.task_num = len(games_to_train)
        self.device = device
        
        # 生存奖励模式的参数
        self.survival_reward = 0.1  # 每步生存奖励
        self.death_penalty = -1.0   # 死亡惩罚
        
        # 初始化视觉奖励组件（如果使用L2模式）
        if reward_mode == "L2":
            # 尝试导入视觉奖励模块
            try:
                from reward_generator import RewardGenerator
                VISION_REWARD_AVAILABLE = True
            except ImportError:
                print("警告: 无法导入reward_generator模块，L2奖励模式将不可用")
                VISION_REWARD_AVAILABLE = False

            # 尝试导入视觉编码器
            try:
                from transformers import CLIPVisionModel, AutoImageProcessor
                VISION_MODEL_AVAILABLE = True
            except ImportError:
                print("警告: 无法导入transformers模块，L2奖励模式将不可用")
                VISION_MODEL_AVAILABLE = False
                
            if VISION_REWARD_AVAILABLE and VISION_MODEL_AVAILABLE:
                print("初始化共享视觉编码器...")
                self.shared_encoder = SharedVisualEncoder(visual_encoder=visual_encoder, device=device)
                
                # 为每个游戏初始化独立的奖励生成器
                self.reward_generators = {
                    game: RewardGenerator() for game in games_to_train
                }
            else:
                print("视觉奖励不可用，回退到L3模式")
                self.reward_mode = "L3"  # 回退到L3模式
                self.shared_encoder = None
                self.reward_generators = None
        else:
            self.shared_encoder = None
            self.reward_generators = None
            
        print(f"奖励模式: {self.reward_mode}")

    def compute_reward(self, game_name: str, env_reward: float, done: bool, obs: Optional[np.ndarray] = None):
        """根据奖励模式计算奖励"""
        if self.reward_mode == "L1":
            # L1模式：生存奖励
            if done:
                return self.death_penalty
            else:
                return self.survival_reward
                
        elif self.reward_mode == "L2":
            # L2模式：视觉奖励
            if self.shared_encoder is None or obs is None:
                # 回退到环境奖励
                return env_reward
                
            try:
                # 使用共享编码器提取特征
                feature = self.shared_encoder.extract_features_from_obs(obs)
                # 使用游戏特定的奖励生成器计算奖励
                visual_reward = self.reward_generators[game_name].generate_reward(feature)
                return visual_reward
            except Exception as e:
                print(f"视觉奖励计算错误: {e}")
                return env_reward
                
        else:  # L3模式
            # L3模式：环境原始奖励
            return env_reward


class SharedVisualEncoder:
    """共享视觉编码器，所有游戏共用一个编码器"""
    def __init__(self, visual_encoder="CLIP", device="cuda"):
        self.device = device
        self.visual_encoder = None
        self.visual_processor = None
        self.visual_encoder_init(visual_encoder)
        print(f"使用共享视觉编码器: {visual_encoder}")
        
    def visual_encoder_init(self, encoder):
        if encoder == "CLIP":
            try:
                from transformers import CLIPVisionModel, AutoImageProcessor
                self.visual_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
                self.visual_encoder.to(self.device)
                self.visual_encoder.eval()
                self.visual_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
            except ImportError:
                raise ImportError("CLIP视觉编码器不可用，请安装transformers库")
        else:
            # 如果CLIP不可用，回退到其他编码器或抛出错误
            raise ImportError(f"视觉编码器 {encoder} 不支持或不可用")
    
    def extract_features(self, frame):
        """提取图像特征"""
        with torch.no_grad():
            inputs = self.visual_processor(images=frame, return_tensors="pt").to(self.device)
            outputs = self.visual_encoder(**inputs)
            feature = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return feature
    
    def extract_features_from_obs(self, obs):
        """从观测值提取特征"""
        # 观测值已经是经过预处理的帧，可能需要转换为适合CLIP的格式
        # 假设obs是形状为(4, 84, 84)的堆叠帧
        # 我们需要取最后一帧并转换为RGB格式
        
        # 取最后一帧（当前帧）
        current_frame = obs[-1] if len(obs.shape) == 3 else obs
        
        # 将单通道转换为三通道（复制灰度值到三个通道）
        if len(current_frame.shape) == 2:  # 灰度图像
            current_frame_rgb = np.stack([current_frame] * 3, axis=-1)
        else:  # 已经是多通道
            current_frame_rgb = current_frame
        
        # 调整大小为CLIP期望的输入尺寸（224x224）
        import cv2
        resized_frame = cv2.resize(current_frame_rgb, (224, 224))
        
        # 使用CLIP提取特征
        return self.extract_features(resized_frame)