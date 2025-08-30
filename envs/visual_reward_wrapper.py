"""
视觉奖励包装器 - 用于DreamerV3
基于预训练视觉模型计算的内在奖励
"""
import numpy as np
import torch
try:
    import gymnasium as gym
except ImportError:
    import gym
from PIL import Image
from transformers import ViTModel, CLIPVisionModel, AutoModel, AutoImageProcessor
import torchvision
from torchvision import transforms
import sys
import os

# 添加rl_game到sys.path以便导入RewardGenerator
sys.path.append('/home/zhuolifeng/rl_game')

try:
    from reward_generator.reward_generator import RewardGenerator
except ImportError:
    print("Warning: Cannot import RewardGenerator, visual reward will be disabled")
    RewardGenerator = None


class VisualRewardWrapper(gym.Wrapper):
    """
    视觉奖励包装器
    在原有环境奖励基础上增加基于视觉特征的内在奖励
    """
    
    def __init__(self, env, visual_encoder="CLIP", visual_reward_weight=0.1, 
                 episode_length=1000, device="auto"):
        super().__init__(env)
        
        # 配置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.visual_encoder_name = visual_encoder
        self.visual_reward_weight = visual_reward_weight
        self.episode_length = episode_length
        
        # 初始化视觉编码器
        self.visual_encoder = None
        self.visual_processor = None
        self._init_visual_encoder(visual_encoder)
        
        # 初始化奖励生成器
        if RewardGenerator is not None:
            self.reward_generator = RewardGenerator()
            self.visual_reward_enabled = True
        else:
            print("Warning: RewardGenerator not available, using dummy visual rewards")
            self.reward_generator = None
            self.visual_reward_enabled = False
        
        # 统计信息
        self.episode_env_reward = 0.0
        self.episode_visual_reward = 0.0
        self.episode_steps = 0
        
        print(f"VisualRewardWrapper initialized:")
        print(f"  Visual encoder: {visual_encoder}")
        print(f"  Reward weight: {visual_reward_weight}")
        print(f"  Device: {self.device}")
        print(f"  Visual reward enabled: {self.visual_reward_enabled}")
    
    def _init_visual_encoder(self, encoder):
        """初始化视觉编码器"""
        try:
            if encoder == "ResNet":
                # 使用ResNet，不需要下载外部模型
                self.visual_encoder = torchvision.models.resnet50(
                    weights=torchvision.models.ResNet50_Weights.DEFAULT
                )
                self.visual_encoder = torch.nn.Sequential(*list(self.visual_encoder.children())[:-1])
                self.visual_processor = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            elif encoder == "CLIP":
                # 尝试加载CLIP，如果失败则回退到ResNet
                try:
                    self.visual_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
                    self.visual_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    print("CLIP visual encoder loaded successfully.")
                except Exception as e:
                    print(f"Warning: Failed to load CLIP, falling back to ResNet: {e}")
                    return self._init_visual_encoder("ResNet")
            elif encoder == "ViT":
                try:
                    self.visual_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
                    self.visual_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
                except Exception as e:
                    print(f"Warning: Failed to load ViT, falling back to ResNet: {e}")
                    return self._init_visual_encoder("ResNet")
            else:  # DINO
                try:
                    self.visual_encoder = AutoModel.from_pretrained("facebook/dino-vitb16")
                    self.visual_processor = AutoImageProcessor.from_pretrained("facebook/dino-vitb16")
                except Exception as e:
                    print(f"Warning: Failed to load DINO, falling back to ResNet: {e}")
                    return self._init_visual_encoder("ResNet")
            
            self.visual_encoder.to(self.device)
            self.visual_encoder.eval()
            
        except Exception as e:
            print(f"Warning: Failed to initialize visual encoder {encoder}: {e}")
            self.visual_encoder = None
            self.visual_processor = None
    
    def _extract_visual_features(self, frame):
        """从帧中提取视觉特征"""
        if self.visual_encoder is None:
            return np.zeros(512)  # 返回零向量作为fallback
        
        try:
            with torch.no_grad():
                # 确保输入是PIL图像
                if isinstance(frame, np.ndarray):
                    # 处理不同的输入格式
                    if frame.dtype == np.float32 or frame.dtype == np.float64:
                        frame = (frame * 255).astype(np.uint8)
                    if len(frame.shape) == 3 and frame.shape[0] in [1, 3]:
                        # CHW格式转换为HWC
                        frame = np.transpose(frame, (1, 2, 0))
                    if frame.shape[-1] == 1:
                        # 灰度图转RGB
                        frame = np.repeat(frame, 3, axis=-1)
                    frame = Image.fromarray(frame)
                
                if self.visual_encoder_name == "ResNet":
                    input_tensor = self.visual_processor(frame).unsqueeze(0).to(self.device)
                    feature = self.visual_encoder(input_tensor).squeeze().cpu().numpy()
                elif self.visual_encoder_name == "ViT":
                    inputs = self.visual_processor(images=frame, return_tensors="pt").to(self.device)
                    outputs = self.visual_encoder(**inputs)
                    feature = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                elif self.visual_encoder_name == "CLIP":
                    inputs = self.visual_processor(images=frame, return_tensors="pt").to(self.device)
                    outputs = self.visual_encoder(**inputs)
                    feature = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                else:  # DINO
                    inputs = self.visual_processor(images=frame, return_tensors="pt").to(self.device)
                    outputs = self.visual_encoder(**inputs)
                    feature = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
                return feature
                
        except Exception as e:
            print(f"Warning: Visual feature extraction failed: {e}")
            return np.zeros(512)  # 返回零向量作为fallback
    
    def _compute_visual_reward(self, frame):
        """计算视觉奖励"""
        if not self.visual_reward_enabled or self.reward_generator is None:
            return 0.0
        
        try:
            # 提取视觉特征
            features = self._extract_visual_features(frame)
            
            # 生成奖励
            visual_reward = self.reward_generator.generate_reward(features)
            return visual_reward
            
        except Exception as e:
            print(f"Warning: Visual reward computation failed: {e}")
            return 0.0
    
    def reset(self, **kwargs):
        """重置环境并重置视觉奖励生成器"""
        obs = self.env.reset(**kwargs)
        
        # 重置奖励生成器
        if self.visual_reward_enabled and RewardGenerator is not None:
            self.reward_generator = RewardGenerator()
        
        # 重置统计信息
        self.episode_env_reward = 0.0
        self.episode_visual_reward = 0.0
        self.episode_steps = 0
        
        return obs
    
    def step(self, action):
        """环境步进，整合视觉奖励"""
        obs, env_reward, done, info = self.env.step(action)
        
        # 计算视觉奖励
        visual_reward = 0.0
        if self.visual_reward_enabled:
            # 获取原始帧用于视觉奖励计算
            raw_frame = None
            
            # 尝试从环境获取原始帧
            if hasattr(self.env, 'get_raw_frame'):
                raw_frame = self.env.get_raw_frame()
            elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'get_raw_frame'):
                raw_frame = self.env.unwrapped.get_raw_frame()
            
            # 如果无法获取原始帧，从观察中提取
            if raw_frame is None:
                if isinstance(obs, dict) and 'image' in obs:
                    raw_frame = obs['image']
                else:
                    raw_frame = obs
            
            if raw_frame is not None:
                visual_reward = self._compute_visual_reward(raw_frame)
        
        # 整合奖励
        total_reward = env_reward + self.visual_reward_weight * visual_reward
        
        # 更新统计信息
        self.episode_env_reward += env_reward
        self.episode_visual_reward += visual_reward
        self.episode_steps += 1
        
        # 在info中添加奖励分解信息
        if isinstance(info, dict):
            info['env_reward'] = env_reward
            info['visual_reward'] = visual_reward
            info['visual_reward_weight'] = self.visual_reward_weight
            info['total_reward'] = total_reward
        
        # 在episode结束时记录统计信息
        if done:
            if isinstance(info, dict):
                info['episode_env_reward'] = self.episode_env_reward
                info['episode_visual_reward'] = self.episode_visual_reward
                info['episode_steps'] = self.episode_steps
                
        return obs, total_reward, done, info


class DummyVisualRewardWrapper(gym.Wrapper):
    """
    虚拟视觉奖励包装器
    当无法使用真实视觉奖励时的fallback
    """
    
    def __init__(self, env, visual_reward_weight=0.1):
        super().__init__(env)
        self.visual_reward_weight = visual_reward_weight
        self.episode_env_reward = 0.0
        self.episode_steps = 0
        print(f"DummyVisualRewardWrapper initialized (visual reward disabled)")
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.episode_env_reward = 0.0
        self.episode_steps = 0
        return obs
    
    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)
        
        # 不添加视觉奖励，但保持接口一致
        total_reward = env_reward
        
        self.episode_env_reward += env_reward
        self.episode_steps += 1
        
        # 在info中添加奖励信息
        if isinstance(info, dict):
            info['env_reward'] = env_reward
            info['visual_reward'] = 0.0
            info['visual_reward_weight'] = 0.0
            info['total_reward'] = total_reward
        
        if done:
            if isinstance(info, dict):
                info['episode_env_reward'] = self.episode_env_reward
                info['episode_visual_reward'] = 0.0
                info['episode_steps'] = self.episode_steps
                
        return obs, total_reward, done, info
