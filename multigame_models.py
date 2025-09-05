import copy
import torch
from torch import nn

import networks
import tools


to_np = lambda x: x.detach().cpu().numpy()

# 导入需要的模块
import models


class MultiHeadImagBehavior(nn.Module):
    """多动作头的ImagBehavior，支持多个不同动作空间的游戏"""
    
    def __init__(self, config, world_model, game_action_spaces):
        """
        初始化多动作头ImagBehavior
        
        Args:
            config: 配置对象
            world_model: 世界模型
            game_action_spaces: 游戏动作空间字典 {game_name: num_actions}
        """
        super(MultiHeadImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self.game_action_spaces = game_action_spaces
        self.num_games = len(game_action_spaces)
        
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        
        # 创建共享的特征提取层
        self.shared_layers = nn.Sequential()
        for i in range(config.actor.get("shared_layers", 2)):
            self.shared_layers.add_module(
                f"shared_linear{i}", nn.Linear(feat_size, config.units)
            )
            if config.norm:
                self.shared_layers.add_module(
                    f"shared_norm{i}", nn.LayerNorm(config.units, eps=1e-03)
                )
            act = getattr(torch.nn, config.act)
            self.shared_layers.add_module(f"shared_act{i}", act())
        
        # 为每个游戏创建独立的动作头
        self.actor_heads = nn.ModuleDict()
        self.actor_opts = nn.ModuleDict()
        
        # 游戏ID到动作头的映射
        self.game_to_idx = {game: idx for idx, game in enumerate(game_action_spaces.keys())}
        
        for game_name, num_actions in game_action_spaces.items():
            self.actor_heads[game_name] = networks.MLP(
                config.units,  # 输入是共享层的输出
                (num_actions,),
                config.actor["layers"],
                config.units,
                config.act,
                config.norm,
                config.actor["dist"],
                config.actor["std"],
                config.actor["min_std"],
                config.actor["max_std"],
                absmax=1.0,
                temp=config.actor["temp"],
                unimix_ratio=config.actor["unimix_ratio"],
                outscale=config.actor["outscale"],
                name=f"Actor_{game_name}",
            )
        
        # 共享的value网络
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        
        # 创建优化器
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        
        # 为每个动作头创建优化器
        self._actor_opts = nn.ModuleDict()
        for game_name in game_action_spaces.keys():
            opt = tools.Optimizer(
                f"actor_{game_name}",
                self.actor_heads[game_name].parameters(),
                config.actor["lr"],
                config.actor["eps"],
                config.actor["grad_clip"],
                **kw,
            )
            self._actor_opts[game_name] = opt
            print(
                f"Optimizer actor_opt_{game_name} has "
                f"{sum(param.numel() for param in self.actor_heads[game_name].parameters())} variables."
            )
        
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        
        if self._config.reward_EMA:
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            from models import RewardEMA
            self.reward_ema = RewardEMA(device=self._config.device)
    
    def get_actor_for_game(self, game_name_or_id):
        """获取指定游戏的动作网络"""
        if isinstance(game_name_or_id, int):
            # 如果是ID，转换为游戏名称
            game_names = list(self.game_action_spaces.keys())
            if 0 <= game_name_or_id < len(game_names):
                game_name = game_names[game_name_or_id]
            else:
                raise ValueError(f"Game ID {game_name_or_id} out of range")
        else:
            game_name = game_name_or_id
        
        if game_name not in self.actor_heads:
            raise ValueError(f"Game {game_name} not found in actor heads")
        
        return self.actor_heads[game_name]
    
    def forward(self, features, game_ids=None):
        """
        前向传播
        
        Args:
            features: 输入特征 [batch_size, ..., feat_size]
            game_ids: 游戏ID [batch_size, ...]
        
        Returns:
            动作分布字典
        """
        batch_size = features.shape[0]
        
        # 共享特征提取
        shared_features = self.shared_layers(features)
        
        # 如果没有提供game_ids，使用默认的游戏ID
        if game_ids is None:
            game_ids = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        
        # 为每个样本获取对应的动作分布
        action_dists = []
        
        unique_games = torch.unique(game_ids)
        for game_id in unique_games:
            game_id_int = game_id.item()
            game_names = list(self.game_action_spaces.keys())
            if game_id_int < len(game_names):
                game_name = game_names[game_id_int]
                mask = (game_ids == game_id)
                
                if mask.any():
                    actor = self.actor_heads[game_name]
                    game_features = shared_features[mask]
                    action_dist = actor(game_features)
                    action_dists.append((mask, action_dist))
        
        return action_dists
    
    def get_action(self, features, game_ids=None, sample=True):
        """获取动作"""
        action_dists = self.forward(features, game_ids)
        
        batch_size = features.shape[0]
        actions = torch.zeros(batch_size, device=features.device)
        logprobs = torch.zeros(batch_size, device=features.device)
        
        for mask, action_dist in action_dists:
            if sample:
                action = action_dist.sample()
            else:
                action = action_dist.mode()
            
            logprob = action_dist.log_prob(action)
            
            actions[mask] = action
            logprobs[mask] = logprob
        
        return actions, logprobs
    
    def _train(self, start, objective):
        """训练方法"""
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # 获取游戏ID（假设从start中获取）
                if hasattr(start, 'game_ids'):
                    game_ids = start.game_ids
                else:
                    # 如果没有游戏ID，使用默认值
                    game_ids = torch.zeros(start['feat'].shape[0], dtype=torch.long, device=start['feat'].device)
                
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self._actor, self._config.imag_horizon, game_ids
                )
                
                reward = objective(imag_feat, imag_state, imag_action)
                target, weights = self._compute_target(
                    reward, imag_feat, imag_state, imag_action
                )
                
                actor_loss, actor_metrics = self._compute_actor_loss(
                    imag_feat, imag_action, target, weights, game_ids
                )
                value_loss, value_metrics = self._compute_value_loss(
                    imag_feat, target, weights
                )
                
                metrics.update(actor_metrics)
                metrics.update(value_metrics)
        
        # 更新优化器
        for game_name, opt in self._actor_opts.items():
            opt(torch.mean(actor_loss.get(game_name, torch.tensor(0.0))))
        
        self._value_opt(torch.mean(value_loss))
        
        return imag_feat, imag_state, imag_action, metrics
    
    def _imagine(self, start, horizon, game_ids):
        """想象轨迹"""
        raise NotImplementedError("Imagination method needs to be implemented")
    
    def _compute_target(self, reward, imag_feat, imag_state, imag_action):
        """计算目标"""
        raise NotImplementedError("Target computation needs to be implemented")
    
    def _compute_actor_loss(self, imag_feat, imag_action, target, weights, game_ids):
        """计算actor损失"""
        raise NotImplementedError("Actor loss computation needs to be implemented")
    
    def _compute_value_loss(self, imag_feat, target, weights):
        """计算value损失"""
        raise NotImplementedError("Value loss computation needs to be implemented")


class MultiGameImagBehavior(nn.Module):
    """简化的多游戏ImagBehavior实现"""
    
    def __init__(self, config, world_model, game_action_spaces):
        """
        初始化多游戏ImagBehavior
        
        Args:
            config: 配置对象
            world_model: 世界模型
            game_action_spaces: 游戏动作空间字典 {game_name: num_actions}
        """
        super(MultiGameImagBehavior, self).__init__()
        self._config = config
        self._world_model = world_model
        self.game_action_spaces = game_action_spaces
        self.num_games = len(game_action_spaces)
        
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        
        # 创建共享的value网络
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        
        # 创建多动作头
        self.actor_heads = nn.ModuleDict()
        game_names = list(game_action_spaces.keys())
        
        for i, (game_name, num_actions) in enumerate(game_action_spaces.items()):
            self.actor_heads[game_name] = networks.MLP(
                feat_size,  # 直接使用feat_size，不经过共享层
                (num_actions,),
                config.actor["layers"],
                config.units,
                config.act,
                config.norm,
                config.actor["dist"],
                config.actor["std"],
                config.actor["min_std"],
                config.actor["max_std"],
                absmax=1.0,
                temp=config.actor["temp"],
                unimix_ratio=config.actor["unimix_ratio"],
                outscale=config.actor["outscale"],
                name=f"Actor_{game_name}",
            )
        
        # 游戏ID到索引的映射
        self.game_to_idx = {game: idx for idx, game in enumerate(game_names)}
        
        # 慢速目标
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        
        # 优化器存储为普通字典
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=True if config.precision == 16 else False)
        
        self._actor_opts = {}
        for game_name in game_action_spaces.keys():
            opt = tools.Optimizer(
                f"actor_{game_name}",
                self.actor_heads[game_name].parameters(),
                config.actor["lr"],
                config.actor["eps"],
                config.actor["grad_clip"],
                **kw,
            )
            self._actor_opts[game_name] = opt
        
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        
        print(f"Created MultiGameImagBehavior with {len(game_action_spaces)} games")
        for game_name, num_actions in game_action_spaces.items():
            print(f"  - {game_name}: {num_actions} actions")
    
    def get_actor_for_game(self, game_id):
        """获取指定游戏的动作网络"""
        game_names = list(self.game_action_spaces.keys())
        if 0 <= game_id < len(game_names):
            game_name = game_names[game_id]
            return self.actor_heads[game_name]
        else:
            raise ValueError(f"Game ID {game_id} out of range")
    
    def get_action(self, feat, game_ids=None, sample=True):
        """获取动作"""
        batch_size = feat.shape[0]
        
        if game_ids is None:
            game_ids = torch.zeros(batch_size, dtype=torch.long, device=feat.device)
        
        actions = torch.zeros(batch_size, device=feat.device, dtype=torch.long)
        logprobs = torch.zeros(batch_size, device=feat.device)
        
        # 为每个游戏ID获取动作
        unique_games = torch.unique(game_ids)
        for game_id in unique_games:
            mask = (game_ids == game_id)
            if mask.any():
                actor = self.get_actor_for_game(game_id.item())
                game_feat = feat[mask]
                
                if sample:
                    action_dist = actor(game_feat)
                    action = action_dist.sample()
                    logprob = action_dist.log_prob(action)
                else:
                    action_dist = actor(game_feat)
                    action = action_dist.mode()
                    logprob = action_dist.log_prob(action)
                
                # 对于onehot分布，将one-hot向量转换为标量动作
                if action.dim() > 1:
                    action = torch.argmax(action, dim=-1)
                
                # 确保logprob是标量张量
                if logprob.dim() > 0:
                    logprob = logprob.squeeze(-1)
                
                actions[mask] = action
                logprobs[mask] = logprob
        
        return actions, logprobs
    
    def _train(self, start, reward_fn):
        """训练方法"""
        self._update_slow_target()
        metrics = {}
        
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(True if self._config.precision == 16 else False):
                # 简化的训练实现
                feat = self._world_model.dynamics.get_feat(start)
                
                # 获取游戏ID
                if hasattr(start, 'game_ids'):
                    game_ids = start.game_ids
                else:
                    game_ids = torch.zeros(feat.shape[0], dtype=torch.long, device=feat.device)
                
                # 获取动作和logprob
                actions, logprobs = self.get_action(feat, game_ids, sample=True)
                
                # 计算value
                values = self.value(feat).mode()
                
                # 计算奖励
                rewards = reward_fn(feat, start, actions)
                
                # 简化的actor和value损失
                actor_loss = -logprobs.mean()
                value_loss = torch.nn.functional.mse_loss(values, rewards.detach())
                
                metrics.update({
                    'actor_loss': to_np(actor_loss),
                    'value_loss': to_np(value_loss),
                    'reward_mean': to_np(rewards.mean()),
                    'value_mean': to_np(values.mean()),
                })
        
        # 更新优化器
        for game_name, opt in self._actor_opts.items():
            opt(actor_loss)
        
        self._value_opt(value_loss)
        
        return start, actions, metrics
    
    def _update_slow_target(self):
        """更新慢速目标"""
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1