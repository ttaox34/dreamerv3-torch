import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class MultiGameDreamer(nn.Module):
    """支持多游戏的Dreamer代理"""
    
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(MultiGameDreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        
        # 创建世界模型
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        
        # 创建多游戏行为模型
        if hasattr(config, 'game_action_spaces') and config.game_action_spaces:
            # 多游戏模式
            from multigame_models import MultiGameImagBehavior
            game_names = []
            if hasattr(config, 'retro_games') and config.retro_games:
                if isinstance(config.retro_games, str):
                    game_names = [game.strip() for game in config.retro_games.split(',')]
                else:
                    game_names = config.retro_games
            
            # 创建游戏动作空间字典
            game_action_spaces = {}
            for i, num_actions in enumerate(config.game_action_spaces):
                if i < len(game_names):
                    game_action_spaces[game_names[i]] = num_actions
                else:
                    game_action_spaces[f"game_{i}"] = num_actions
            
            self._task_behavior = MultiGameImagBehavior(config, self._wm, game_action_spaces)
            self._is_multigame = True
            print(f"Created multi-game behavior model with games: {list(game_action_spaces.keys())}")
        else:
            # 单游戏模式
            self._task_behavior = models.ImagBehavior(config, self._wm)
            self._is_multigame = False
        
        # 编译优化
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)
    
    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
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
        
        # 获取游戏ID（如果存在）
        game_ids = obs.get("game_id", None)
        
        if not training:
            if self._is_multigame:
                # 多游戏模式：使用对应的动作头
                action, logprob = self._task_behavior.get_action(
                    feat.unsqueeze(0), 
                    game_ids.unsqueeze(0) if game_ids is not None else None,
                    sample=False
                )
                action = action.squeeze(0)
                logprob = logprob.squeeze(0)
            else:
                actor = self._task_behavior.actor(feat)
                action = actor.mode()
                logprob = actor.log_prob(action)
        elif self._should_expl(self._step):
            if self._is_multigame:
                action, logprob = self._task_behavior.get_action(
                    feat.unsqueeze(0), 
                    game_ids.unsqueeze(0) if game_ids is not None else None,
                    sample=True
                )
                action = action.squeeze(0)
                logprob = logprob.squeeze(0)
            else:
                actor = self._expl_behavior.actor(feat)
                action = actor.sample()
                logprob = actor.log_prob(action)
        else:
            if self._is_multigame:
                action, logprob = self._task_behavior.get_action(
                    feat.unsqueeze(0), 
                    game_ids.unsqueeze(0) if game_ids is not None else None,
                    sample=True
                )
                action = action.squeeze(0)
                logprob = logprob.squeeze(0)
            else:
                actor = self._task_behavior.actor(feat)
                action = actor.sample()
                logprob = actor.log_prob(action)
        
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        
        if self._is_multigame:
            # 多游戏模式：传递游戏ID
            game_ids = data.get("game_id", None)
            if game_ids is not None:
                start['game_ids'] = game_ids
            
            metrics.update(self._task_behavior._train(start, reward)[-1])
        else:
            metrics.update(self._task_behavior._train(start, reward)[-1])
        
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)