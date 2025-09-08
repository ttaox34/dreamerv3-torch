import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml
import gymnasium

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd

# Import SubprocVecEnv and related wrappers for retro parallel environments
try:
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
    from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not available, parallel retro environments will be disabled")
    STABLE_BASELINES3_AVAILABLE = False

# Import retro for environment creation
try:
    import retro
    RETRO_AVAILABLE = True
except ImportError:
    print("Warning: retro not available")
    RETRO_AVAILABLE = False


class SingleVecEnvWrapper:
    """将单环境的SubprocVecEnv适配为Dreamer期望的环境接口"""
    def __init__(self, vec_env):
        self.vec_env = vec_env
        self.id = 0  # Dreamer需要这个属性
        
    @property
    def action_space(self):
        return self.vec_env.action_space
    
    @property 
    def observation_space(self):
        # Return a dict observation space that matches our observation format
        from gymnasium.spaces import Dict, Box
        original_obs_space = self.vec_env.observation_space
        
        return Dict({
            "image": original_obs_space,
            "is_first": Box(low=0, high=1, shape=(), dtype=bool),
            "is_last": Box(low=0, high=1, shape=(), dtype=bool),
            "is_terminal": Box(low=0, high=1, shape=(), dtype=bool),
        })
    
    def reset(self):
        """Reset the environment - return a callable as Dreamer expects"""
        def _reset():
            obs = self.vec_env.reset()
            # Return first (and only) environment's observation
            raw_obs = obs[0] if isinstance(obs, (list, np.ndarray)) and len(obs) > 0 else obs
            
            # Convert to Dreamer's expected dict format
            return {
                "image": raw_obs,
                "is_first": True,
                "is_last": False,
                "is_terminal": False,
            }
        return _reset
    
    def step(self, action):
        """Step the environment - return a callable as Dreamer expects"""
        def _step():
            # Step with single action
            actions = [action]
            result = self.vec_env.step(actions)
            
            # Handle both 4-value and 5-value returns
            if len(result) == 4:
                # Old gym API: (obs, reward, done, info)
                obs, rewards, dones, infos = result
                # Convert done to terminated and truncated
                terminated = dones
                truncated = [False] * len(dones)  # Assume no truncation for old API
            else:
                # New gymnasium API: (obs, reward, terminated, truncated, info)
                obs, rewards, terminated, truncated, infos = result
            
            # Extract scalar results for single environment
            raw_obs = obs[0] if isinstance(obs, (list, np.ndarray)) else obs
            reward = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards
            term = terminated[0] if isinstance(terminated, (list, np.ndarray)) else terminated
            trunc = truncated[0] if isinstance(truncated, (list, np.ndarray)) else truncated
            info = infos[0] if isinstance(infos, (list, np.ndarray)) else infos
            
            # Ensure info is always a dict for Dreamer compatibility
            if not isinstance(info, dict):
                info = {}  # Default empty dict if info is not a dict
            
            # Convert to Dreamer's expected format
            obs_dict = {
                "image": raw_obs,
                "is_first": False,
                "is_last": term or trunc,
                "is_terminal": term,
            }
            
            return obs_dict, reward, term, trunc, info
        return _step
    
    def close(self):
        """Close the vectorized environment"""
        if hasattr(self.vec_env, 'close'):
            self.vec_env.close()


class VecEnvWrapper:
    """将SubprocVecEnv适配为Dreamer期望的环境接口"""
    def __init__(self, vec_env, env_id):
        self.vec_env = vec_env
        self.env_id = env_id
        self.id = env_id  # Dreamer需要这个属性
        self.num_envs = vec_env.num_envs
        self._last_obs = None
        self._done = False
        
    @property
    def action_space(self):
        return self.vec_env.action_space
    
    @property 
    def observation_space(self):
        return self.vec_env.observation_space
    
    def reset(self):
        """Reset the specific environment - return a callable as Dreamer expects"""
        def _reset():
            # For vectorized env, we reset all and take our slice
            obs = self.vec_env.reset()
            self._last_obs = obs[self.env_id]
            self._done = False
            return self._last_obs
        return _reset
    
    def step(self, action):
        """Step the specific environment - return a callable as Dreamer expects"""
        def _step():
            # Create action array for all environments
            actions = [np.zeros_like(action)] * self.num_envs  # Default actions for other envs
            actions[self.env_id] = action
            
            # Step the vectorized environment
            obs, rewards, dones, infos = self.vec_env.step(actions)
            
            # Extract results for our specific environment
            self._last_obs = obs[self.env_id]
            reward = rewards[self.env_id]
            done = dones[self.env_id]
            info = infos[self.env_id] if isinstance(infos, list) else {}
            self._done = done
            
            return self._last_obs, reward, done, info
        return _step
        
        # Fill other actions with random/dummy actions
        for i in range(self.num_envs):
            if i != self.env_id:
                actions[i] = self.vec_env.action_space.sample()
        
        # Step all environments
        obs, rewards, dones, infos = self.vec_env.step(actions)
        
        # Return only our environment's results
        return obs[self.env_id], rewards[self.env_id], dones[self.env_id], infos[self.env_id]
    
    def close(self):
        """Close the vectorized environment"""
        if hasattr(self.vec_env, 'close'):
            self.vec_env.close()


to_np = lambda x: x.detach().cpu().numpy()


class CustomWarpFrame(gymnasium.ObservationWrapper):
    """Custom WarpFrame that uses configurable image size instead of hardcoded 84x84"""
    def __init__(self, env, width=64, height=64, grayscale=True, dict_space_key=None):
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            obs = self._process_frame(obs)
        else:
            obs = obs.copy()
            obs[self._key] = self._process_frame(obs[self._key])
        return obs

    def _process_frame(self, frame):
        import cv2
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        if self._grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


def make_retro_game_env(game_name, config, max_episode_steps=4500):
    """为单个retro游戏创建环境（PPO风格，用于多进程）"""
    def _init():
        if not RETRO_AVAILABLE:
            raise ImportError("retro not available")
            
        # Create retro environment similar to PPO
        state = retro.State.DEFAULT
        env = retro.make(game_name, state, render_mode='rgb_array')
        
        # Add frame skip wrapper
        env = StochasticFrameSkip(env, n=4, stickprob=0.25)
        
        # Add time limit
        if max_episode_steps is not None:
            from gymnasium.wrappers import TimeLimit
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
        
        # Apply retro-specific wrappers (similar to PPO)
        # Custom resize wrapper using config size instead of hardcoded 84x84
        env = CustomWarpFrame(env, width=config.size[0], height=config.size[1])
        env = ClipRewardEnv(env)
        
        # For retro environments, handle MultiBinary action space FIRST
        if hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0:
            # This is a MultiBinary action space, convert to discrete
            from gymnasium.spaces import Discrete
            # Create a discrete action space that maps to all possible button combinations
            # For simplicity, we'll use a subset of actions
            n_actions = min(env.action_space.shape[0], 12)  # Use first N buttons
            env = MultiBinaryToDiscreteWrapper(env, n_actions)
        
        # Apply Dreamer-specific wrappers in correct order
        env = wrappers.OneHotAction(env)  # Convert discrete to one-hot first
        env = wrappers.TimeLimit(env, max_episode_steps)
        env = wrappers.SelectAction(env, key="action")  # Then extract from dict
        env = wrappers.UUID(env)
        
        return env
    return _init


class MultiBinaryToDiscreteWrapper(gymnasium.Wrapper):
    """Convert MultiBinary action space to Discrete for Dreamer compatibility"""
    def __init__(self, env, n_actions):
        super().__init__(env)
        self.n_actions = n_actions
        # Save the original action space
        self.original_action_space = env.action_space
        from gymnasium.spaces import Discrete
        self.action_space = Discrete(n_actions)
        
    def step(self, action):
        # Convert discrete action to MultiBinary
        if hasattr(self.original_action_space, 'shape') and len(self.original_action_space.shape) > 0:
            multi_action = np.zeros(self.original_action_space.shape[0], dtype=np.int8)
        else:
            multi_action = np.zeros(self.n_actions, dtype=np.int8)
        if action < len(multi_action):
            multi_action[action] = 1
        return self.env.step(multi_action)


class StochasticFrameSkip(gymnasium.Wrapper):
    """Frame skip wrapper from PPO (simplified for Dreamer)"""
    def __init__(self, env, n, stickprob):
        super().__init__(env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        
    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)
    
    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            if self.curac is None:
                self.curac = ac
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            elif i == 1:
                self.curac = ac
            
            ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def close(self):
        return self.env.close()
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
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
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
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
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
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
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id, game_override=None):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == "retro":
        import envs.stable_retro as stable_retro

        # Use game_override if provided, otherwise use task
        game_name = game_override if game_override is not None else task
        
        env = stable_retro.StableRetro(
            game=game_name,
            action_repeat=config.action_repeat,
            size=config.size,
            grayscale=config.grayscale,
            seed=config.seed + id,
        )
        
        # Add visual reward wrapper if enabled
        if getattr(config, 'visual_reward', False) and getattr(config, 'visual_reward_weight', 0.0) > 0.0:
            try:
                from envs.visual_reward_wrapper import VisualRewardWrapper
                
                visual_device = getattr(config, 'visual_device', 'auto')
                if visual_device == 'auto':
                    visual_device = config.device if hasattr(config, 'device') else 'auto'
                
                env = VisualRewardWrapper(
                    env,
                    visual_encoder=getattr(config, 'visual_encoder', 'CLIP'),
                    visual_reward_weight=getattr(config, 'visual_reward_weight', 0.1),
                    episode_length=getattr(config, 'visual_episode_length', 1000),
                    device=visual_device
                )
                print(f"Visual reward enabled for retro environment (weight: {config.visual_reward_weight})")
            except Exception as e:
                print(f"Warning: Failed to enable visual reward: {e}")
                print("Continuing without visual reward...")
        
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id, game=None: make_env(config, mode, id, game_override=game)
    
    # Check if multi-game retro training is enabled
    suite, task = config.task.split("_", 1)
    if suite == "retro" and getattr(config, 'retro_parallel_games', False):
        # Multi-game parallel retro training using PPO's approach
        if not STABLE_BASELINES3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for parallel retro environments")
        
        games_list = getattr(config, 'retro_games_list', [])
        if not games_list:
            # If no games list provided, use the task as single game
            games_list = [task]
            print(f"No games list provided, using single game: {task}")
        
        print(f"Creating parallel retro environments for {len(games_list)} games: {games_list}")
        print("Using SubprocVecEnv approach (PPO style)")
        
        # Create separate environment for each game
        train_envs = []
        eval_envs = []
        
        for game in games_list:
            # Create environment function for this game
            env_fn = make_retro_game_env(
                game, 
                config,
                max_episode_steps=getattr(config, 'time_limit', 108000) * getattr(config, 'action_repeat', 4)
            )
            
            # Create single-environment SubprocVecEnv for this game
            single_vec_env = SubprocVecEnv([env_fn])
            # Add frame stacking
            vec_env_with_stack = VecFrameStack(single_vec_env, n_stack=4)
            # Add image transpose
            final_vec_env = VecTransposeImage(vec_env_with_stack)
            
            # Wrap to provide Dreamer-compatible interface
            train_env = SingleVecEnvWrapper(final_vec_env)
            eval_env = train_env  # Reuse for evaluation
            
            train_envs.append(train_env)
            eval_envs.append(eval_env)
        
        print(f"✅ Created {len(train_envs)} training environments using SubprocVecEnv")
        
        # Set config.envs to match the number of games
        config.envs = len(games_list)
        
    elif suite == "retro":
        # Single retro game (legacy behavior)
        print("Warning: Using single environment for retro game")
        shared_env = make("train", 0)
        if config.parallel:
            shared_env = Parallel(lambda: shared_env, "process")
        else:
            shared_env = Damy(shared_env)
        train_envs = [shared_env]
        eval_envs = [shared_env]
    else:
        # Create environments normally for non-retro suites
        train_envs = []
        eval_envs = []
        for i in range(config.envs):
            if config.parallel:
                # Use closure to capture current value of i
                train_env = Parallel(lambda i=i: make("train", i), "process")
                eval_env = Parallel(lambda i=i: make("eval", i), "process")
            else:
                train_env = Damy(make("train", i))
                eval_env = Damy(make("eval", i))
            train_envs.append(train_env)
            eval_envs.append(eval_env)
    
    # Handle action space for multi-game scenarios
    if suite == "retro" and getattr(config, 'retro_parallel_games', False):
        # For multi-game retro, we need to handle potentially different action spaces
        # Get action space from first environment as reference
        acts = train_envs[0].action_space
        
        # Check if all games have the same action space
        action_spaces = [env.action_space for env in train_envs]
        all_same = all(
            space.n == acts.n if hasattr(space, 'n') else space.shape == acts.shape 
            for space in action_spaces
        )
        
        if not all_same:
            print("Warning: Different action spaces detected across games:")
            games_list = getattr(config, 'retro_games_list', [])
            for i, space in enumerate(action_spaces):
                game_name = games_list[i] if i < len(games_list) else f"game_{i}"
                if hasattr(space, 'n'):
                    print(f"  {game_name}: {space.n} discrete actions")
                else:
                    print(f"  {game_name}: {space.shape} continuous actions")
            
            # Use the maximum action space size
            if hasattr(acts, 'n'):
                max_actions = max(space.n for space in action_spaces)
                print(f"Using maximum action space size: {max_actions}")
                config.num_actions = max_actions
            else:
                print("Warning: Continuous action spaces may not be compatible")
                config.num_actions = acts.shape[0]
        else:
            print(f"All games have consistent action space: {acts}")
            config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    else:
        acts = train_envs[0].action_space
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    
    print("Action Space", acts)
    print(f"Number of actions: {config.num_actions}")

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
