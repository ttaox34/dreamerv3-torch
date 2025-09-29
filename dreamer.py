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


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset, game_name, game_action_spaces):
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
        self._game_name = game_name  # Add game name to identify current game
        # Add game_action_spaces to config for multi-game RSSM support
        config.game_action_spaces = game_action_spaces
        
        # Initialize reward manager if reward mode is specified
        self._reward_manager = None
        if hasattr(config, 'reward_mode') and config.reward_mode in ["L1", "L2", "L3"]:
            try:
                from reward_manager import RewardManager
                self._reward_manager = RewardManager(
                    reward_mode=config.reward_mode,
                    games_to_train=list(game_action_spaces.keys()),
                    visual_encoder=getattr(config, 'visual_encoder', 'CLIP'),
                    device=getattr(config, 'device', 'cuda')
                )
                print(f"Reward manager initialized for mode: {config.reward_mode}")
            except Exception as e:
                print(f"Warning: Failed to initialize reward manager: {e}")
        
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.MultiGameImagBehavior(config, self._wm, game_action_spaces)
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
                    openl = self._wm.video_pred(next(self._dataset), game_name=self._game_name)
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        # Debug information
        # print(f"DEBUG: _policy called for game {self._game_name}")
        if state is None:
            # print("DEBUG: State is None")
            latent = action = None
        else:
            latent, action = state
            # print(f"DEBUG: State exists, action shape: {action.shape if action is not None else 'None'}")
            
            # Make sure action dimension matches the current game
            if action is not None:
                expected_action_size = self._config.num_actions
                actual_action_size = action.shape[-1]
                # print(f"DEBUG: Expected action size: {expected_action_size}, Actual action size: {actual_action_size}")
                if actual_action_size != expected_action_size:
                    # If action size doesn't match current game, reset to None
                    # This should trigger initialization in obs_step
                    # print(f"DEBUG: Action size mismatch, resetting state")
                    latent = action = None

        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        # print(f"DEBUG: Calling obs_step with game {self._game_name}")
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"], game_name=self._game_name)
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actors[self._game_name](feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            # For exploration, we need to use the current game's actor
            actor = self._task_behavior.actors[self._game_name](feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actors[self._game_name](feat)
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
        post, context, mets = self._wm._train(data, game_name=self._game_name)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward, self._game_name)[-1])
        if self._config.expl_behavior != "greedy":
            if hasattr(self._expl_behavior, 'train') and hasattr(self._expl_behavior, '_behavior'):
                # For Plan2Explore, we need to pass the task behavior
                mets = self._expl_behavior.train(start, context, data, self._task_behavior)[-1]
            else:
                # For other exploration methods
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


def make_env(config, mode, id):
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

        env = stable_retro.StableRetro(
            game=task,
            action_repeat=config.action_repeat,
            size=config.size,
            grayscale=config.grayscale,
            seed=config.seed + id,
        )
        
        # Add reward mode wrapper if enabled
        if hasattr(config, 'reward_mode') and config.reward_mode in ["L1", "L2", "L3"]:
            try:
                from reward_manager import RewardManager, SharedVisualEncoder
                
                # Get game name from task
                game_name = task.replace("retro_", "")  # Remove "retro_" prefix to get game name
                
                # Create reward manager with appropriate settings
                reward_manager = RewardManager(
                    reward_mode=config.reward_mode,
                    games_to_train=[game_name],  # Single game context
                    visual_encoder=getattr(config, 'visual_encoder', 'CLIP'),
                    device=getattr(config, 'device', 'cuda')
                )
                
                from envs.wrappers import RewardModeWrapper
                env = RewardModeWrapper(env, reward_manager, game_name)
                print(f"Reward mode {config.reward_mode} enabled for retro environment: {game_name}")
            except Exception as e:
                print(f"Warning: Failed to enable reward mode {config.reward_mode}: {e}")
                print("Continuing with original reward...")
        
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
    
    # Handle multiple games for retro environments
    suite, task = config.task.split("_", 1)
    
    if suite == "retro":
        # Parse multiple games from task config - format like "retro_game1,game2,game3"
        games = task.split(",") if "," in task else [task]
        
        # Create action space mapping for all games
        game_action_spaces = {}
        for game in games:
            temp_env = make_env(type('Config', (), {
                'task': f"retro_{game}",
                'action_repeat': config.action_repeat,
                'size': config.size,
                'grayscale': getattr(config, 'grayscale', False),
                'seed': config.seed,
                'time_limit': getattr(config, 'time_limit', 1000)  # Add missing attribute
            })(), "train", 0)
            game_action_spaces[game] = temp_env.action_space.n if hasattr(temp_env.action_space, "n") else temp_env.action_space.shape[0]
            temp_env.close()
        
        print(f"Multi-game setup: {list(game_action_spaces.keys())}")
        print(f"Action spaces: {game_action_spaces}")
        
        # Initialize with first game to get observation space
        first_game_config = type('Config', (), {
            'task': f"retro_{games[0]}",
            'action_repeat': config.action_repeat,
            'size': config.size,
            'grayscale': getattr(config, 'grayscale', False),
            'seed': config.seed,
            'time_limit': getattr(config, 'time_limit', 1000)  # Add missing attribute
        })()
        
        # Get observation space from first game
        temp_env = make_env(first_game_config, "train", 0)
        obs_space = temp_env.observation_space
        temp_env.close()
        
        # Create the agent with all game action spaces
        # Set num_actions to the maximum action space across all games for consistent architecture
        max_action_size = max(game_action_spaces.values())
        config.num_actions = max_action_size
        agent = Dreamer(
            obs_space,
            None,  # Will be set dynamically per game
            config,
            logger,
            None,  # Will be set dynamically per game
            games[0],  # Start with first game
            game_action_spaces
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)
        
        if (logdir / "latest.pt").exists():
            checkpoint = torch.load(logdir / "latest.pt")
            tools.load_agent_state_dict(agent, checkpoint["agent_state_dict"])
            tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
            agent._should_pretrain._once = False

        # Training loop for multiple retro games
        current_game_idx = 0
        while agent._step < config.steps + config.eval_every:
            current_game = games[current_game_idx]
            print(f"Switching to game: {current_game}")
            
            # Change the agent's current game
            agent._game_name = current_game
            
            # Create environment for current game
            current_config = type('Config', (), {
                'task': f"retro_{current_game}",
                'action_repeat': config.action_repeat,
                'size': config.size,
                'grayscale': getattr(config, 'grayscale', False),
                'seed': config.seed + current_game_idx,
                'time_limit': getattr(config, 'time_limit', 1000)  # Add missing attribute
            })()
            
            # Create the environment and datasets for current game
            if config.offline_traindir:
                directory = config.offline_traindir.format(**vars(config))
            else:
                directory = config.traindir / current_game  # Separate directory per game
            directory.mkdir(parents=True, exist_ok=True)
            train_eps = tools.load_episodes(directory, limit=config.dataset_size)
            
            if config.offline_evaldir:
                eval_directory = config.offline_evaldir.format(**vars(config))
            else:
                eval_directory = config.evaldir / current_game  # Separate directory per game
            eval_directory.mkdir(parents=True, exist_ok=True)
            eval_eps = tools.load_episodes(eval_directory, limit=1)
            
            # Create environments for current game
            shared_env = make_env(current_config, "train", 0)
            train_envs = [shared_env]
            eval_envs = [shared_env]  # Reuse the same environment
            
            if config.parallel:
                train_envs = [Parallel(env, "process") for env in train_envs]
                eval_envs = [Parallel(env, "process") for env in eval_envs]
            else:
                train_envs = [Damy(env) for env in train_envs]
                eval_envs = [Damy(env) for env in eval_envs]
                
            acts = train_envs[0].action_space
            print(f"Action Space for {current_game}: {acts}")
            config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
            
            # Set the dataset for the current game
            train_dataset = make_dataset(train_eps, config)
            agent._dataset = train_dataset  # Update the agent's dataset
            
            # Update exploration behavior with the current game's task behavior
            reward = lambda f, s, a: agent._wm.heads["reward"](f).mean()
            plan2explore_fn = lambda: expl.Plan2Explore(config, agent._wm, reward)
            agent._expl_behavior = dict(
                greedy=lambda: agent._task_behavior,
                random=lambda: expl.Random(config, acts),
                plan2explore=plan2explore_fn,
            )[config.expl_behavior]().to(config.device)

            state = None
            if not config.offline_traindir:
                prefill = max(0, config.prefill - count_steps(config.traindir / current_game))
                print(f"Prefill dataset for {current_game} ({prefill} steps).")
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
                    config.traindir / current_game,
                    logger,
                    limit=config.dataset_size,
                    steps=prefill,
                )
                logger.step += prefill * config.action_repeat
                print(f"Logger: ({logger.step} steps).")

            print(f"Simulate agent for {current_game}.")
            eval_dataset = make_dataset(eval_eps, config)
            
            logger.write()
            if config.eval_episode_num > 0:
                print(f"Start evaluation for {current_game}.")
                eval_policy = functools.partial(agent, training=False)
                tools.simulate(
                    eval_policy,
                    eval_envs,
                    eval_eps,
                    config.evaldir / current_game,
                    logger,
                    is_eval=True,
                    episodes=config.eval_episode_num,
                )
                if config.video_pred_log:
                    video_pred = agent._wm.video_pred(next(eval_dataset), game_name=agent._game_name)
                    logger.video(f"eval_openl_{current_game}", to_np(video_pred))
            
            print(f"Start training for {current_game}.")
            state = tools.simulate(
                agent,
                train_envs,
                train_eps,
                config.traindir / current_game,
                logger,
                limit=config.dataset_size,
                steps=config.eval_every,
                state=state,
            )
            
            # Switch to next game for next iteration
            current_game_idx = (current_game_idx + 1) % len(games)
            
            # Save model checkpoint
            items_to_save = {
                "agent_state_dict": tools.save_agent_state_dict(agent),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                "game_action_spaces": game_action_spaces,  # Save game action spaces info
                "step": logger.step  # Save current step
            }
            torch.save(items_to_save, logdir / "latest.pt")
            
            # Close current environment
            for env in train_envs + eval_envs:
                try:
                    env.close()
                except Exception:
                    pass
    else:
        # Original code for non-retro environments
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
        make = lambda mode, id: make_env(config, mode, id)
        
        # Special handling for retro environments due to single emulator limitation
        if suite == "retro":
            # For retro, create only one environment and reuse for both train and eval
            print("Warning: Using single environment for both train and eval due to retro emulator limitation")
            shared_env = make("train", 0)
            train_envs = [shared_env]
            eval_envs = [shared_env]  # Reuse the same environment
        else:
            train_envs = [make("train", i) for i in range(config.envs)]
            eval_envs = [make("eval", i) for i in range(config.envs)]
        
        if config.parallel:
            train_envs = [Parallel(env, "process") for env in train_envs]
            eval_envs = [Parallel(env, "process") for env in eval_envs]
        else:
            train_envs = [Damy(env) for env in train_envs]
            eval_envs = [Damy(env) for env in eval_envs]
        acts = train_envs[0].action_space
        print("Action Space", acts)
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

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
            task,  # game name
            {task: config.num_actions}  # game action spaces (only one game in this case)
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)
        if (logdir / "latest.pt").exists():
            checkpoint = torch.load(logdir / "latest.pt")
            tools.load_agent_state_dict(agent, checkpoint["agent_state_dict"])
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
                    video_pred = agent._wm.video_pred(next(eval_dataset), game_name=agent._game_name)
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
