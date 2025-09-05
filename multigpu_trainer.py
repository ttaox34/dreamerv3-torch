import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import functools
import os
import pathlib
import numpy as np
import time
from collections import defaultdict
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy


def make_dataset(episodes, config):
    """创建数据集"""
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


class OptimizerManager:
    """分布式优化器管理器"""
    
    def __init__(self, model, config, device):
        self.config = config
        self.device = device
        
        # 收集所有需要优化的参数
        self.params = []
        self.optimizers = {}
        self.scalers = {}
        
        # 递归收集所有优化器
        self._collect_optimizers(model)
        
        # 为每个优化器创建梯度缩放器
        for name, optimizer in self.optimizers.items():
            self.scalers[name] = torch.cuda.amp.GradScaler(
                enabled=(config.precision == 16)
            )
    
    def _collect_optimizers(self, module):
        """递归收集模块中的所有优化器"""
        if hasattr(module, '_optims'):
            for name, optimizer in module._optims.items():
                if optimizer not in self.optimizers.values():
                    self.optimizers[f"{module.__class__.__name__}_{name}"] = optimizer
                    self.params.extend([p for p in optimizer.param_groups[0]['params'] if p.requires_grad])
        
        # 递归检查子模块
        for child in module.children():
            self._collect_optimizers(child)
    
    def zero_grad(self):
        """清零所有优化器的梯度"""
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
    
    def step(self):
        """执行所有优化器的步骤"""
        for optimizer in self.optimizers.values():
            optimizer.step()
    
    def scale_loss(self, loss, optimizer_name=None):
        """缩放损失用于混合精度训练"""
        if optimizer_name and optimizer_name in self.scalers:
            return self.scalers[optimizer_name].scale(loss)
        else:
            # 如果没有指定优化器，使用第一个缩放器
            return list(self.scalers.values())[0].scale(loss)
    
    def unscale_grads(self, optimizer_name=None):
        """反缩放梯度"""
        if optimizer_name and optimizer_name in self.scalers:
            self.scalers[optimizer_name].unscale_(self.optimizers[optimizer_name])
        else:
            # 反缩放所有优化器
            for scaler, optimizer in zip(self.scalers.values(), self.optimizers.values()):
                scaler.unscale_(optimizer)
    
    def clip_grads(self, max_norm):
        """裁剪梯度"""
        for optimizer in self.optimizers.values():
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm)


class MultiGPUTrainer:
    """增强版多GPU并行训练管理器"""
    
    def __init__(self, config, logger, world_size=None, rank=None):
        self.config = config
        self.logger = logger
        self.world_size = world_size or torch.cuda.device_count()
        self.rank = rank or 0
        
        # 设备配置
        self.device = f'cuda:{rank}' if rank is not None else config.device
        torch.cuda.set_device(self.device)
        
        # 多GPU环境分配
        self.envs_per_gpu = max(1, config.envs // self.world_size)
        self.total_envs = self.envs_per_gpu * self.world_size
        
        # 训练统计
        self.step = 0
        self.episode_count = 0
        self.metrics = defaultdict(list)
        
        # 混合精度训练
        self.use_amp = config.precision == 16
        
        # 空日志记录器（用于非主进程）
        self.empty_logger = self._create_empty_logger()
        
        print(f"MultiGPU Trainer initialized:")
        print(f"  World size: {self.world_size}")
        print(f"  Rank: {self.rank}")
        print(f"  Device: {self.device}")
        print(f"  Envs per GPU: {self.envs_per_gpu}")
        print(f"  Total envs: {self.total_envs}")
        print(f"  Mixed precision: {self.use_amp}")
    
    def _create_empty_logger(self):
        """创建空日志记录器用于非主进程"""
        class EmptyLogger:
            def __init__(self):
                self.step = 0
            
            def scalar(self, name, value):
                pass
            
            def write(self, **kwargs):
                pass
        
        return EmptyLogger()
        
    def setup_distributed(self):
        """设置分布式训练"""
        if self.world_size > 1:
            # 初始化进程组
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.multi_gpu_backend,
                    world_size=self.world_size,
                    rank=self.rank
                )
            
            # 设置分布式采样器种子
            torch.manual_seed(self.config.seed + self.rank)
            
            print(f"Distributed training initialized on rank {self.rank}")
    
    def cleanup_distributed(self):
        """清理分布式训练"""
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def create_environments(self, config, mode):
        """为当前GPU创建环境"""
        print(f"Creating {self.envs_per_gpu} {mode} environments on {self.device}")
        
        make_env = lambda mode, id: self._make_single_env(config, mode, id)
        envs = []
        
        for i in range(self.envs_per_gpu):
            env_id = self.rank * self.envs_per_gpu + i
            env = make_env(mode, env_id)
            envs.append(env)
        
        # 使用并行包装器
        if config.parallel:
            envs = [Parallel(env, "process") for env in envs]
        else:
            envs = [Damy(env) for env in envs]
        
        return envs
    
    def _make_single_env(self, config, mode, id):
        """创建单个环境"""
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
                task, config.action_repeat, config.size,
                gray=config.grayscale, noops=config.noops,
                lives=config.lives, sticky=config.stickey,
                actions=config.actions, resize=config.resize,
                seed=config.seed + id
            )
            env = wrappers.OneHotAction(env)
        elif suite == "retro":
            # 支持多游戏retro训练
            if hasattr(config, 'retro_games') and config.retro_games:
                import envs.multigame_retro as multigame_retro
                
                games = config.retro_games
                if isinstance(games, str):
                    games = [game.strip() for game in games.split(',')]
                
                env = multigame_retro.MultiGameRetroEnv(
                    games=games,
                    action_repeat=config.action_repeat,
                    size=config.size,
                    grayscale=config.grayscale,
                    seed=config.seed + id,
                    visual_reward=getattr(config, 'visual_reward', False),
                    visual_reward_weight=getattr(config, 'visual_reward_weight', 0.1),
                    visual_encoder=getattr(config, 'visual_encoder', 'CLIP'),
                    visual_episode_length=getattr(config, 'visual_episode_length', 1000),
                    visual_device=getattr(config, 'visual_device', 'auto')
                )
                
                # 配置游戏切换策略
                if hasattr(config, 'game_switch_strategy'):
                    game_switch_freq = getattr(config, 'game_switch_freq', 1000)
                    env.set_game_switch_strategy(config.game_switch_strategy, game_switch_freq)
            else:
                import envs.stable_retro as stable_retro
                env = stable_retro.StableRetro(
                    game=task,
                    action_repeat=config.action_repeat,
                    size=config.size,
                    grayscale=config.grayscale,
                    seed=config.seed + id,
                )
                env = wrappers.OneHotAction(env)
        else:
            raise NotImplementedError(f"Environment suite {suite} not supported")
        
        # 通用包装器
        env = wrappers.TimeLimit(env, config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)
        
        return env
    
    def create_model(self, obs_space, act_space, config, dataset):
        """创建模型（根据是否多游戏模式选择）"""
        suite, task = config.task.split("_", 1)
        
        if suite == "retro" and hasattr(config, 'retro_games') and config.retro_games:
            from multigame_dreamer import MultiGameDreamer
            model = MultiGameDreamer(
                obs_space, act_space, config, self.logger or self.empty_logger, dataset
            ).to(self.device)
            print(f"Created Multi-Game Dreamer model on {self.device}")
        else:
            import models
            model = models.Dreamer(
                obs_space, act_space, config, self.logger or self.empty_logger, dataset
            ).to(self.device)
            print(f"Created Single-Game Dreamer model on {self.device}")
        
        # 分布式包装
        if self.world_size > 1:
            model = DDP(
                model, 
                device_ids=[self.rank], 
                output_device=self.rank,
                find_unused_parameters=getattr(config, 'multi_gpu_find_unused_parameters', True)
            )
            print(f"Model wrapped with DDP on {self.device}")
        
        return model
    
    def create_optimizer_manager(self, model):
        """创建优化器管理器"""
        return OptimizerManager(model, self.config, self.device)
    
    def train_step(self, model, optimizer_manager, data):
        """执行单个训练步骤"""
        optimizer_manager.zero_grad()
        
        # 前向传播
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            metrics = model._train(data)
        
        # 计算损失（假设metrics包含loss）
        if isinstance(metrics, dict) and 'loss' in metrics:
            loss = metrics['loss']
        else:
            # 如果没有直接返回loss，从metrics中计算
            loss = 0.0
            for key, value in metrics.items():
                if 'loss' in key.lower():
                    loss += value
        
        # 反向传播
        if self.use_amp:
            # 混合精度训练
            scaled_loss = optimizer_manager.scale_loss(loss)
            scaled_loss.backward()
            
            # 反缩放梯度
            optimizer_manager.unscale_grads()
            
            # 梯度裁剪
            optimizer_manager.clip_grads(self.config.grad_clip)
            
            # 优化器步骤
            optimizer_manager.step()
        else:
            # 标准训练
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            optimizer_manager.step()
        
        # 同步梯度（分布式训练）
        if self.world_size > 1:
            self.synchronize_gradients(model)
        
        return metrics
    
    def synchronize_gradients(self, model):
        """同步梯度（仅在分布式模式下）"""
        if self.world_size > 1 and dist.is_initialized():
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.world_size
    
    def synchronize_metrics(self, metrics):
        """同步指标"""
        if self.world_size <= 1:
            return metrics
        
        synchronized_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # 转换为tensor并同步
                tensor = torch.tensor([value], device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                synchronized_metrics[key] = tensor.item() / self.world_size
            else:
                synchronized_metrics[key] = value
        
        return synchronized_metrics
    
    def collect_data_parallel(self, agent, envs, episodes, directory, limit=None, steps=None):
        """并行收集数据"""
        if self.world_size == 1:
            # 单GPU模式，使用原有的simulate函数
            return tools.simulate(
                agent, envs, episodes, directory, self.logger,
                limit=limit, steps=steps
            )
        else:
            # 多GPU模式，每个GPU独立收集数据
            return self._simulate_distributed(agent, envs, episodes, directory, limit, steps)
    
    def _simulate_distributed(self, agent, envs, episodes, directory, limit, steps):
        """分布式数据收集"""
        print(f"Rank {self.rank} starting data collection")
        
        # 确保缓存是字典
        cache = episodes if isinstance(episodes, dict) else {}
        
        # 每个GPU独立收集数据
        local_episodes = tools.simulate(
            agent, envs, cache, directory / f"rank_{self.rank}", 
            self.logger or self.empty_logger, limit=limit // self.world_size, 
            steps=steps
        )
        
        # 等待所有GPU完成数据收集
        if dist.is_initialized():
            dist.barrier()
        
        return local_episodes
    
    def should_log(self, step):
        """判断是否应该记录日志"""
        log_interval = self.config.log_every // self.config.action_repeat
        return step % log_interval == 0
    
    def should_evaluate(self, step):
        """判断是否应该评估"""
        eval_interval = self.config.eval_every // self.config.action_repeat
        return step % eval_interval == 0
    
    def log_metrics(self, metrics, step):
        """记录指标"""
        if self.rank == 0:  # 只在主进程记录
            # 同步所有GPU的指标
            synchronized_metrics = self.synchronize_metrics(metrics)
            
            # 记录到日志
            for name, value in synchronized_metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.scalar(name, value)
            
            # 写入日志
            self.logger.write(fps=True, step=step * self.config.action_repeat)


def run_distributed_training(rank, world_size, config, logger):
    """运行分布式训练"""
    try:
        # 设置分布式训练
        trainer = MultiGPUTrainer(config, logger, world_size, rank)
        trainer.setup_distributed()
        
        # 创建日志目录和数据目录
        logdir = pathlib.Path(config.logdir).expanduser()
        if rank == 0:  # 只在主进程创建目录
            logdir.mkdir(parents=True, exist_ok=True)
            
        config.traindir = config.traindir or logdir / "train_eps"
        config.evaldir = config.evaldir or logdir / "eval_eps"
        
        if rank == 0:  # 只在主进程创建目录
            config.traindir.mkdir(parents=True, exist_ok=True)
            config.evaldir.mkdir(parents=True, exist_ok=True)
        
        # 等待所有进程同步
        if world_size > 1:
            torch.distributed.barrier()
        
        # 创建日志记录器（只在主进程）
        if rank == 0:
            from tools import Logger
            logger = Logger(logdir, 0)
        else:
            # 为非主进程创建一个空的logger
            class EmptyLogger:
                def __init__(self):
                    self.step = 0
                
                def scalar(self, name, value):
                    pass
                
                def write(self, **kwargs):
                    pass
            
            logger = EmptyLogger()
        
        # 创建环境
        train_envs = trainer.create_environments(config, "train")
        eval_envs = trainer.create_environments(config, "eval")
        
        # 获取动作空间
        acts = train_envs[0].action_space
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        
        # 多游戏模式特殊处理
        suite, task = config.task.split("_", 1)
        if suite == "retro" and hasattr(config, 'retro_games') and config.retro_games:
            if hasattr(train_envs[0], 'max_num_actions'):
                config.num_actions = train_envs[0].max_num_actions
                config.game_action_spaces = train_envs[0].num_actions_list
        
        # 创建数据集（处理空目录情况）
        train_eps = []
        if config.traindir and pathlib.Path(config.traindir).exists():
            try:
                train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Could not load training episodes from {config.traindir}: {e}")
        
        eval_eps = []
        if config.evaldir and pathlib.Path(config.evaldir).exists():
            try:
                eval_eps = tools.load_episodes(config.evaldir, limit=1)
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Could not load evaluation episodes from {config.evaldir}: {e}")
        
        if rank == 0 and not train_eps:
            print("Starting with empty dataset, will collect data during prefill.")
        
        # 创建数据集（处理空数据集情况）
        if len(train_eps) == 0:
            if rank == 0:
                print("Creating empty dataset generator for initial training")
            # 创建一个空的数据生成器
            def empty_generator():
                while True:
                    # 返回空的模拟数据
                    batch_size = config.batch_size
                    batch_length = config.batch_length
                    yield {
                        'image': torch.zeros(batch_size, batch_length, 3, 64, 64),
                        'action': torch.zeros(batch_size, batch_length, config.num_actions),
                        'reward': torch.zeros(batch_size, batch_length),
                        'discount': torch.zeros(batch_size, batch_length),
                        'is_first': torch.zeros(batch_size, batch_length),
                        'is_terminal': torch.zeros(batch_size, batch_length),
                        'is_last': torch.zeros(batch_size, batch_length)
                    }
            
            train_dataset = tools.from_generator(empty_generator(), config.batch_size)
        else:
            train_dataset = make_dataset(train_eps, config)
        
        if len(eval_eps) == 0:
            eval_dataset = None
        else:
            eval_dataset = tools.make_dataset(eval_eps, config)
        
        # 创建模型和优化器管理器
        model = trainer.create_model(
            train_envs[0].observation_space,
            train_envs[0].action_space,
            config,
            train_dataset
        )
        
        optimizer_manager = trainer.create_optimizer_manager(model)
        
        # 训练循环
        state = None
        step = 0
        
        # 检查是否需要预填充数据
        if not train_eps and config.prefill > 0:
            if rank == 0:
                print(f"Starting prefill with {config.prefill} steps")
            
            # 创建随机策略进行预填充
            if hasattr(acts, "n"):
                random_actor = tools.OneHotDist(
                    torch.zeros(config.num_actions).repeat(trainer.envs_per_gpu, 1)
                )
            else:
                random_actor = torch.distributions.independent.Independent(
                    torch.distributions.uniform.Uniform(
                        torch.tensor(acts.low).repeat(trainer.envs_per_gpu, 1),
                        torch.tensor(acts.high).repeat(trainer.envs_per_gpu, 1),
                    ),
                    1,
                )
            
            def random_agent(o, d, s):
                action = random_actor.sample()
                logprob = random_actor.log_prob(action)
                return {"action": action, "logprob": logprob}, None
            
            # 收集预填充数据
            state = tools.simulate(
                random_agent,
                train_envs,
                {},  # 空的缓存字典
                config.traindir,
                trainer.logger or trainer.empty_logger,
                limit=config.dataset_size,
                steps=config.prefill,
            )
            
            if rank == 0:
                print("Prefill completed, creating new dataset")
            
            # 重新创建数据集
            train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
            train_dataset = make_dataset(train_eps, config)
        
        while step < config.steps:
            # 数据收集
            if rank == 0:
                print(f"Starting data collection at step {step}")
            
            # 并行收集数据
            state = trainer.collect_data_parallel(
                model, train_envs, None, config.traindir,
                limit=config.dataset_size, steps=config.eval_every
            )
            
            # 训练
            if rank == 0:
                print(f"Starting training at step {step}")
            
            # 执行训练步骤
            train_metrics = defaultdict(list)
            for _ in range(config.train_ratio):
                try:
                    data = next(train_dataset)
                    metrics = trainer.train_step(model, optimizer_manager, data)
                    
                    # 收集指标
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            train_metrics[key].append(value)
                except StopIteration:
                    # 数据集耗尽，重新创建
                    if rank == 0:
                        print("Dataset exhausted, reloading...")
                    train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
                    train_dataset = make_dataset(train_eps, config)
                    continue
            
            # 计算平均指标
            if train_metrics:
                avg_metrics = {key: np.mean(values) for key, values in train_metrics.items()}
            else:
                avg_metrics = {}
            
            # 记录指标
            if trainer.should_log(step) and avg_metrics:
                trainer.log_metrics(avg_metrics, step)
            
            # 评估
            if trainer.should_evaluate(step) and config.eval_episode_num > 0:
                if rank == 0:
                    print("Starting evaluation")
                    eval_policy = functools.partial(model, training=False)
                    tools.simulate(
                        eval_policy, eval_envs, {}, config.evaldir,
                        logger, is_eval=True, episodes=config.eval_episode_num
                    )
                    
                    # 保存检查点
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer_manager.optimizers,
                        'step': step,
                        'config': config
                    }
                    torch.save(checkpoint, config.logdir / "latest_multi_gpu.pt")
                    print(f"Saved checkpoint at step {step}")
            
            step += config.eval_every // config.action_repeat
        
        print(f"Training completed on rank {rank}")
    
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        # 确保在异常时也能清理资源
        try:
            cleanup_distributed_training(locals().get('trainer'), locals().get('train_envs', []), locals().get('eval_envs', []))
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")
        raise
    finally:
        # 清理资源
        cleanup_distributed_training(trainer, train_envs, eval_envs)


def cleanup_distributed_training(trainer, train_envs, eval_envs):
    """清理分布式训练资源"""
    if trainer is not None:
        print(f"Cleaning up distributed training for rank {trainer.rank}")
        
        # 清理分布式设置
        try:
            trainer.cleanup_distributed()
        except Exception as e:
            print(f"Error cleaning up distributed training: {e}")
    
    # 关闭环境
    for env_list in [train_envs or [], eval_envs or []]:
        for env in env_list:
            try:
                env.close()
            except Exception as e:
                print(f"Error closing environment: {e}")


def launch_multi_gpu_training(config, logger):
    """启动多GPU训练"""
    world_size = torch.cuda.device_count()
    
    if world_size <= 1:
        print("Only one GPU available, using single GPU training")
        return None
    
    print(f"Launching multi-GPU training on {world_size} GPUs")
    
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 使用mp.spawn启动多进程
    mp.spawn(
        run_distributed_training,
        args=(world_size, config, logger),
        nprocs=world_size,
        join=True
    )