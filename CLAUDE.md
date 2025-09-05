# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of DreamerV3, a scalable world model-based reinforcement learning algorithm that outperforms previous approaches across various domains with fixed hyperparameters. The implementation supports multiple environments including DMC (DeepMind Control), Atari, Crafter, Minecraft, and Memory Maze.

## Key Commands

### Training
```bash
# Basic training on DMC Vision
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk

# Monitor training with TensorBoard
tensorboard --logdir ./logdir
```

### Dependencies
```bash
# Install Python 3.11 dependencies
pip install -r requirements.txt
```

### Docker Support
```bash
# Build and run with Docker
docker build -f Dockerfile -t img .
docker run -it --rm --gpus all -v $PWD:/workspace img sh xvfb_run.sh python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir "./logdir/dmc_walker_walk"
```

## Architecture Overview

### Core Components

1. **dreamer.py**: Main training loop and agent implementation
   - Contains the `Dreamer` class that orchestrates the entire training process
   - Manages world model, actor-critic, and exploration strategies
   - Handles training, evaluation, and logging

2. **models.py**: World model and behavior implementations
   - `WorldModel`: Core world model with encoder, RSSM dynamics, and decoder heads
   - `ImagBehavior`: Actor-critic for imagined trajectories
   - `RewardEMA`: Exponential moving average for reward normalization

3. **networks.py**: Neural network architectures
   - `RSSM`: Recurrent State Space Model core dynamics
   - Various encoders, decoders, and MLP components
   - CNN and MLP network building blocks

4. **tools.py**: Utility functions and helpers
   - `Logger`: TensorBoard logging and metrics tracking
   - `Optimizer`: Custom optimizer wrapper with AMP support
   - Utility functions like symlog/symexp transformations

5. **exploration.py**: Exploration strategies
   - Random exploration
   - Plan2Explore for intrinsic motivation

### Environment Support (envs/)

- **dmc.py**: DeepMind Control Suite environments
- **atari.py**: Atari 100k benchmark support
- **crafter.py**: Crafter survival environment
- **minecraft_*.py**: Minecraft environments (MineRL interface)
- **memorymaze.py**: Memory Maze for long-term memory evaluation
- **stable_retro.py**: Retro gaming environments
- **visual_reward_wrapper.py**: Visual reward computation from pixels

### Configuration System

- **configs.yaml**: Centralized configuration with hyperparameters
- Supports different environment configurations (dmc_vision, dmc_proprio, atari, etc.)
- All training parameters are configurable via command line and YAML

## Key Features

- **World Model**: Uses RSSM (Recurrent State Space Model) for learning latent dynamics
- **Visual Input**: Supports both pixel-based and state-based observations
- **Multi-Environment**: Unified API across different environment types
- **Mixed Precision**: Supports FP16 training with AMP
- **Compilation**: Uses `torch.compile()` for performance optimization
- **Exploration**: Multiple exploration strategies including Plan2Explore
- **Reward Normalization**: Built-in reward normalization with EMA

## Development Notes

- The codebase uses PyTorch 2.4.1 with CUDA support
- Environments require specific setup scripts in `envs/setup_scripts/`
- Training logs are saved to TensorBoard format
- The implementation follows the original DreamerV3 paper architecture
- All models are device-aware and support GPU acceleration

## Environment Setup

For specific environments:
- Atari: Check `envs/setup_scripts/` for Atari setup
- Minecraft: Requires MineRL installation and setup
- DMC: Works out of the box with dm_control package
- Visual rewards: Optional visual reward computation from pixel observations