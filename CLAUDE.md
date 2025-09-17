# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of DreamerV3, a scalable world model algorithm that outperforms previous approaches across various domains with fixed hyperparameters. The implementation supports multiple environments including DMC (DeepMind Control), Atari, Crafter, Minecraft, and Memory Maze.

## Common Commands

### Environment Setup
```bash
# Install dependencies (requires Python 3.11)
pip install -r requirements.txt

# Setup specific environments
# Check envs/setup_scripts/ for Atari and Minecraft setup instructions
```

### Training
```bash
# Basic training on DMC Vision
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk

# Monitor results with TensorBoard
tensorboard --logdir ./logdir
```

### Docker Usage
```bash
# Build and run with Docker
docker build -f Dockerfile -t img .
docker run -it --rm --gpus all -v $PWD:/workspace img sh xvfb_run.sh python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir "./logdir/dmc_walker_walk"
```

### X11 Virtual Display (for headless environments)
```bash
# Use the provided xvfb wrapper script
./xvfb_run.sh python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```

## Architecture Overview

### Core Components

1. **dreamer.py**: Main training loop and Dreamer agent implementation
   - Contains the `Dreamer` class that orchestrates the world model and behavior learning
   - Handles training, evaluation, and logging logic

2. **models.py**: World model and agent behavior implementations
   - `WorldModel`: Combines encoder, RSSM dynamics, decoder, and reward models
   - `ImagBehavior`: Agent behavior learned through imagination rollouts
   - `ActorCritic`: Standard actor-critic architecture for policy and value learning

3. **networks.py**: Neural network building blocks
   - `RSSM`: Recurrent State Space Model core dynamics
   - `MultiEncoder`: Handles multiple observation modalities
   - `MultiDecoder`: Reconstructs observations from latent states
   - Various CNN and MLP network components

4. **tools.py**: Utility functions and training helpers
   - `Logger`: TensorBoard logging and metrics tracking
   - `Counter`: Step counting utilities
   - Various helper classes for training management

5. **exploration.py**: Exploration strategies
   - Implements different exploration mechanisms for training

### Environment Support

The `envs/` directory contains wrappers and implementations for various environments:
- **dmc.py**: DeepMind Control Suite
- **atari.py**: Atari games with preprocessing
- **crafter.py**: Crafter survival environment
- **minecraft/**: Minecraft environments (multiple variants)
- **memorymaze.py**: Memory maze for long-term memory evaluation
- **stable_retro.py**: Stable-retro game environments
- **visual_reward_wrapper.py**: Visual reward computation for sparse reward environments

### Configuration System

- **configs.yaml**: Centralized hyperparameter configuration
- Uses YAML format with hierarchical structure
- Supports environment-specific configurations through `--configs` flag
- Key sections: general settings, environment config, model architecture, training parameters

### Key Architectural Features

1. **World Model Architecture**:
   - Encoder: Processes observations to latent representations
   - RSSM: Recurrent State Space Model for dynamics learning
   - Decoder: Reconstructs observations from latent states
   - Reward model: Predicts rewards from latent states
   - Continuity model: Predicts episode termination

2. **Training Process**:
   - Collects real environment interactions
   - Trains world model on collected data
   - Performs imagination rollouts in latent space
   - Trains actor-critic on imagined trajectories
   - Uses compiled models (`torch.compile`) for performance

3. **Multi-modality Support**:
   - Handles both image and state observations
   - Supports discrete and continuous action spaces
   - Configurable input preprocessing (grayscale, resizing)

## Development Notes

### Device Management
- Uses CUDA by default (`cuda:0`)
- Supports automatic mixed precision (AMP) for memory efficiency
- Models can be compiled with `torch.compile` for better performance

### Environment Setup Requirements
- Requires Python 3.11
- Uses PyTorch 2.4.1 with CUDA support
- Needs MuJoCo for physics simulation
- Requires X11 display or virtual display for visualization

### Logging and Monitoring
- Uses TensorBoard for experiment tracking
- Logs training metrics, evaluation results, and video predictions
- Supports configurable logging intervals

### Data Flow
1. Environment interactions → Replay buffer
2. World model training on batched data
3. Imagination rollouts in latent space
4. Actor-critic training on imagined trajectories
5. Policy execution in real environment