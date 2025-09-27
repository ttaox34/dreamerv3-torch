# dreamerv3-torch Project Context

## Project Overview

Dreamerv3-torch is a PyTorch implementation of the DreamerV3 algorithm from the paper "Mastering Diverse Domains through World Models". DreamerV3 is a scalable reinforcement learning algorithm that outperforms previous approaches across various domains with fixed hyperparameters.

The project implements a model-based reinforcement learning approach that learns world models to plan and perform actions in various environments. It supports multiple domains including:
- DeepMind Control Suite (DMC Proprio and Vision)
- Atari 100k benchmark
- Crafter environment
- Minecraft (with MineRL)
- Memory Maze
- Retro games

## Architecture

The core architecture consists of:

1. **World Model (`WorldModel`)**: Learns to represent the environment's dynamics using:
   - Encoder: Processes observations (images/states)
   - RSSM (Recurrent State Space Model): Maintains latent state representation
   - Decoder: Reconstructs observations from latent states
   - Reward head: Predicts rewards
   - Continue head: Predicts episode continuation

2. **ImagBehavior (`ImagBehavior`)**: The policy and value function that operates in the learned latent space:
   - Actor: Decides actions based on latent states
   - Value: Estimates state values

3. **Exploration (`exploration.py`)**: Includes random and plan2explore strategies for exploration

4. **Networks (`networks.py`)**: Implements neural network components including:
   - RSSM (Recurrent State Space Model)
   - MultiEncoder/MultiDecoder for handling different observation types
   - Convolutional networks for image processing
   - MLPs for various tasks

## Key Components

- `dreamer.py`: Main training loop and Dreamer class
- `models.py`: World model and behavior model implementations
- `networks.py`: Neural network architectures
- `tools.py`: Utility functions, logging, data processing
- `exploration.py`: Exploration strategies
- `configs.yaml`: Configuration parameters for different environments
- `parallel.py`: Parallel environment execution

## Building and Running

### Dependencies
Install dependencies using:
```
pip install -r requirements.txt
```

### Running Training
Example for DMC Vision:
```
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```

### Monitoring Results
```
tensorboard --logdir ./logdir
```

### Docker Usage
Build and run with Docker:
```
docker build -f Dockerfile -t img . && \
docker run -it --rm --gpus all -v $PWD:/workspace/img \
  sh xvfb_run.sh python3 dreamer.py \
  --configs dmc_vision --task dmc_walker_walk \
  --logdir "./logdir/dmc_walker_walk"
```

## Environment Support

The project supports multiple environments through different wrappers:
- DMC (DeepMind Control): For continuous control tasks
- Atari: For discrete action games
- Crafter: For survival tasks
- Minecraft: For complex 3D environments
- Memory Maze: For long-term memory evaluation

## Key Configuration Options

- `task`: Specifies the environment (e.g., `dmc_walker_walk`)
- `steps`: Total training steps
- `envs`: Number of parallel environments
- `batch_size`, `batch_length`: Training batch parameters
- Various hyperparameters for world model, actor, critic, and exploration

## Development Conventions

- PyTorch-based implementation with support for mixed precision training
- Modular design with separate modules for world models, behaviors, and utilities
- Extensive configuration system using YAML files
- Logging with TensorBoard integration
- Support for both single and parallel environment execution
- Deterministic training options for reproducibility

## File Structure
- `dreamer.py`: Main training script and Dreamer class
- `models.py`: Core model implementations
- `networks.py`: Neural network architectures
- `tools.py`: Utility functions and helpers
- `exploration.py`: Exploration strategies
- `configs.yaml`: Configuration files
- `envs/`: Environment-specific code
- `parallel.py`: Parallel execution utilities