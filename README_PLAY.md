# DreamerV3 Game Player

This tool allows you to play games using trained DreamerV3 checkpoints and save the gameplay as video files.

## Features

- Load trained DreamerV3 checkpoints
- Play games in evaluation mode
- Save gameplay as MP4 video
- Support for all DreamerV3 environments (DMC, Atari, Minecraft, etc.)
- Configurable gameplay parameters

## Installation

Make sure you have the required dependencies:

```bash
pip install opencv-python
```

## Usage

### Basic Usage

```bash
# Play a game with a trained checkpoint
python3 play_game.py --checkpoint /path/to/checkpoint.pt --task dmc_walker_walk --output gameplay.mp4

# Play Atari game
python3 play_game.py --checkpoint /path/to/checkpoint.pt --task atari_breakout --output atari_gameplay.mp4

# Play with custom configuration
python3 play_game.py --checkpoint /path/to/checkpoint.pt --task minecraft_creative --configs dmc_vision --output minecraft_gameplay.mp4
```

### Parameters

- `--checkpoint`: Path to the trained checkpoint file (.pt)
- `--task`: Task to play (e.g., `dmc_walker_walk`, `atari_breakout`, `minecraft_creative`)
- `--output`: Output video path (default: `gameplay.mp4`)
- `--max_steps`: Maximum steps to play (default: 10000)
- `--configs`: Configuration names to use (default: `defaults`)
- `--logdir`: Log directory (default: `./logdir`)

### Supported Tasks

#### DeepMind Control (DMC)
```bash
python3 play_game.py --checkpoint checkpoint.pt --task dmc_walker_walk
python3 play_game.py --checkpoint checkpoint.pt --task dmc_cheetah_run
python3 play_game.py --checkpoint checkpoint.pt --task dmc_hopper_hop
```

#### Atari
```bash
python3 play_game.py --checkpoint checkpoint.pt --task atari_breakout
python3 play_game.py --checkpoint checkpoint.pt --task atari_space_invaders
python3 play_game.py --checkpoint checkpoint.pt --task atari_qbert
```

#### Minecraft
```bash
python3 play_game.py --checkpoint checkpoint.pt --task minecraft_creative
python3 play_game.py --checkpoint checkpoint.pt --task minecraft_survival
```

#### Memory Maze
```bash
python3 play_game.py --checkpoint checkpoint.pt --task memorymaze_maze
```

### Examples

#### Example 1: Play DMC Walker
```bash
python3 play_game.py \
    --checkpoint ./logdir/dmc_walker_walk/latest.pt \
    --task dmc_walker_walk \
    --output walker_gameplay.mp4 \
    --max_steps 5000
```

#### Example 2: Play Atari with Custom Config
```bash
python3 play_game.py \
    --checkpoint ./logdir/atari_breakout/latest.pt \
    --task atari_breakout \
    --configs atari \
    --output breakout_gameplay.mp4 \
    --max_steps 10000
```

#### Example 3: Play Minecraft
```bash
python3 play_game.py \
    --checkpoint ./logdir/minecraft_creative/latest.pt \
    --task minecraft_creative \
    --output minecraft_gameplay.mp4 \
    --max_steps 2000
```

## Video Output

The gameplay is saved as an MP4 video file at the specified output path. The video shows:
- Environment rendering
- Agent's gameplay in real-time
- Complete episode from start to finish

## Troubleshooting

### Common Issues

1. **Checkpoint not found**: Make sure the checkpoint file exists and the path is correct
2. **Environment creation failed**: Verify the task name is correct and all dependencies are installed
3. **Video not saved**: Check if OpenCV is installed (`pip install opencv-python`)
4. **CUDA out of memory**: Reduce `max_steps` or use CPU with `--device cpu`

### Device Selection

The tool automatically detects and uses CUDA if available. To force CPU usage, you can modify the script or set the device in the configuration.

### Debug Mode

For debugging, you can modify the script to print additional information:
- Add print statements to track gameplay progress
- Enable verbose logging in the configuration
- Check environment observations and actions

## Technical Details

### Checkpoint Loading

The tool supports multiple checkpoint formats:
- Complete agent state dictionaries
- World model only checkpoints
- Partial checkpoints with optimizer states

### Video Recording

Videos are recorded using OpenCV with:
- MP4 format (mp4v codec)
- 30 FPS frame rate
- RGB to BGR conversion for compatibility
- Automatic frame normalization

### Environment Support

All environments supported by DreamerV3 are compatible:
- DMC (DeepMind Control Suite)
- Atari games
- Minecraft environments
- Memory Maze
- Crafter
- Stable-retro games

## Integration with Training

This tool complements the main DreamerV3 training pipeline:

1. Train your agent using `dreamer.py`
2. Use the saved checkpoints (`latest.pt`) with `play_game.py`
3. Evaluate agent performance visually
4. Share gameplay videos for demonstration

## Contributing

To extend the functionality:
- Add new environment support in `make_env()`
- Modify video encoding parameters
- Add new checkpoint loading formats
- Enhance gameplay recording features