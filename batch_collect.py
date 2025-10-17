
#!/usr/bin/env python
import argparse
import os
import pathlib
import subprocess
import sys
import numpy as np
from collections import defaultdict

def find_latest_checkpoints(log_dir, reward_mode):
    """Finds the latest checkpoint for the specified reward mode for each game."""
    print(f"Scanning for checkpoints in: {log_dir}")
    print(f"Filtering for reward mode: {reward_mode}")

    game_checkpoints = defaultdict(lambda: {'timestamp': '', 'path': None})

    base_path = pathlib.Path(log_dir) / '200'
    if not base_path.exists():
        print(f"Error: Directory not found: {base_path}")
        return {}

    for game_path in base_path.iterdir():
        if not game_path.is_dir():
            continue
        game_name = game_path.name
        
        for run_path in game_path.iterdir():
            if not run_path.is_dir():
                continue
            
            run_name = run_path.name
            parts = run_name.split('-')
            if len(parts) != 2:
                continue

            mode, timestamp = parts
            if mode.upper() == reward_mode.upper():
                checkpoint_path = run_path / 'latest.pt'
                if checkpoint_path.exists():
                    # If this one is newer, update the entry for the game
                    if timestamp > game_checkpoints[game_name]['timestamp']:
                        game_checkpoints[game_name]['timestamp'] = timestamp
                        game_checkpoints[game_name]['path'] = str(checkpoint_path)
    
    # Filter out games where no valid checkpoint was found
    latest_checkpoints = {game: data['path'] for game, data in game_checkpoints.items() if data['path']}
    return latest_checkpoints

def main(args):
    reward_map = {
        'L1': 'survive',
        'L2': 'explore',
        'L3': 'win',
    }
    if args.reward_mode.upper() not in reward_map:
        print(f"Error: Invalid reward mode '{args.reward_mode}'. Must be L1, L2, or L3.")
        return

    checkpoints_to_run = find_latest_checkpoints(args.logdir, args.reward_mode)

    if not checkpoints_to_run:
        print("No checkpoints found to process. Exiting.")
        return

    print(f"\nFound {len(checkpoints_to_run)} games with valid checkpoints for mode {args.reward_mode}.")

    # --- GPU Parallelization Setup ---
    try:
        import torch
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("Warning: No CUDA-enabled GPUs found. Running sequentially on CPU.")
            num_gpus = 1 # Fallback to sequential
    except ImportError:
        print("Warning: PyTorch not found. Running sequentially on CPU.")
        num_gpus = 1
    
    print(f"Detected {num_gpus} available GPUs.")

    tasks = list(checkpoints_to_run.items())
    tasks_per_gpu = np.array_split(tasks, num_gpus)

    processes = []
    for gpu_id, gpu_tasks in enumerate(tasks_per_gpu):
        if gpu_tasks.size == 0:
            continue

        for game_name, checkpoint_path in gpu_tasks:
            print(f"\n--- Assigning task to GPU {gpu_id}: {game_name} ---")

            # Construct the retro game name
            retro_game_name = f"retro_{game_name}"

            # Construct the output directory
            output_reward_name = reward_map[args.reward_mode.upper()]
            output_path = pathlib.Path(args.output_dir) / game_name / output_reward_name

            # Construct the command
            command = [
                sys.executable,
                'collect_data.py',
                '--checkpoint', checkpoint_path,
                '--configs', 'retro', 'vjepa_base', 'vjepa_retro_vitl',
                '--game', retro_game_name,
                '--reward_mode', args.reward_mode.upper(),
                '--output_dir', str(output_path),
                '--steps', str(args.steps),
                '--device', f'cuda:0' # The script sees only one GPU, so it's always cuda:0
            ]

            # Set environment variable for the subprocess
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            print(f"Starting process on GPU {gpu_id} with command: {' '.join(command)}")
            # Launch the subprocess
            proc = subprocess.Popen(command, env=env)
            processes.append(proc)

    # Wait for all processes to complete
    print(f"\nLaunched {len(processes)} processes. Waiting for completion...")
    for proc in processes:
        proc.wait()

    print("\nBatch collection finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch collect data from the latest checkpoints for a given reward mode.")
    parser.add_argument("--reward_mode", required=True, choices=['L1', 'L2', 'L3'], help="The reward mode to collect data for (L1, L2, or L3).")
    parser.add_argument("--output_dir", required=True, help="The root directory to save all collected data.")
    parser.add_argument("--steps", type=int, default=10000, help="Total number of steps to collect per game. Default: 10000.")
    parser.add_argument("--logdir", default='./logdir', help="The root directory of the training logs. Default: ./logdir")

    main(parser.parse_args())
