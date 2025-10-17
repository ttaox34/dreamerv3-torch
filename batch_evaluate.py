#!/usr/bin/env python
import argparse
import collections
import os
import pathlib
import re
import sys
from datetime import datetime

import numpy as np
import ruamel.yaml as yaml
import torch

# Add project root to path to allow imports from the project
sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
from dreamer import make_env, Dreamer


def find_latest_checkpoints(logdir_root):
    logdir_root = pathlib.Path(logdir_root)
    if not logdir_root.exists():
        print(f"ERROR: Logdir root not found at {logdir_root}")
        return []

    checkpoints = collections.defaultdict(list)
    for pt_path in logdir_root.glob("**/latest.pt"):
        try:
            game_name = pt_path.parts[-3]
            run_folder = pt_path.parts[-2]
            match = re.match(r"(L[1-3])-(\d+)", run_folder)
            if not match:
                continue
            reward_mode, timestamp_str = match.groups()
            checkpoints[(game_name, reward_mode)].append((timestamp_str, pt_path))
        except IndexError:
            continue
            
    latest_checkpoints = []
    for (game_name, reward_mode), runs in checkpoints.items():
        runs.sort(key=lambda x: x[0], reverse=True)
        latest_path = runs[0][1]
        latest_checkpoints.append({
            "game": game_name,
            "reward_mode": reward_mode,
            "path": latest_path
        })
        
    latest_checkpoints.sort(key=lambda x: (x['game'], x['reward_mode']))
    return latest_checkpoints

def load_previous_results(filepath):
    if not os.path.exists(filepath):
        return []
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(r"Game: (\S+)\s+Mode: (\S+)\s+Status: (\S+)", line)
            if match:
                game, mode, status = match.groups()
                results.append({
                    "game": game.strip(),
                    "reward_mode": mode.strip(),
                    "is_active": True if status == "ACTIVE" else False if status == "INACTIVE" else None
                })
    return results

def write_summary_file(filepath, results):
    total = len(results)
    inactive_count = sum(1 for r in results if r['is_active'] is False)
    active_count = sum(1 for r in results if r['is_active'] is True)
    error_count = sum(1 for r in results if r['is_active'] is None)

    with open(filepath, 'w') as f:
        f.write(f"Batch Evaluation Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Evaluated {total} checkpoints.\n\n")
        
        for res in sorted(results, key=lambda x: (x['game'], x['reward_mode'])):
            status = "ERROR" if res['is_active'] is None else "ACTIVE" if res['is_active'] else "INACTIVE"
            f.write(f"Game: {res['game']:<30} Mode: {res['reward_mode']:<4} Status: {status}\n")

        if total > 0:
            inactive_percentage = (inactive_count / total) * 100 if total > 0 else 0
            f.write("\n" + "="*50 + "\n")
            f.write("SUMMARY:\n")
            f.write(f"- Total Checkpoints: {total}\n")
            f.write(f"- Active Agents:     {active_count}\n")
            f.write(f"- Inactive Agents:   {inactive_count} ({inactive_percentage:.1f}%)\n")
            f.write(f"- Errors:            {error_count}\n")
            f.write("="*50 + "\n")

def evaluate_checkpoint(agent, checkpoint_info, base_configs):
    game = checkpoint_info["game"]
    reward_mode = checkpoint_info["reward_mode"]
    checkpoint_path = checkpoint_info["path"]

    print(f"  - Evaluating {game} ({reward_mode})...")

    config_dict = base_configs.copy()
    config_dict['task'] = f"retro_{game}"
    config_dict['reward_mode'] = reward_mode
    config = argparse.Namespace(**config_dict)

    try:
        env = make_env(config, "eval", 0)
        
        ckpt = torch.load(checkpoint_path, map_location=config.device)
        cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["agent_state_dict"].items()}
        agent.load_state_dict(cleaned_state_dict)
        agent.eval()

        obs = env.reset()
        agent_state = None
        is_active = False
        done = False

        while not done:
            obs_for_agent = {k: torch.from_numpy(v).unsqueeze(0).to(config.device) for k, v in obs.items() if k == 'image'}
            obs_for_agent['is_first'] = torch.tensor([obs['is_first']], device=config.device)
            obs_for_agent['is_terminal'] = torch.tensor([obs['is_terminal']], device=config.device)

            with torch.no_grad():
                policy_output, agent_state = agent._policy(obs_for_agent, agent_state, training=False)
            
            action_onehot = policy_output['action'].squeeze(0).cpu().numpy()
            discrete_action = np.argmax(action_onehot)

            if discrete_action != 0:
                is_active = True

            obs, _, done, _ = env.step({'action': action_onehot})
        
        env.close()
        return is_active

    except Exception as e:
        print(f"  - ERROR evaluating {game} ({reward_mode}): {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir_root", type=str, default="./logdir/200", help="Root directory containing game checkpoints.")
    parser.add_argument("--output_file", type=str, default="batch_eval_summary.txt", help="File to save the summary.")
    parser.add_argument("--configs", nargs="+", default=["vjepa_retro_vitl"], help="List of base configs to use.")
    args = parser.parse_args()

    # --- Load Base Configs ---
    configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    def resolve_config_hierarchy(names, all_configs):
        resolved_order = []
        processed = set()
        def _resolve(config_names):
            for name in config_names:
                if name in processed:
                    continue
                config_part = all_configs.get(name, {})
                if 'defaults' in config_part:
                    _resolve(config_part['defaults'])
                if name not in processed:
                    resolved_order.append(name)
                    processed.add(name)
        _resolve(names)
        return resolved_order

    initial_names = ["defaults", *args.configs]
    ordered_names = resolve_config_hierarchy(initial_names, configs)
    
    print(f"Applying configs in order: {ordered_names}")

    defaults = {}
    for name in ordered_names:
        config_part = configs.get(name, {})
        recursive_update(defaults, config_part)

    # --- Resume Logic ---
    previous_results = load_previous_results(args.output_file)
    completed_set = {(r['game'], r['reward_mode']) for r in previous_results}
    if completed_set:
        print(f"Found {len(completed_set)} previously evaluated checkpoints. Will skip them.")

    # --- Find Checkpoints ---
    all_checkpoints = find_latest_checkpoints(args.logdir_root)
    checkpoints_to_eval = [c for c in all_checkpoints if (c['game'], c['reward_mode']) not in completed_set]
    print(f"Found {len(all_checkpoints)} total unique checkpoints. {len(checkpoints_to_eval)} remaining to evaluate.")

    if not checkpoints_to_eval:
        print("All checkpoints already evaluated. Exiting.")
        return

    # --- Create Persistent Agent (Optimization) ---
    print("Creating persistent agent instance to reuse...")
    template_info = checkpoints_to_eval[0]
    config_dict = defaults.copy()
    config_dict['task'] = f"retro_{template_info['game']}"
    config_dict['compile'] = False
    template_config = argparse.Namespace(**config_dict)
    
    # A dummy env is needed just to get observation and action space for agent creation
    dummy_env = make_env(template_config, "eval", 0)

    # Set num_actions from the dummy env's action space, which is required by the model
    acts = dummy_env.action_space
    template_config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    agent = Dreamer(dummy_env.observation_space, dummy_env.action_space, template_config, None, None).to(template_config.device)
    agent.requires_grad_(requires_grad=False)
    dummy_env.close()
    print("Agent created successfully.")

    # --- Run Evaluations ---
    current_results = list(previous_results)
    for i, checkpoint_info in enumerate(checkpoints_to_eval):
        print(f"\nProcessing {i+1}/{len(checkpoints_to_eval)}...")
        is_active = evaluate_checkpoint(agent, checkpoint_info, defaults)
        
        current_results.append({**checkpoint_info, "is_active": is_active})
        
        # Overwrite the summary file after each evaluation for robustness
        write_summary_file(args.output_file, current_results)

    print("\n" + "="*50)
    print("      BATCH EVALUATION COMPLETE")
    print("="*50)
    final_total = len(current_results)
    final_inactive = sum(1 for r in current_results if r['is_active'] is False)
    final_active = sum(1 for r in current_results if r['is_active'] is True)
    final_errors = final_total - final_inactive - final_active
    print(f"Total: {final_total}, Active: {final_active}, Inactive: {final_inactive}, Errors: {final_errors}")
    if final_total > 0:
        print(f"Inactive Percentage: {(final_inactive / final_total) * 100:.1f}%")
    print(f"Full report saved to {args.output_file}")

if __name__ == "__main__":
    main()