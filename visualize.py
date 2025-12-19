print("\n" + "="*40)
print("--- RUNNING CLEAN VISUALIZER (v3.0) ---")
print("="*40 + "\n")

import os
import argparse
import yaml
import numpy as np
import imageio
import tonic
import torch
import warnings

# Suppress Gym warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

# Import your custom classes
from model import SwimmerModule, SwimmerActor

def write_video(filepath, frames, fps=60, quality=10):
    """Saves a sequence of frames as a highly compatible MP4 video."""
    print(f"Saving video to: {filepath}")
    try:
        with imageio.get_writer(filepath, fps=fps, macro_block_size=None, 
                                ffmpeg_params=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']) as video:
            for frame in frames:
                video.append_data(frame)
        print("Video saved successfully!")
    except Exception as e:
        print(f"Standard save failed ({e}), trying fallback mode...")
        with imageio.get_writer(filepath, fps=fps) as video:
            for frame in frames:
                video.append_data(frame)

def play_model(path, checkpoint='last', environment='default', seed=None, header=None, save_video=True):
    
    # --- 1. RESOLVE CHECKPOINT ---
    if checkpoint == 'none':
        checkpoint_path = None
        ckpt_id = 'none'
    else:
        checkpoint_dir = os.path.join(path, 'checkpoints')
        if not os.path.isdir(checkpoint_dir):
            print(f"Error: {checkpoint_dir} is not a directory.")
            return

        files_in_dir = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') and 'step_' in f]
        if not files_in_dir:
            print(f"No checkpoints found in {checkpoint_dir}")
            return

        checkpoint_ids = []
        for f in files_in_dir:
            try:
                cid = int(f.split('_')[1].split('.')[0])
                checkpoint_ids.append(cid)
            except:
                continue

        if checkpoint == 'last':
            ckpt_id = max(checkpoint_ids)
        elif checkpoint == 'first':
            ckpt_id = min(checkpoint_ids)
        else:
            clean_str = checkpoint.replace('step_', '').replace('.pt', '')
            try:
                ckpt_id = int(clean_str)
            except ValueError:
                print(f"Invalid checkpoint format: {checkpoint}")
                return
            if ckpt_id not in checkpoint_ids:
                print(f"Checkpoint ID {ckpt_id} not found.")
                return

        # Find exact filename
        found_file = None
        for fname in files_in_dir:
            if f"step_{ckpt_id}.pt" == fname:
                found_file = fname
                break
        
        if found_file:
            checkpoint_path = os.path.join(checkpoint_dir, found_file)
        else:
            print(f"CRITICAL: Could not find file for ID {ckpt_id}")
            return

    # --- 2. LOAD CONFIG ---
    config_path = os.path.join(path, 'config.yaml')
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = argparse.Namespace(**config_dict)

    # --- 3. EXECUTE HEADERS ---
    exec("import tonic.torch")
    exec("import torch")
    if config.header:
        exec(config.header)
    if header:
        exec(header)

    # --- 4. BUILD ENVIRONMENT ---
    if environment == 'default':
        environment_str = config.environment
    else:
        environment_str = environment

    env = tonic.environments.distribute(lambda: eval(environment_str))
    if seed is not None:
        env.seed(seed)

    # --- 5. INITIALIZE AGENT (CLEAN MODE) ---
    agent = eval(config.agent)
    
    # Initialize naturally with the environment's actual shape
    agent.initialize(
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=seed
    )

    # --- 6. LOAD WEIGHTS ---
    if checkpoint_path:
        # Strip extension to please Tonic
        path_clean = os.path.splitext(checkpoint_path)[0]
        print(f"Loading weights from: {path_clean}.pt")
        agent.load(path_clean)

    # --- 7. RUN SIMULATION ---
    print("Starting simulation...")
    steps = 0
    observations = env.start()
    
    frames = [env.render('rgb_array', camera_id=0, width=640, height=480)[0]]
    
    score = 0
    length = 0

    while True:
        actions = agent.test_step(observations, steps)
        observations, infos = env.step(actions)
        
        frame = env.render('rgb_array', camera_id=0, width=640, height=480)[0]
        frames.append(frame)
        
        agent.test_update(**infos, steps=steps)

        score += infos['rewards'][0]
        length += 1

        if infos['resets'][0]:
            break

    print(f"Simulation finished. Total Reward: {score:.2f}, Length: {length} steps")

    if save_video:
        video_name = f"video_checkpoint_{ckpt_id}.mp4"
        video_path = os.path.join(path, video_name)
        write_video(video_path, frames, fps=30)
        return video_path
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to the experiment folder')
    parser.add_argument('--checkpoint', type=str, default='last', help='Checkpoint ID, "last", or "first"')
    args = parser.parse_args()

    clean_path = os.path.normpath(args.path)
    if os.path.basename(clean_path) == 'checkpoints':
        args.path = os.path.dirname(clean_path)
    
    play_model(args.path, checkpoint=args.checkpoint)