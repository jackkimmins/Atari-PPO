import time
import os
import glob
import torch
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from config import DEVICE, EVAL_RENDER, EVAL_DELAY
from model import AtariCNN
from env_utils import make_single_env

FULLY_TRAINED_DIR = "fully_trained"

def get_available_models():
    pattern = os.path.join(FULLY_TRAINED_DIR, "*.pt")
    model_files = glob.glob(pattern)
    return sorted(model_files)


def extract_env_name(model_path):
    filename = os.path.basename(model_path)
    if filename.startswith("model_"):
        env_short = filename[6:-3]
    else:
        env_short = filename[:-3]
    return f"ALE/{env_short}"


def select_model():
    models = get_available_models()
    
    if not models:
        print(f"Error: No .pt files found in '{FULLY_TRAINED_DIR}/' folder.")
        print("Please add trained model files to this folder.")
        return None, None
    
    print("\n" + "=" * 50)
    print("Available Trained Models")
    print("=" * 50)
    
    for i, model_path in enumerate(models, 1):
        env_name = extract_env_name(model_path)
        game_name = env_name.split("/")[-1].replace("-v5", "")
        print(f"  {i}. {game_name:20s} ({os.path.basename(model_path)})")
    
    print("=" * 50)
    
    while True:
        try:
            choice = input(f"\nSelect a model (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected_path = models[idx]
                selected_env = extract_env_name(selected_path)
                return selected_path, selected_env
            else:
                print(f"Please enter a number between 1 and {len(models)}.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None, None


def test():
    # Let user select a model
    model_path, env_name = select_model()
    if model_path is None:
        return
    
    print(f"\nDevice: {DEVICE}")
    print(f"Environment: {env_name}")
    print(f"Loading model from: {model_path}")
    
    # Create environment for the selected game
    render_mode = "human" if EVAL_RENDER else None
    env = make_single_env(render_mode=render_mode, training=False, env_name=env_name)
    num_actions = env.action_space.n
    
    # Load trained model
    model = AtariCNN(num_actions).to(DEVICE)
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from update {checkpoint.get('update', 'unknown')}")
        print(f"Training mean return: {checkpoint.get('mean_return', 'unknown'):.1f}")
    except FileNotFoundError:
        print(f"Error: Can't find '{model_path}'.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval()
    
    game_name = env_name.split("/")[-1].replace("-v5", "")
    print("\n" + "=" * 60)
    print(f"Playing {game_name}...")
    print("=" * 60 + "\n")
    
    # Play one episode
    obs, _ = env.reset()
    obs = torch.tensor(obs, device=DEVICE).unsqueeze(0)
    
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(obs)
        
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward
        steps += 1
        
        obs = torch.tensor(next_obs, device=DEVICE).unsqueeze(0)
        
        if EVAL_RENDER and EVAL_DELAY > 0:
            time.sleep(EVAL_DELAY)
    
    env.close()
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Game Over - {game_name}")
    print("=" * 60)
    print(f"Environment:  {env_name}")
    print(f"Final Score:  {total_reward:.0f}")
    print(f"Steps:        {steps}")
    print("=" * 60)

if __name__ == "__main__":
    test()
