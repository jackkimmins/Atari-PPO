import os
import sys
import time
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import subprocess
import webbrowser
import socket
import select

# Cross-platform keyboard input
if sys.platform == 'win32':
    import msvcrt
else:
    import tty
    import termios

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import (
    DEVICE, NUM_ENVS, NUM_STEPS, BATCH_SIZE, MINIBATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, GAMMA, GAE_LAMBDA, CLIP_EPSILON, VALUE_COEF, ENTROPY_COEF,
    ENTROPY_COEF_FINAL, MAX_GRAD_NORM, ANNEAL_LR, ANNEAL_ENTROPY, TOTAL_TIMESTEPS, 
    MODEL_PATH, CHECKPOINT_PATH, RESUME_TRAINING, SAVE_INTERVAL, TENSORBOARD_DIR, ENV_NAME,
)

from model import AtariCNN
from env_utils import make_vec_envs, make_single_env

# Generalised Advantage Estimation (Looks ahead multiple steps to get a better estimate of advantage)
def compute_gae(rewards, values, dones, next_value):
    num_steps = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_adv = 0
    
    for t in reversed(range(num_steps)):
        next_val = next_value if t == num_steps - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_val * not_done - values[t]
        advantages[t] = delta + GAMMA * GAE_LAMBDA * not_done * last_adv
        last_adv = advantages[t]
    
    returns = advantages + values
    return advantages, returns


def save_checkpoint(model, optimiser, update, global_step, episode_returns, best_mean_return, tb_history):
    # Handle DataParallel wrapper - save the underlying model
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'model_state_dict': model_state,
        'optimiser_state_dict': optimiser.state_dict(),
        'update': update,
        'global_step': global_step,
        'episode_returns': episode_returns,
        'best_mean_return': best_mean_return,
        'tb_history': tb_history,
    }, CHECKPOINT_PATH)


def load_checkpoint(model, optimiser):
    if not RESUME_TRAINING or not os.path.exists(CHECKPOINT_PATH):
        return 1, 0, [], float('-inf'), []
    
    print("Resuming from checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    
    # Handle loading into DataParallel wrapped model
    # Checkpoints are always saved without 'module.' prefix
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    
    start_update = checkpoint['update'] + 1
    global_step = checkpoint['global_step']
    episode_returns = checkpoint.get('episode_returns', [])
    best_mean_return = checkpoint.get('best_mean_return', float('-inf'))
    tb_history = checkpoint.get('tb_history', [])
    
    return start_update, global_step, episode_returns, best_mean_return, tb_history


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def kill_tensorboard():
    try:
        subprocess.run(
            ['taskkill', '/F', '/IM', 'tensorboard.exe'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
    except Exception:
        pass

# Start TensorBoard
def start_tensorboard(logdir, port=6006, restart=False):
    if restart:
        kill_tensorboard()
    
    if is_port_in_use(port) and not restart:
        print(f"TensorBoard already running on port {port}")
    else:
        print(f"Starting TensorBoard on port {port}...")
        subprocess.Popen(
            ['tensorboard', '--logdir', logdir, '--port', str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        time.sleep(2)
    
    url = f"http://localhost:{port}"
    print(f"TensorBoard: {url}")
    webbrowser.open(url)


def format_sim_time(seconds):
    """Format simulated time as days:hours:minutes:seconds."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if days > 0:
        return f"{days}d{hours:02d}h"
    elif hours > 0:
        return f"{hours}h{minutes:02d}m"
    elif minutes > 0:
        return f"{minutes}m{secs:02d}s"
    else:
        return f"{secs}s"


# Run a single demo episode with rendering.
def run_demo(model, num_actions):
    env = make_single_env(render_mode="human", training=False)
    
    obs, _ = env.reset()
    obs = torch.tensor(obs, device=DEVICE).unsqueeze(0)
    
    done = False
    total_reward = 0
    
    while not done:
        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(obs)
        
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward
        obs = torch.tensor(next_obs, device=DEVICE).unsqueeze(0)
    
    env.close()
    
    # Force pygame to fully quit on Windows
    try:
        import pygame
        pygame.display.quit()
        pygame.quit()
    except Exception:
        pass
    
    return total_reward


def train():
    print(f"Device: {DEVICE}")
    num_gpus = torch.cuda.device_count()
    use_multi_gpu = num_gpus > 1
    
    if torch.cuda.is_available():
        if use_multi_gpu:
            print(f"Multi-GPU Training Enabled: {num_gpus} GPUs detected")
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set up environments
    envs = make_vec_envs(NUM_ENVS, training=True)
    num_actions = envs.single_action_space.n
    print(f"Atari Environment: {ENV_NAME}")
    print(f"Obs: {envs.single_observation_space.shape}, Actions: {num_actions}")
    
    # Create model and optimiser
    model = AtariCNN(num_actions).to(DEVICE)
    
    # Wrap model in DataParallel if multiple GPUs available
    if use_multi_gpu:
        model = nn.DataParallel(model)
        print(f"Model wrapped in DataParallel across {num_gpus} GPUs")
    
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    num_updates = TOTAL_TIMESTEPS // BATCH_SIZE
    
    # Try loading from checkpoint
    start_update, global_step, episode_returns, best_mean_return, tb_history = load_checkpoint(model, optimiser)
    is_resuming = start_update > 1
    
    # Kill TensorBoard if resuming
    if is_resuming:
        kill_tensorboard()
        time.sleep(1)
    
    if os.path.exists(TENSORBOARD_DIR):
        import shutil
        try:
            shutil.rmtree(TENSORBOARD_DIR)
        except PermissionError:
            print("Warning: Could not clear TensorBoard logs.")
    
    writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
    
    # Replay saved history to reconstruct TensorBoard graphs
    if tb_history:
        print(f"Reconstructing TensorBoard from {len(tb_history)} saved points...")
        for entry in tb_history:
            writer.add_scalar('charts/mean_return', entry['mean_return'], entry['step'])
            writer.add_scalar('charts/episodes', entry['episodes'], entry['step'])
        writer.flush()
    else:
        # Fresh start - write initial data point so TensorBoard has something to show
        writer.add_scalar('charts/mean_return', 0, 0)
        writer.add_scalar('charts/episodes', 0, 0)
        writer.flush()
    
    start_tensorboard(TENSORBOARD_DIR)
    
    # Buffers for rollout data
    obs_buf = torch.zeros((NUM_STEPS, NUM_ENVS, 4, 84, 84), device=DEVICE)
    action_buf = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.long, device=DEVICE)
    logprob_buf = torch.zeros((NUM_STEPS, NUM_ENVS), device=DEVICE)
    reward_buf = torch.zeros((NUM_STEPS, NUM_ENVS), device=DEVICE)
    done_buf = torch.zeros((NUM_STEPS, NUM_ENVS), device=DEVICE)
    value_buf = torch.zeros((NUM_STEPS, NUM_ENVS), device=DEVICE)
    
    # Initialise
    obs, _ = envs.reset()
    obs = torch.tensor(obs, device=DEVICE)
    done = torch.zeros(NUM_ENVS, device=DEVICE)
    
    start_time = time.time()
    session_steps = 0  # Steps in this session (for accurate FPS)
    
    print("Press 'D' at any time to watch a demo of the current model")
    
    # Progress bar
    pbar = tqdm(
        range(start_update, num_updates + 1),
        initial=start_update - 1,
        total=num_updates,
        desc="Training (D=demo)",
        unit="update",
        ncols=130,
    )
    
    for update in pbar:
        # Anneal learning rate
        progress = 1.0 - (update - 1) / num_updates
        if ANNEAL_LR:
            optimiser.param_groups[0]["lr"] = progress * LEARNING_RATE
        
        # Anneal entropy coefficient for refined exploitation later in training
        if ANNEAL_ENTROPY:
            current_entropy_coef = ENTROPY_COEF_FINAL + progress * (ENTROPY_COEF - ENTROPY_COEF_FINAL)
        else:
            current_entropy_coef = ENTROPY_COEF
        
        # Collect experience
        model.eval()
        for step in range(NUM_STEPS):
            global_step += NUM_ENVS
            session_steps += NUM_ENVS
            
            obs_buf[step] = obs
            done_buf[step] = done
            
            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(obs)
            
            action_buf[step] = action
            logprob_buf[step] = logprob
            value_buf[step] = value
            
            # Step all environments
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = torch.tensor(terminated | truncated, dtype=torch.float32, device=DEVICE)
            reward_buf[step] = torch.tensor(reward, device=DEVICE)
            obs = torch.tensor(next_obs, device=DEVICE)
            
            # Track completed episodes - look for true game overs (lives=0)
            if 'episode_return' in infos:
                for i in range(NUM_ENVS):
                    ep_ret = infos['episode_return'][i]
                    if ep_ret is not None and ep_ret > 0:
                        episode_returns.append(float(ep_ret))
        
        # Bootstrap final value
        with torch.no_grad():
            next_value = model.get_value(obs)
        
        # Compute advantages
        advantages, returns = compute_gae(reward_buf, value_buf, done_buf, next_value)
        
        # Flatten for training
        b_obs = obs_buf.reshape(-1, 4, 84, 84)
        b_actions = action_buf.reshape(-1)
        b_logprobs = logprob_buf.reshape(-1)
        b_values = value_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        # Normalise advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # PPO update
        model.train()
        indices = np.arange(BATCH_SIZE)
        
        for _ in range(NUM_EPOCHS):
            np.random.shuffle(indices)
            
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                mb_idx = indices[start:start + MINIBATCH_SIZE]
                
                _, new_logprob, entropy, new_value = model.get_action_and_value(b_obs[mb_idx], b_actions[mb_idx])
                
                # Policy loss
                ratio = (new_logprob - b_logprobs[mb_idx]).exp()
                mb_adv = b_advantages[mb_idx]
                
                loss1 = -mb_adv * ratio
                loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                policy_loss = torch.max(loss1, loss2).mean()
                
                # Value loss with clipping (PPO paper recommendation)
                v_clipped = b_values[mb_idx] + torch.clamp(
                    new_value - b_values[mb_idx], -CLIP_EPSILON, CLIP_EPSILON
                )
                v_loss_unclipped = (new_value - b_returns[mb_idx]) ** 2
                v_loss_clipped = (v_clipped - b_returns[mb_idx]) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                
                # Entropy bonus
                entropy_loss = entropy.mean()
                
                # Combined
                loss = policy_loss + VALUE_COEF * value_loss - current_entropy_coef * entropy_loss
                
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimiser.step()
        
        # Update progress bar
        elapsed = time.time() - start_time
        fps = session_steps / elapsed if elapsed > 0 else 0
        mean_ret = np.mean(episode_returns[-100:]) if episode_returns else 0
        
        # Simulated time
        sim_time_seconds = global_step * 4 / 60.0
        sim_time_str = format_sim_time(sim_time_seconds)
        
        pbar.set_postfix({
            'FPS': f'{fps:.0f}',
            'Return': f'{mean_ret:.1f}',
            'Episodes': len(episode_returns),
            'SimTime': sim_time_str,
        })
        
        if episode_returns and mean_ret > best_mean_return:
            best_mean_return = mean_ret
        
        # Log to TensorBoard
        writer.add_scalar('charts/mean_return', mean_ret, global_step)
        writer.add_scalar('charts/episodes', len(episode_returns), global_step)
        writer.add_scalar('charts/fps', fps, global_step)
        writer.add_scalar('charts/sim_time_hours', sim_time_seconds / 3600, global_step)
        writer.add_scalar('charts/learning_rate', optimiser.param_groups[0]['lr'], global_step)
        writer.add_scalar('losses/policy_loss', policy_loss.item(), global_step)
        writer.add_scalar('losses/value_loss', value_loss.item(), global_step)
        writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)
        
        # Save key metrics to history
        tb_history.append({
            'step': global_step,
            'mean_return': float(mean_ret),
            'episodes': len(episode_returns),
        })
        
        # Save checkpoint
        if update % SAVE_INTERVAL == 0:
            save_checkpoint(model, optimiser, update, global_step, episode_returns, best_mean_return, tb_history)
            # Handle DataParallel wrapper - save the underlying model
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'model_state_dict': model_state,
                'update': update,
                'global_step': global_step,
                'mean_return': mean_ret,
            }, MODEL_PATH)
        
        # Check for keyboard input (cross-platform)
        key_pressed = None
        if sys.platform == 'win32':
            if msvcrt.kbhit():
                key_pressed = msvcrt.getch().decode('utf-8', errors='ignore').lower()
        else:
            # Linux/macOS: use select to check for input
            if select.select([sys.stdin], [], [], 0)[0]:
                key_pressed = sys.stdin.read(1).lower()
        
        if key_pressed == 'd':
                pbar.write(f"\n{'='*50}")
                pbar.write(f"Demo time! Watching the agent play...")
                pbar.write(f"{'='*50}")
                
                model.eval()
                # Use underlying model for demo (runs on single GPU)
                demo_model = model.module if hasattr(model, 'module') else model
                demo_score = run_demo(demo_model, num_actions)
                pbar.write(f"Demo score: {demo_score:.0f}")
                pbar.write(f"{'='*50}\n")
    
    pbar.close()
    writer.close()
    
    # Final save
    mean_ret = np.mean(episode_returns[-100:]) if episode_returns else 0
    # Handle DataParallel wrapper - save the underlying model
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'model_state_dict': model_state,
        'update': num_updates,
        'global_step': global_step,
        'mean_return': mean_ret,
    }, MODEL_PATH)
    
    # Clean up checkpoint since we finished
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    
    envs.close()
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Final mean return: {mean_ret:.1f}")
    print(f"Best mean return: {best_mean_return:.1f}")
    print(f"Model saved to: {MODEL_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    train()
