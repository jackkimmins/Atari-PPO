import gymnasium as gym
import pygame
import numpy as np
from config import ENV_NAME, FRAME_SKIP

import ale_py
gym.register_envs(ale_py)

KEY_MAPPINGS = {
    pygame.K_SPACE: 1,   # FIRE
    pygame.K_RIGHT: 2,   # RIGHT
    pygame.K_LEFT: 3,    # LEFT
    pygame.K_d: 2,       # RIGHT (alternative)
    pygame.K_a: 3,       # LEFT (alternative)
    pygame.K_UP: 1,      # FIRE (alternative)
    pygame.K_w: 1,       # FIRE (alternative)
}

# Combined actions
COMBINED_ACTIONS = {
    (pygame.K_RIGHT, pygame.K_SPACE): 4,  # RIGHT + FIRE
    (pygame.K_LEFT, pygame.K_SPACE): 5,   # LEFT + FIRE
    (pygame.K_d, pygame.K_SPACE): 4,      # RIGHT + FIRE
    (pygame.K_a, pygame.K_SPACE): 5,      # LEFT + FIRE
}


def get_action_meanings(env):
    if hasattr(env.unwrapped, 'get_action_meanings'):
        meanings = env.unwrapped.get_action_meanings()
        print("\nAvailable actions:")
        for i, meaning in enumerate(meanings):
            print(f"  {i}: {meaning}")
        return meanings
    return None


def is_breakout_game(env_name):
    return 'breakout' in env_name.lower()


def print_controls(use_mouse=False):
    print("\n" + "=" * 50)
    print("CONTROLS")
    print("=" * 50)
    print("  Arrow Keys / WASD: Move")
    print("  Space / W / Up: Fire")
    if use_mouse:
        print("  Mouse: Move paddle (Breakout)")
        print("  Left Click: Fire")
    print("  R: Reset game")
    print("  Q / ESC: Quit")
    print("  +/-: Adjust game speed")
    print("=" * 50 + "\n")


def play():
    print(f"\nStarting interactive play for: {ENV_NAME}")
    
    # Game speed settings
    FRAME_SKIP_PLAY = 4
    TARGET_FPS = 30
    
    # Create environment with human rendering
    env = gym.make(
        ENV_NAME,
        render_mode="human",
        frameskip=1,
    )
    
    # Get action meanings
    action_meanings = get_action_meanings(env)
    
    # Check if this is Breakout for mouse support
    use_mouse = is_breakout_game(ENV_NAME)
    print_controls(use_mouse)
    
    pygame.init()
    
    # Mouse tracking for Breakout
    if use_mouse:
        pygame.event.set_grab(False)
        last_mouse_x = None
        mouse_deadzone = 5
        print("Mouse control enabled for Breakout!")
    
    # Reset environment
    obs, info = env.reset()
    
    total_reward = 0
    episode_count = 1
    lives = info.get('lives', 0)
    frame_skip = FRAME_SKIP_PLAY
    
    print(f"\nEpisode {episode_count} started!")
    if lives > 0:
        print(f"Lives: {lives}")
    print(f"Game speed: {frame_skip} frame skip (use +/- to adjust)")
    
    running = True
    clock = pygame.time.Clock()
    
    try:
        while running:
            action = 0
            
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        # Reset the game
                        obs, info = env.reset()
                        total_reward = 0
                        episode_count += 1
                        lives = info.get('lives', 0)
                        print(f"\nEpisode {episode_count} started! (Manual reset)")
                        if lives > 0:
                            print(f"Lives: {lives}")
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        # Slow down
                        frame_skip = min(frame_skip + 1, 8)
                        print(f"Game speed: {frame_skip} frame skip (slower)")
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        # Speed up
                        frame_skip = max(frame_skip - 1, 1)
                        print(f"Game speed: {frame_skip} frame skip (faster)")
            
            if not running:
                break
            
            # Get currently pressed keys
            keys = pygame.key.get_pressed()
            
            # Mouse control for Breakout
            if use_mouse:
                mouse_x, _ = pygame.mouse.get_pos()
                mouse_buttons = pygame.mouse.get_pressed()
                
                if last_mouse_x is not None:
                    mouse_delta = mouse_x - last_mouse_x
                    
                    # Check mouse movement with deadzone
                    if mouse_delta > mouse_deadzone:
                        # Moving right
                        if mouse_buttons[0]:  # Left click + right = RIGHT+FIRE
                            action = 4
                        else:
                            action = 2  # RIGHT
                    elif mouse_delta < -mouse_deadzone:
                        # Moving left
                        if mouse_buttons[0]:  # Left click + left = LEFT+FIRE
                            action = 5
                        else:
                            action = 3  # LEFT
                    elif mouse_buttons[0]:
                        # Just clicking (fire)
                        action = 1  # FIRE
                
                last_mouse_x = mouse_x
            
            # Keyboard controls (can override mouse)
            # Check for combined actions first
            action_found = False
            for key_combo, combo_action in COMBINED_ACTIONS.items():
                if all(keys[k] for k in key_combo):
                    action = combo_action
                    action_found = True
                    break
            
            # If no combined action, check single keys
            if not action_found:
                for key, key_action in KEY_MAPPINGS.items():
                    if keys[key]:
                        action = key_action
                        break
            
            # Apply frame skip - repeat the action for multiple frames
            step_reward = 0
            for _ in range(frame_skip):
                obs, reward, terminated, truncated, info = env.step(action)
                step_reward += reward
                if terminated or truncated:
                    break
            
            total_reward += step_reward
            
            # Check for life loss
            new_lives = info.get('lives', 0)
            if new_lives < lives and new_lives > 0:
                print(f"Life lost! Lives remaining: {new_lives}")
            lives = new_lives
            
            # Show reward when earned
            if step_reward > 0:
                print(f"  +{step_reward:.0f} points! (Total: {total_reward:.0f})")
            
            # Handle episode end
            if terminated or truncated:
                print(f"\nEpisode {episode_count} finished!")
                print(f"Total reward: {total_reward:.0f}")
                print("\nPress R to play again or Q/ESC to quit...")
                
                # Wait for user input
                waiting = True
                while waiting and running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                                running = False
                                waiting = False
                            elif event.key == pygame.K_r:
                                obs, info = env.reset()
                                total_reward = 0
                                episode_count += 1
                                lives = info.get('lives', 0)
                                print(f"\nEpisode {episode_count} started!")
                                if lives > 0:
                                    print(f"Lives: {lives}")
                                waiting = False
                    clock.tick(30)
            
            clock.tick(TARGET_FPS)
    
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    
    finally:
        print(f"\nGame Over! You played {episode_count} episode(s).")
        env.close()
        pygame.quit()


if __name__ == "__main__":
    play()