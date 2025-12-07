import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (ResizeObservation, GrayscaleObservation, FrameStackObservation)

import ale_py
gym.register_envs(ale_py)

from config import ENV_NAME, FRAME_STACK, RESIZE_SHAPE, STICKY_ACTION_PROB

class MaxAndSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
    
    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            if terminated or truncated:
                break
        
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer[0] = obs
        self._obs_buffer[1] = obs
        return obs, info


class FireOnResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        self.needs_fire = len(action_meanings) > 1 and action_meanings[1] == "FIRE"
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.needs_fire:
            # Press FIRE
            obs, _, terminated, truncated, info = self.env.step(1)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.game_over = True
        self.episode_return = 0.0
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track reward for true episode return
        self.episode_return += reward
        
        real_done = terminated or truncated
        self.game_over = real_done
        
        current_lives = self.env.unwrapped.ale.lives()
        if 0 < current_lives < self.lives:
            # Fake episode end on life loss
            terminated = True
        
        # If true game over (all lives lost), report episode return
        if real_done:
            info['episode_return'] = self.episode_return
            self.episode_return = 0.0
            
        self.lives = current_lives
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        if self.game_over:
            obs, info = self.env.reset(**kwargs)
            self.episode_return = 0.0
        else:
            # Continue from where we were (just lost a life)
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class ClipRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)


class RandomStartWrapper(gym.Wrapper):
    def __init__(self, env, max_noops=30):
        super().__init__(env)
        self.max_noops = max_noops
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        num_noops = np.random.randint(1, self.max_noops + 1)
        for _ in range(num_noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


def make_env(render_mode=None, training=True, env_name=None):    
    def _create():
        env = gym.make(
            env_name or ENV_NAME,
            render_mode=render_mode,
            frameskip=1,
            # Stochastic during training
            repeat_action_probability=STICKY_ACTION_PROB if training else 0.0,
        )
        
        # Frame skipping with max pooling
        env = MaxAndSkipWrapper(env, skip=4)
        
        if training:
            env = RandomStartWrapper(env, max_noops=30)
            env = EpisodicLifeWrapper(env)
            env = FireOnResetWrapper(env)
            env = ClipRewardWrapper(env)
        else:
            env = FireOnResetWrapper(env)
        
        # Standard preprocessing
        env = ResizeObservation(env, RESIZE_SHAPE)
        env = GrayscaleObservation(env, keep_dim=False)
        env = FrameStackObservation(env, FRAME_STACK)
        
        return env
    
    return _create


def make_vec_envs(num_envs, training=True, env_name=None):
    env_fns = [make_env(training=training, env_name=env_name) for _ in range(num_envs)]
    return gym.vector.AsyncVectorEnv(env_fns)

def make_single_env(render_mode=None, training=False, env_name=None):
    return make_env(render_mode=render_mode, training=training, env_name=env_name)()