import os
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from pong import (
    PongEnv,
    simple_tracking_policy,
    Action,
    STAY,
)


class SB3PongEnv(gym.Env):
    """
    Gymnasium wrapper around the custom PongEnv.
    The learning agent controls the left paddle; the right paddle uses a fixed policy.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        opponent_policy: Optional[Callable[[tuple, bool], Action]] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.env = PongEnv(render_mode=render_mode)
        self.opponent_policy = opponent_policy or (lambda obs, is_left: STAY)
        self.last_obs: Optional[tuple] = None

        # Observations are normalized: [bx, by, bvx, bvy, ly, ry]
        low = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        obs, info = self.env.reset()
        self.last_obs = obs
        return np.array(obs, dtype=np.float32), info

    def step(self, action: int):
        if self.last_obs is None:
            raise RuntimeError("Call reset() before step().")

        right_action = self.opponent_policy(self.last_obs, is_left=False)
        obs, reward, done, info = self.env.step(action, right_action)
        self.last_obs = obs

        terminated = False  # Episodes end only by truncation (step cap) in PongEnv.
        truncated = bool(done)
        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Vectorize custom Pong environment for PPO.
    env = make_vec_env(
        lambda: SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None),
        n_envs=8,
        seed=0,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs",
        n_steps=256,
        batch_size=512,
        n_epochs=4,
        gamma=0.99,
        learning_rate=2.5e-4,
    )

    # Shorter run for a quick playable model; bump this higher for better skill.
    model.learn(total_timesteps=100_000)
    model.save("models/ppo_pong_custom")
    env.close()

    print("Saved: models/ppo_pong_custom.zip")


if __name__ == "__main__":
    main()
