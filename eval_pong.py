import argparse
import random

import numpy as np
from stable_baselines3 import PPO

from pong import PongEnv, simple_tracking_policy, STAY
from train_pong_ppo import SB3PongEnv


def evaluate(model_path: str, episodes: int, render: bool) -> None:
    env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode="human" if render else None)
    model = PPO.load(model_path, env=env)

    rewards = []
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_rew = 0.0
        returns_this_ep = 0
        steps = 0
        while not done and steps < env.env.cfg.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            if rew > 0:
                returns_this_ep += 1
            ep_rew += rew
            steps += 1
            done = terminated or truncated
        rewards.append(ep_rew)
        returns.append(returns_this_ep)
        print(f"Episode {ep + 1}: reward={ep_rew:.3f}, ball_returns={returns_this_ep}")

    env.close()
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_returns = float(np.mean(returns)) if returns else 0.0
    print(f"\nResults over {episodes} episodes")
    print(f"- Average reward: {avg_reward:.3f}")
    print(f"- Average ball returns: {avg_returns:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Pong PPO checkpoint.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ppo_pong_custom_latest.zip",
        help="Path to the PPO checkpoint to evaluate.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true", help="Render a human-visible window during evaluation.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    evaluate(args.model_path, args.episodes, args.render)


if __name__ == "__main__":
    main()
