import argparse
import os
import random
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from pong import PongEnv, simple_tracking_policy, STAY
from train_pong_ppo import SB3PongEnv, evaluate_model


def evaluate(model_path: str, episodes: int, render: bool) -> None:
    env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode="human" if render else None)
    model = PPO.load(model_path, env=env)
    metrics = evaluate_model(model, episodes)
    env.close()

    print(f"\nResults over {episodes} episodes")
    print(f"- Average reward: {metrics['avg_reward']:.3f} +/- {metrics['avg_reward_ci']:.3f}")
    print(f"- Win rate: {metrics['win_rate']:.2f}")
    print(f"- Average rally length: {metrics['avg_rally_length']:.2f}")
    print(f"- Average ball returns: {metrics['avg_return_rate']:.2f} +/- {metrics['avg_return_rate_ci']:.3f}")


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
    parser.add_argument("--device", type=str, default="auto", help="Device to load the model on.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV to append metrics to.")
    parser.add_argument("--deterministic", action="store_true", help="Toggle deterministic torch ops where possible.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        import torch

        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)

    if not Path(args.model_path).exists():
        print(f"Model not found at {args.model_path}. Exiting cleanly.")
        return

    try:
        evaluate(args.model_path, args.episodes, args.render)
    except FileNotFoundError:
        print(f"Could not load model from {args.model_path}")
        return

    if args.output_csv:
        metrics = evaluate_model(PPO.load(args.model_path, device=args.device), args.episodes)
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        headers = ["model_path", "episodes", "avg_reward", "avg_reward_ci", "win_rate", "avg_return_rate", "avg_return_rate_ci", "avg_rally_length"]
        exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "model_path": args.model_path,
                    "episodes": args.episodes,
                    **metrics,
                }
            )


if __name__ == "__main__":
    main()
