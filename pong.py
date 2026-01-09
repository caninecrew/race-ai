import os
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import numpy as np
import pygame


# ----------------------------
# Discrete actions
# ----------------------------
Action = int
STAY: Action = 0
UP: Action = 1
DOWN: Action = 2


@dataclass
class PongConfig:
    width: int = 600
    height: int = 400

    paddle_w: int = 8
    paddle_h: int = 80
    ball_radius: int = 10

    paddle_speed: float = 7.5
    ball_speed_min_x: float = 3.0
    ball_speed_max_x: float = 4.5
    ball_speed_min_y: float = 2.0
    ball_speed_max_y: float = 3.8

    ball_speedup: float = 1.03
    max_ball_speed: float = 12.0

    max_steps: int = 5000

    reward_score: float = 1.0
    reward_concede: float = -1.0
    reward_hit: float = 0.02
    reward_step: float = -0.0005

    render_fps: int = 60  # only used when render_mode="human"
    ball_color: Tuple[int, int, int] = (255, 0, 0)  # RGB


class PongEnv:
    """
    RL-friendly Pong environment with optional real-time rendering.

    - render_mode=None     -> headless (fast training)
    - render_mode="human"  -> open a pygame window (watch play)
    - render_mode="rgb_array" -> offscreen frames for video capture
    """

    def __init__(
        self,
        config: PongConfig = PongConfig(),
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        ball_color: Optional[Tuple[int, int, int]] = None,
    ):
        self.cfg = config
        self.render_mode = render_mode  # None or "human"
        self.ball_color = ball_color or self.cfg.ball_color
        self._rng = random.Random(seed)

        # Important: dummy video driver must be set BEFORE pygame.init()
        if self.render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()

        self.screen = None
        self.font = pygame.font.SysFont("Arial", 18)
        self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self._create_window()
        elif self.render_mode == "rgb_array":
            self._create_surface()

        # State
        self.left_score = 0
        self.right_score = 0
        self.steps = 0

        # Entities
        self.left_paddle = pygame.Rect(0, 0, self.cfg.paddle_w, self.cfg.paddle_h)
        self.right_paddle = pygame.Rect(0, 0, self.cfg.paddle_w, self.cfg.paddle_h)
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)

        self.reset(seed=seed)

    # ----------------------------
    # Rendering controls
    # ----------------------------
    def enable_render(self) -> None:
        """
        Enable a visible pygame window. Best used BEFORE training begins.
        (Switching from dummy->human in the same process can be platform-dependent.)
        """
        if self.render_mode == "human":
            return
        self.render_mode = "human"
        # Attempt to create window
        self._create_window()

    def disable_render(self) -> None:
        """
        Disable rendering. Keeps env functional; stops drawing.
        """
        self.render_mode = None
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None

    def _create_window(self) -> None:
        self.screen = pygame.display.set_mode((self.cfg.width, self.cfg.height))
        pygame.display.set_caption("Pong (AI View)")

    def _create_surface(self) -> None:
        # Offscreen surface used for rgb_array capture without opening a window.
        self.screen = pygame.Surface((self.cfg.width, self.cfg.height))

    # ----------------------------
    # API
    # ----------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[Tuple[float, ...], Dict]:
        if seed is not None:
            self._rng.seed(seed)
        self.left_score = 0
        self.right_score = 0
        self.steps = 0

        self.left_paddle.x = 1
        self.left_paddle.y = (self.cfg.height - self.cfg.paddle_h) // 2

        self.right_paddle.x = self.cfg.width - self.cfg.paddle_w - 1
        self.right_paddle.y = (self.cfg.height - self.cfg.paddle_h) // 2

        self._reset_ball(to_right=self._rng.choice([True, False]))

        return self._get_obs(), self._info()

    def step(self, left_action: Action, right_action: Action) -> Tuple[Tuple[float, ...], float, bool, Dict]:
        self.steps += 1

        # Keep window responsive
        if self.render_mode == "human":
            # Only process quit events here; leave other input for caller to handle.
            for event in pygame.event.get([pygame.QUIT]):
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit

        # Apply actions
        self._apply_action(self.left_paddle, left_action)
        self._apply_action(self.right_paddle, right_action)

        # Rewards (left is "main" reward returned)
        reward_left = self.cfg.reward_step
        reward_right = self.cfg.reward_step

        # Move ball
        self.ball_pos += self.ball_vel

        # Top/bottom wall collisions
        if self.ball_pos.y - self.cfg.ball_radius <= 0:
            self.ball_pos.y = self.cfg.ball_radius
            self.ball_vel.y *= -1
        elif self.ball_pos.y + self.cfg.ball_radius >= self.cfg.height:
            self.ball_pos.y = self.cfg.height - self.cfg.ball_radius
            self.ball_vel.y *= -1

        # Scoring happens the moment the ball crosses a gutter (before paddle checks).
        # This prevents bouncing off the back of a paddle, which is not how Pong behaves.
        scored = None
        if self.ball_pos.x - self.cfg.ball_radius <= 0:
            self.right_score += 1
            reward_left += self.cfg.reward_concede
            reward_right += self.cfg.reward_score
            scored = "right"
            self._reset_ball(to_right=True)

        elif self.ball_pos.x + self.cfg.ball_radius >= self.cfg.width:
            self.left_score += 1
            reward_left += self.cfg.reward_score
            reward_right += self.cfg.reward_concede
            scored = "left"
            self._reset_ball(to_right=False)

        # Paddle collisions (only if a point was not just scored)
        if scored is None:
            ball_rect = pygame.Rect(
                int(self.ball_pos.x - self.cfg.ball_radius),
                int(self.ball_pos.y - self.cfg.ball_radius),
                self.cfg.ball_radius * 2,
                self.cfg.ball_radius * 2,
            )

            if self.ball_vel.x < 0 and ball_rect.colliderect(self.left_paddle):
                self.ball_pos.x = self.left_paddle.right + self.cfg.ball_radius
                self._bounce_from_paddle(self.left_paddle)
                reward_left += self.cfg.reward_hit

            if self.ball_vel.x > 0 and ball_rect.colliderect(self.right_paddle):
                self.ball_pos.x = self.right_paddle.left - self.cfg.ball_radius
                self._bounce_from_paddle(self.right_paddle)
                reward_right += self.cfg.reward_hit

        # Episode termination only on step limit
        done = False

        if self.steps >= self.cfg.max_steps:
            done = True

        obs = self._get_obs()
        info = self._info()
        info["reward_right"] = reward_right
        info["steps"] = self.steps

        if self.render_mode == "human":
            self.render()

        return obs, reward_left, done, info

    def render(self) -> None:
        if self.render_mode not in ("human", "rgb_array") or self.screen is None:
            return

        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        BLACK = (0, 0, 0)
        YELLOW = (255, 255, 0)

        self.screen.fill(BLACK)

        # Court
        pygame.draw.line(self.screen, WHITE, (self.cfg.width // 2, 0), (self.cfg.width // 2, self.cfg.height), 1)
        pygame.draw.line(self.screen, WHITE, (self.cfg.paddle_w, 0), (self.cfg.paddle_w, self.cfg.height), 1)
        pygame.draw.line(self.screen, WHITE, (self.cfg.width - self.cfg.paddle_w, 0), (self.cfg.width - self.cfg.paddle_w, self.cfg.height), 1)
        pygame.draw.circle(self.screen, WHITE, (self.cfg.width // 2, self.cfg.height // 2), 70, 1)

        # Entities
        pygame.draw.rect(self.screen, GREEN, self.left_paddle)
        pygame.draw.rect(self.screen, GREEN, self.right_paddle)
        pygame.draw.circle(self.screen, self.ball_color, (int(self.ball_pos.x), int(self.ball_pos.y)), self.cfg.ball_radius)

        # Scores
        ltxt = self.font.render(f"L {self.left_score}", True, YELLOW)
        rtxt = self.font.render(f"R {self.right_score}", True, YELLOW)
        self.screen.blit(ltxt, (30, 15))
        self.screen.blit(rtxt, (self.cfg.width - 70, 15))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.cfg.render_fps)
        elif self.render_mode == "rgb_array":
            # Convert surface to (H, W, 3) numpy array for video capture.
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))
            return frame

    def close(self) -> None:
        pygame.quit()

    # ----------------------------
    # Internals
    # ----------------------------
    def _reset_ball(self, to_right: bool) -> None:
        self.ball_pos = pygame.Vector2(self.cfg.width / 2, self.cfg.height / 2)

        vx = self._rng.uniform(self.cfg.ball_speed_min_x, self.cfg.ball_speed_max_x)
        vy = self._rng.uniform(self.cfg.ball_speed_min_y, self.cfg.ball_speed_max_y)

        if not to_right:
            vx = -vx
        if self._rng.choice([True, False]):
            vy = -vy

        self.ball_vel = pygame.Vector2(vx, vy)

    def _apply_action(self, paddle: pygame.Rect, action: Action) -> None:
        if action == UP:
            paddle.y -= int(self.cfg.paddle_speed)
        elif action == DOWN:
            paddle.y += int(self.cfg.paddle_speed)

        paddle.y = max(0, min(self.cfg.height - self.cfg.paddle_h, paddle.y))

    def _bounce_from_paddle(self, paddle: pygame.Rect) -> None:
        self.ball_vel.x *= -1

        # English
        offset = (self.ball_pos.y - paddle.centery) / (self.cfg.paddle_h / 2)
        self.ball_vel.y += float(offset) * 1.2

        # Speed up and clamp
        self.ball_vel *= self.cfg.ball_speedup
        self.ball_vel.x = max(-self.cfg.max_ball_speed, min(self.cfg.max_ball_speed, self.ball_vel.x))
        self.ball_vel.y = max(-self.cfg.max_ball_speed, min(self.cfg.max_ball_speed, self.ball_vel.y))

        # Prevent stalling
        if abs(self.ball_vel.x) < 1.5:
            self.ball_vel.x = 1.5 if self.ball_vel.x >= 0 else -1.5

    def _get_obs(self) -> Tuple[float, ...]:
        bx = self.ball_pos.x / self.cfg.width
        by = self.ball_pos.y / self.cfg.height

        v_scale = self.cfg.max_ball_speed
        bvx = max(-1.0, min(1.0, self.ball_vel.x / v_scale))
        bvy = max(-1.0, min(1.0, self.ball_vel.y / v_scale))

        ly = self.left_paddle.centery / self.cfg.height
        ry = self.right_paddle.centery / self.cfg.height

        return (bx, by, bvx, bvy, ly, ry)

    def _info(self) -> Dict:
        return {"left_score": self.left_score, "right_score": self.right_score}


# ----------------------------
# Watch an "AI" play (demo)
# ----------------------------
def simple_tracking_policy(obs: Tuple[float, ...], is_left: bool) -> Action:
    """
    A simple scripted opponent to test your pipeline.
    This is NOT learning, just a baseline.
    """
    bx, by, bvx, bvy, ly, ry = obs
    paddle_y = ly if is_left else ry

    # If ball is above paddle center, go up; if below, go down
    if by < paddle_y - 0.02:
        return UP
    if by > paddle_y + 0.02:
        return DOWN
    return STAY


def play_demo(render: bool = True) -> None:
    env = PongEnv(render_mode="human" if render else None)
    obs, _ = env.reset()

    while True:
        if render:
            # Human controls: W/S for left paddle, Up/Down for right paddle.
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            left_action = STAY
            right_action = STAY

            if keys[pygame.K_w]:
                left_action = UP
            elif keys[pygame.K_s]:
                left_action = DOWN

            if keys[pygame.K_UP]:
                right_action = UP
            elif keys[pygame.K_DOWN]:
                right_action = DOWN
        else:
            # Headless demo fallback uses scripted tracking.
            left_action = simple_tracking_policy(obs, is_left=True)
            right_action = simple_tracking_policy(obs, is_left=False)

        obs, reward, done, info = env.step(left_action, right_action)
        if done:
            obs, _ = env.reset()


def play_human_vs_model(model_path: str, human_left: bool = True) -> None:
    env = PongEnv(render_mode="human")
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    obs, _ = env.reset()
    while True:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        human_action = STAY
        if keys[pygame.K_w]:
            human_action = UP
        elif keys[pygame.K_s]:
            human_action = DOWN

        model_action, _ = model.predict(obs, deterministic=True)
        if human_left:
            left_action = human_action
            right_action = int(model_action)
        else:
            left_action = int(model_action)
            right_action = human_action

        obs, _, done, _ = env.step(left_action, right_action)
        if done:
            obs, _ = env.reset()


def play_model_vs_model(left_path: str, right_path: str) -> None:
    env = PongEnv(render_mode="human")
    from stable_baselines3 import PPO

    left_model = PPO.load(left_path, device="cpu")
    right_model = PPO.load(right_path, device="cpu")
    obs, _ = env.reset()
    while True:
        left_action, _ = left_model.predict(obs, deterministic=True)
        right_action, _ = right_model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(int(left_action), int(right_action))
        if done:
            obs, _ = env.reset()


def _parse_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Pong demo modes.")
    parser.add_argument("--mode", choices=["demo", "human-vs-model", "model-vs-model"], default="demo")
    parser.add_argument("--left-model", type=str, default="models/ppo_pong_custom_latest.zip")
    parser.add_argument("--right-model", type=str, default="models/ppo_pong_custom_latest.zip")
    parser.add_argument("--human-right", action="store_true", help="Human controls right paddle.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    if args.mode == "human-vs-model":
        play_human_vs_model(args.right_model, human_left=not args.human_right)
    elif args.mode == "model-vs-model":
        play_model_vs_model(args.left_model, args.right_model)
    else:
        play_demo(render=True)
