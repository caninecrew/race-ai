import numpy as np

from pong import PongEnv, STAY, UP, DOWN


def test_reset_and_step_shapes():
    env = PongEnv(render_mode=None, seed=123)
    obs, info = env.reset()
    assert len(obs) == 6
    assert info["left_score"] == 0 and info["right_score"] == 0
    next_obs, reward, done, step_info = env.step(STAY, STAY)
    assert len(next_obs) == 6
    assert isinstance(reward, float)
    assert done is False
    assert 0.0 <= next_obs[0] <= 1.0  # normalized ball x
    env.close()


def test_seed_reproducibility():
    env1 = PongEnv(render_mode=None, seed=7)
    env2 = PongEnv(render_mode=None, seed=7)
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    assert np.allclose(obs1, obs2)
    step1 = env1.step(STAY, STAY)[0]
    step2 = env2.step(STAY, STAY)[0]
    assert np.allclose(step1, step2)
    env1.close()
    env2.close()


def test_paddles_stay_in_bounds():
    env = PongEnv(render_mode=None, seed=0)
    env.reset()
    for _ in range(100):
        env.step(UP, DOWN)
    assert env.left_paddle.top >= 0
    assert env.right_paddle.bottom <= env.cfg.height
    env.close()
