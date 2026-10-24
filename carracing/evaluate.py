"""Evaluate a submitted model the way the organizer does (Appendix A).

    uv run python -m carracing.evaluate runs/<run>/best.pt
    uv run python -m carracing.evaluate runs/<run>/best.pt --render
"""
import argparse

import gymnasium as gym
import numpy as np
import torch

from carracing.driver import Driver

TEST_SEEDS = (202, 42, 203)
DNF_TIME = 70.0


def run_lap(driver: Driver, seed: int, max_steps: int = 3000, render: bool = False):
    """One run of the official protocol. Returns (lap_time or None, raw return)."""
    env = gym.make(
        "CarRacing-v3", max_episode_steps=max_steps, render_mode="human" if render else None
    )
    obs, _ = env.reset(seed=seed)
    driver.reset()
    done = False
    info = {}
    total = 0.0
    while not done:
        action = driver(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
    lap_time = env.unwrapped.t if info.get("lap_finished", False) else None
    env.close()
    return lap_time, total


def evaluate(driver: Driver, seeds=TEST_SEEDS, render: bool = False) -> dict:
    results = {}
    for seed in seeds:
        lap_time, total = run_lap(driver, seed, render=render)
        results[seed] = (lap_time, total)
    return results


def score(results: dict) -> tuple[float, float]:
    """(mean lap time with DNF = 70 s, mean raw return). Lower time is better."""
    times = [DNF_TIME if t is None else t for t, _ in results.values()]
    returns = [r for _, r in results.values()]
    return float(np.mean(times)), float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TEST_SEEDS))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Same call as the official script, plus weights_only=False: PyTorch >= 2.6
    # refuses to unpickle a full model without it.
    model = torch.load(args.model_path, weights_only=False)
    model.eval()

    results = evaluate(model, args.seeds, args.render)
    for seed, (lap_time, total) in results.items():
        shown = f"{lap_time:.2f}s" if lap_time is not None else "DNF"
        print(f"seed {seed:>4}: lap {shown:>8}  return {total:7.1f}")
    mean_time, mean_return = score(results)
    print(f"mean time (DNF=70s): {mean_time:.2f}s  mean return: {mean_return:.1f}")


if __name__ == "__main__":
    main()
