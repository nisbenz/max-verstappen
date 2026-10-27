"""Evaluate a submitted model the way the organizer does (Appendix A).

    uv run python -m carracing.evaluate runs/<run>/best.pt
    uv run python -m carracing.evaluate runs/<run>/best.pt --render
"""
import argparse
import multiprocessing as mp
import signal

import gymnasium as gym
import numpy as np
import torch

from carracing.driver import Driver

TEST_SEEDS = (202, 42, 203)
# Checkpoints are chosen on these, so the test seeds stay unseen until the end.
VAL_SEEDS = tuple(range(1000, 1012))
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


def _init_eval_worker():
    # One episode per process: more CPU threads per process only fight each other.
    torch.set_num_threads(1)
    # Ctrl+C is handled by the training process, which stops cleanly; the pool must survive it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_seed(job):
    state_dict, seed = job
    driver = Driver()
    driver.policy.load_state_dict(state_dict)
    driver.eval()
    return seed, run_lap(driver, seed)


def make_eval_pool(n_workers: int):
    """Process pool for `evaluate_parallel`. spawn, not fork: the parent holds a GPU context."""
    return mp.get_context("spawn").Pool(n_workers, initializer=_init_eval_worker)


def evaluate_parallel(pool, policy: torch.nn.Module, seeds) -> dict:
    """Same result as `evaluate(Driver(policy), seeds)`, one episode per process, on CPU
    like the organizer (GPU and CPU round differently, which changes the laps)."""
    state_dict = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    return dict(pool.map(_run_seed, [(state_dict, s) for s in seeds]))


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
