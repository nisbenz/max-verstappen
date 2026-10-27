"""Check that carracing.fast_render gives exactly the official CarRacing-v3 output.

    uv run python scripts/check_fast_render.py [--model runs/v1/best.pt]

Every episode runs twice in fresh processes: once with the official renderer,
once with the fast one, from the same seed with the same actions. Every
observation (96x96x3 frame), reward and termination flag must be identical.
Episodes are driven by a trained model (on-track driving, tiles changing
colour) and by random actions (grass, spins, the track edge).
"""
import argparse
import hashlib
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_episode(job):
    seed, driver_kind, fast, model_path, max_steps = job
    import gymnasium as gym
    import torch

    torch.set_num_threads(1)
    env = gym.make("CarRacing-v3", max_episode_steps=max_steps)
    if fast:
        from carracing.fast_render import install

        install(env.unwrapped)
    rng = np.random.default_rng(seed)
    driver = None
    if driver_kind == "model":
        driver = torch.load(model_path, weights_only=False)
        driver.eval()
    obs, _ = env.reset(seed=seed)
    trace = [hashlib.sha1(obs.tobytes()).hexdigest()]
    action = np.zeros(3, dtype=np.float32)
    done = False
    t0 = time.perf_counter()
    while not done:
        if driver is not None:
            action = driver(obs)
        elif rng.random() < 0.1:  # random actions, held for a few frames
            action = np.array([rng.uniform(-1, 1), rng.uniform(0, 1), rng.uniform(0, 0.3)],
                              dtype=np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        trace.append(f"{hashlib.sha1(obs.tobytes()).hexdigest()} {reward!r} {terminated} {truncated}")
        done = terminated or truncated
    return trace, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/v1/best.pt")
    p.add_argument("--workers", type=int, default=14)
    args = p.parse_args()

    episodes = [(s, "model", 3000) for s in (202, 42, 203, 1000, 1001, 1002, 7, 8, 9, 10)]
    episodes += [(s, "random", 600) for s in range(100, 112)]
    jobs = [(s, kind, fast, args.model, n) for s, kind, n in episodes for fast in (False, True)]
    # One process per episode: the official runs never share a process with the
    # patch (its font cache is process-wide).
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        results = pool.map(run_episode, jobs)

    frames = mismatches = 0
    t_official = t_fast = 0.0
    for i, (seed, kind, _) in enumerate(episodes):
        (official, t_o), (fast, t_f) = results[2 * i], results[2 * i + 1]
        t_official += t_o
        t_fast += t_f
        frames += len(official)
        bad = [k for k, (a, b) in enumerate(zip(official, fast)) if a != b]
        if len(official) != len(fast):
            bad.append(min(len(official), len(fast)))
        mismatches += len(bad)
        status = "identical" if not bad else f"MISMATCH at steps {bad[:5]}"
        print(f"seed {seed:>4} ({kind:6s}): {len(official):5d} frames {status}")
    print(f"\n{frames} frames compared, {mismatches} mismatches")
    print(f"env time (contended, all renders): official {t_official:.1f}s  fast {t_fast:.1f}s  "
          f"({t_official / t_fast:.2f}x)")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
