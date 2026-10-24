"""PPO training loop.

    uv run python -m carracing.train --run-name v0
    uv run python -m carracing.train --run-name smoke --total-steps 4096 --eval-every 1

One update:
  1. collect n_envs * n_steps transitions with the current policy (random tracks)
  2. compute advantages with GAE
  3. run `epochs` passes of minibatch gradient steps on the clipped PPO loss
Every `eval_every` updates, the greedy driver is evaluated on seeds 202/42/203
with the official protocol; the best one is saved to runs/<run>/best.pt.
"""
import argparse
import copy
import csv
import os
import time
from collections import deque
from pathlib import Path

# The RX 6700/6750 XT (gfx1031) is not on ROCm's official list, but runs the
# gfx1030 kernels fine. Must be set before torch initializes the GPU.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from carracing.driver import Driver
from carracing.evaluate import evaluate, score
from carracing.model import ActorCritic
from carracing.wrappers import ACTION_REPEAT, make_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=2_000_000, help="agent steps (x4 frames)")
    p.add_argument("--n-envs", type=int, default=12)
    p.add_argument("--n-steps", type=int, default=256, help="rollout length per env")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--grass-penalty", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=50, help="agent steps without a new tile")
    p.add_argument("--eval-every", type=int, default=25, help="updates between evaluations")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Each env draws a new random track at every reset. 202/42/203 are only
    # used for evaluation.
    env_fns = [
        lambda: make_env(grass_penalty=args.grass_penalty, patience=args.patience)
        for _ in range(args.n_envs)
    ]
    # SAME_STEP: when an episode ends, step() already returns the first obs of
    # the next one and puts the last obs of the old one in info["final_obs"].
    envs = gym.vector.AsyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    policy = ActorCritic().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    T, N = args.n_steps, args.n_envs
    obs_buf = torch.zeros((T, N, 4, 96, 96), dtype=torch.uint8)
    act_buf = torch.zeros((T, N, 2))
    logp_buf = torch.zeros((T, N))
    rew_buf = torch.zeros((T, N))
    done_buf = torch.zeros((T, N))
    val_buf = torch.zeros((T, N))

    batch_size = T * N
    n_updates = args.total_steps // batch_size
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.from_numpy(next_obs)
    recent_returns = deque(maxlen=20)
    laps_finished = 0
    best = (float("inf"), -float("inf"))  # (mean lap time, -mean return)
    global_step = 0
    start = time.time()

    log_file = open(run_dir / "log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["update", "agent_steps", "frames", "mean_return_20", "laps", "policy_loss",
                  "value_loss", "entropy", "approx_kl", "clip_frac", "lr", "sps",
                  "eval_time", "eval_return"])

    for update in range(1, n_updates + 1):
        # Learning rate decays linearly to 0 over training.
        lr = args.lr * (1.0 - (update - 1) / n_updates)
        optimizer.param_groups[0]["lr"] = lr

        # ---- 1. Rollout ----------------------------------------------------
        policy.eval()
        for t in range(T):
            obs_buf[t] = next_obs
            with torch.no_grad():
                action, logp, _, value = policy.get_action_and_value(next_obs.to(device))
            act_buf[t], logp_buf[t], val_buf[t] = action.cpu(), logp.cpu(), value.cpu()

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            next_obs = torch.from_numpy(next_obs)
            reward = torch.as_tensor(reward, dtype=torch.float32)
            global_step += N

            # Truncated (time limit) is not a real end: the car could have kept
            # driving. Bootstrap: add the critic's estimate of what would have
            # followed, gamma * V(final obs), then treat the step as done.
            if truncated.any():
                final_obs = np.stack([info["final_obs"][i] for i in np.where(truncated)[0]])
                with torch.no_grad():
                    final_v = policy.get_value(torch.from_numpy(final_obs).to(device)).cpu()
                reward[torch.from_numpy(truncated)] += args.gamma * final_v
            rew_buf[t] = reward
            done_buf[t] = torch.from_numpy(terminated | truncated).float()

            if "final_info" in info:
                final = info["final_info"]
                for i in np.where(final.get("_episode_raw_return", []))[0]:
                    recent_returns.append(final["episode_raw_return"][i])
                laps_finished += int(np.sum(final.get("lap_finished", [])))

        # ---- 2. GAE --------------------------------------------------------
        # delta_t = r_t + gamma * V(s_t+1) * (1 - done_t) - V(s_t)
        # A_t     = delta_t + gamma * lambda * (1 - done_t) * A_t+1
        with torch.no_grad():
            next_value = policy.get_value(next_obs.to(device)).cpu()
        advantages = torch.zeros((T, N))
        last_gae = torch.zeros(N)
        for t in reversed(range(T)):
            next_v = next_value if t == T - 1 else val_buf[t + 1]
            not_done = 1.0 - done_buf[t]
            delta = rew_buf[t] + args.gamma * next_v * not_done - val_buf[t]
            last_gae = delta + args.gamma * args.gae_lambda * not_done * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf  # the critic's regression target G

        # ---- 3. PPO update -------------------------------------------------
        # The whole batch goes to the GPU once (~110 MB of uint8 frames).
        b_obs = obs_buf.reshape(batch_size, 4, 96, 96).to(device)
        b_act = act_buf.reshape(batch_size, 2).to(device)
        b_logp = logp_buf.reshape(batch_size).to(device)
        b_adv = advantages.reshape(batch_size).to(device)
        b_ret = returns.reshape(batch_size).to(device)

        policy.train()
        clip_fracs, kls = [], []
        for _ in range(args.epochs):
            perm = torch.randperm(batch_size, device=device)
            for start_i in range(0, batch_size, args.minibatch_size):
                idx = perm[start_i : start_i + args.minibatch_size]
                _, new_logp, entropy, new_value = policy.get_action_and_value(
                    b_obs[idx], b_act[idx]
                )
                adv = b_adv[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # ratio = pi_new(a|s) / pi_old(a|s), computed in log space.
                log_ratio = new_logp - b_logp[idx]
                ratio = log_ratio.exp()
                # Clipped objective: once the ratio leaves [1-clip, 1+clip],
                # pushing it further brings no extra gain, so no gradient.
                pg_loss = -torch.min(
                    ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv
                ).mean()
                v_loss = ((new_value - b_ret[idx]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    kls.append(((ratio - 1) - log_ratio).mean().item())
                    clip_fracs.append(((ratio - 1).abs() > args.clip).float().mean().item())

        # ---- Logging, evaluation, checkpoints ------------------------------
        sps = global_step / (time.time() - start)
        mean_ret = float(np.mean(recent_returns)) if recent_returns else float("nan")
        eval_time = eval_return = ""
        if update % args.eval_every == 0 or update == n_updates:
            driver = Driver(policy).eval()
            mean_time, mean_return = score(evaluate(driver))
            eval_time, eval_return = f"{mean_time:.2f}", f"{mean_return:.1f}"
            # Saved files must start with an empty frame buffer, otherwise the
            # first official run would see frames left over from this eval.
            # Saved on CPU so the organizer can load it without a GPU.
            driver.reset()
            cpu_driver = copy.deepcopy(driver).cpu()
            if (mean_time, -mean_return) < best:
                best = (mean_time, -mean_return)
                torch.save(cpu_driver, run_dir / "best.pt")
            torch.save(cpu_driver, run_dir / "last.pt")

        log.writerow([update, global_step, global_step * ACTION_REPEAT, f"{mean_ret:.1f}",
                      laps_finished, f"{pg_loss.item():.4f}", f"{v_loss.item():.3f}",
                      f"{ent.item():.3f}", f"{np.mean(kls):.4f}", f"{np.mean(clip_fracs):.3f}",
                      f"{lr:.2e}", f"{sps:.0f}", eval_time, eval_return])
        log_file.flush()
        print(f"upd {update}/{n_updates} steps {global_step} ret20 {mean_ret:7.1f} "
              f"laps {laps_finished} vloss {v_loss.item():8.2f} ent {ent.item():6.3f} "
              f"kl {np.mean(kls):.4f} sps {sps:.0f}"
              + (f" | eval time {eval_time}s return {eval_return}" if eval_time else ""))

    envs.close()
    log_file.close()


if __name__ == "__main__":
    main()
