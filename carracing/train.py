"""PPO training loop.

    uv run python -m carracing.train --run-name v0
    uv run python -m carracing.train --run-name smoke --total-steps 4096 --eval-every 1
    uv run python -m carracing.train --run-name v0 --resume   # continue a stopped run

Stopping: press Ctrl+C once (or `kill <pid>`). The current update finishes,
runs/<run>/checkpoint.pt is written, and the process exits. It is also written
every --save-every updates, so a crash or power cut loses only a few minutes.

One update:
  1. collect n_envs * n_steps transitions with the current policy (random tracks)
  2. compute advantages with GAE
  3. run `epochs` passes of minibatch gradient steps on the clipped PPO loss
Every `eval_every` updates, the greedy driver is evaluated with the official
protocol on the validation seeds (VAL_SEEDS) and on the test seeds 202/42/203.
The best one on the validation seeds is saved to runs/<run>/best.pt: choosing it
on the test seeds would make the test score look better than it is.
"""
import argparse
import copy
import csv
import os
import signal
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
from carracing.evaluate import TEST_SEEDS, VAL_SEEDS, evaluate_parallel, make_eval_pool, score
from carracing.model import ActorCritic
from carracing.wrappers import ACTION_REPEAT, make_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=8_000_000, help="agent steps (x4 frames)")
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=256, help="rollout length per env")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-floor", type=float, default=0.1, help="final lr, as a fraction of --lr")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    # Reward shaping: off by default (env reward only). v1 used grass 0.5 and
    # patience 50; v2 added missed-tile 10 and steer 0.2.
    p.add_argument("--grass-penalty", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=0,
                   help="end the episode after this many agent steps without a new tile (0 = off)")
    p.add_argument("--missed-tile-penalty", type=float, default=0.0)
    p.add_argument("--steer-penalty", type=float, default=0.0, help="per unit of steering change")
    p.add_argument("--max-minutes", type=float, default=None,
                   help="stop cleanly (as with Ctrl+C) after this much wall-clock time")
    p.add_argument("--eval-every", type=int, default=25, help="updates between evaluations")
    p.add_argument("--eval-workers", type=int, default=8)
    p.add_argument("--init", default=None, help="start from this checkpoint (a saved Driver)")
    p.add_argument("--save-every", type=int, default=10, help="updates between checkpoint.pt saves")
    p.add_argument("--resume", action="store_true",
                   help="continue runs/<run-name>/checkpoint.pt with its saved settings")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# Settings that may differ when resuming; every other one comes from the checkpoint,
# so the resumed run is the same experiment (same lr schedule, batch size, ...).
MACHINE_ARGS = ("run_name", "resume", "device", "eval_workers", "max_minutes")


def make_train_env(**wrapper_kwargs) -> gym.Env:
    # Runs in the env worker process. Ctrl+C reaches every process of the
    # terminal; the workers ignore it so the main process can stop cleanly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    return make_env(**wrapper_kwargs)


def save_checkpoint(path: Path, state: dict):
    # Written to a temp file first: a crash mid-write never corrupts the last good checkpoint.
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


class RewardScaler:
    """Divides rewards by the running std of the discounted return (like SB3's VecNormalize).

    Raw returns reach ~900, so V(s) targets were in the hundreds and the value
    loss (100-700) took most of the gradient-clipping budget of the shared CNN,
    leaving little for the policy. After scaling, returns are ~1.
    """

    def __init__(self, n_envs: int, gamma: float, clip: float = 10.0):
        self.gamma, self.clip = gamma, clip
        self.ret = np.zeros(n_envs)
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        self.ret = self.ret * self.gamma + reward
        # Running variance of the return, merged batch by batch (Chan et al.).
        b_mean, b_var, b_count = self.ret.mean(), self.ret.var(), len(self.ret)
        delta, total = b_mean - self.mean, self.count + b_count
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / total
        self.mean, self.var, self.count = self.mean + delta * b_count / total, m2 / total, total
        self.ret[done] = 0.0
        return np.clip(reward / np.sqrt(self.var + 1e-8), -self.clip, self.clip)


def main():
    args = parse_args()
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint.pt"
    ckpt = None
    if args.resume:
        ckpt = torch.load(ckpt_path, weights_only=False)
        for key, value in ckpt["args"].items():
            if key not in MACHINE_ARGS:
                setattr(args, key, value)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Each env draws a new random track at every reset. 202/42/203 are only
    # used for evaluation.
    wrapper_kwargs = dict(
        grass_penalty=args.grass_penalty,
        patience=args.patience,
        missed_tile_penalty=args.missed_tile_penalty,
        steer_penalty=args.steer_penalty,
    )
    env_fns = [lambda: make_train_env(**wrapper_kwargs) for _ in range(args.n_envs)]
    # SAME_STEP: when an episode ends, step() already returns the first obs of
    # the next one and puts the last obs of the old one in info["final_obs"].
    envs = gym.vector.AsyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    # Spawned before the env workers and the GPU, so it costs nothing until used.
    eval_pool = make_eval_pool(args.eval_workers)

    policy = ActorCritic()
    if args.init:
        policy.load_state_dict(torch.load(args.init, weights_only=False).policy.state_dict())
    policy = policy.to(device)
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
    reward_scaler = RewardScaler(N, args.gamma)
    recent_returns = deque(maxlen=20)
    recent_missed = deque(maxlen=20)  # tiles skipped per episode (not saved in checkpoints)
    laps_finished = 0
    best = (float("inf"), float("inf"))  # validation (mean lap time, -mean return)
    global_step = 0
    first_update = 1

    if ckpt is not None:
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        reward_scaler.__dict__.update(ckpt["reward_scaler"])
        recent_returns.extend(ckpt["recent_returns"])
        laps_finished, best = ckpt["laps_finished"], tuple(ckpt["best"])
        global_step, first_update = ckpt["global_step"], ckpt["update"] + 1
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        print(f"resumed {ckpt_path} at update {ckpt['update']}/{n_updates}", flush=True)

    # Episodes that were running at the stop are lost (env state is not saved):
    # resumed runs start fresh episodes, on tracks the first session did not draw.
    next_obs, _ = envs.reset(seed=args.seed + 1000 * (first_update - 1))
    next_obs = torch.from_numpy(next_obs)
    start, start_step = time.time(), global_step

    # Ctrl+C / kill: finish the current update, save, then exit.
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("\nstop requested: finishing this update, then saving checkpoint.pt", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    log_file = open(run_dir / "log.csv", "a" if ckpt is not None else "w", newline="")
    log = csv.writer(log_file)
    if ckpt is None:
        log.writerow(["update", "agent_steps", "frames", "mean_return_20", "laps", "policy_loss",
                      "value_loss", "entropy", "approx_kl", "clip_frac", "lr", "sps",
                      "rollout_s", "update_s", "val_time", "val_return", "test_time",
                      "test_return"])

    for update in range(first_update, n_updates + 1):
        # Learning rate decays linearly to lr_floor * lr. Decaying to 0 froze
        # the last ~15% of v0 (KL ~1e-4): those updates changed nothing.
        progress = (update - 1) / n_updates
        lr = args.lr * (1.0 - (1.0 - args.lr_floor) * progress)
        optimizer.param_groups[0]["lr"] = lr

        # ---- 1. Rollout ----------------------------------------------------
        t_rollout = time.time()
        policy.eval()
        for t in range(T):
            obs_buf[t] = next_obs
            with torch.no_grad():
                action, logp, _, value = policy.get_action_and_value(next_obs.to(device))
            act_buf[t], logp_buf[t], val_buf[t] = action.cpu(), logp.cpu(), value.cpu()

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            next_obs = torch.from_numpy(next_obs)
            done = terminated | truncated
            reward = torch.as_tensor(reward_scaler(reward, done), dtype=torch.float32)
            global_step += N

            # Truncated (time limit) is not a real end: the car could have kept
            # driving. Bootstrap: add the critic's estimate of what would have
            # followed, gamma * V(final obs), then treat the step as done.
            # V is in scaled units, like the reward it is added to.
            if truncated.any():
                final_obs = np.stack([info["final_obs"][i] for i in np.where(truncated)[0]])
                with torch.no_grad():
                    final_v = policy.get_value(torch.from_numpy(final_obs).to(device)).cpu()
                reward[torch.from_numpy(truncated)] += args.gamma * final_v
            rew_buf[t] = reward
            done_buf[t] = torch.from_numpy(done).float()

            if "final_info" in info:
                final = info["final_info"]
                for i in np.where(final.get("_episode_raw_return", []))[0]:
                    recent_returns.append(final["episode_raw_return"][i])
                    recent_missed.append(final["episode_missed_tiles"][i])
                laps_finished += int(np.sum(final.get("lap_finished", [])))

        rollout_s = time.time() - t_rollout

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

        t_update = time.time()
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

        update_s = time.time() - t_update

        if args.max_minutes is not None and time.time() - start > args.max_minutes * 60:
            stop_requested = True
            print(f"--max-minutes {args.max_minutes} reached", flush=True)

        # ---- Logging, evaluation, checkpoints ------------------------------
        sps = (global_step - start_step) / (time.time() - start)
        mean_ret = float(np.mean(recent_returns)) if recent_returns else float("nan")
        mean_missed = float(np.mean(recent_missed)) if recent_missed else float("nan")
        val_time = val_return = test_time = test_return = ""
        # Also on a stop, so best.pt / last.pt include the newest policy.
        if update % args.eval_every == 0 or update == n_updates or stop_requested:
            results = evaluate_parallel(eval_pool, policy, VAL_SEEDS + TEST_SEEDS)
            v_time, v_return = score({s: results[s] for s in VAL_SEEDS})
            t_time, t_return = score({s: results[s] for s in TEST_SEEDS})
            val_time, val_return = f"{v_time:.2f}", f"{v_return:.1f}"
            test_time, test_return = f"{t_time:.2f}", f"{t_return:.1f}"
            # Saved on CPU so the organizer can load it without a GPU. A fresh
            # Driver starts with an empty frame buffer.
            cpu_driver = Driver(copy.deepcopy(policy).cpu()).eval()
            if (v_time, -v_return) < best:
                best = (v_time, -v_return)
                torch.save(cpu_driver, run_dir / "best.pt")
            torch.save(cpu_driver, run_dir / "last.pt")

        log.writerow([update, global_step, global_step * ACTION_REPEAT, f"{mean_ret:.1f}",
                      laps_finished, f"{pg_loss.item():.4f}", f"{v_loss.item():.3f}",
                      f"{ent.item():.3f}", f"{np.mean(kls):.4f}", f"{np.mean(clip_fracs):.3f}",
                      f"{lr:.2e}", f"{sps:.0f}", f"{rollout_s:.1f}", f"{update_s:.1f}",
                      val_time, val_return, test_time, test_return])
        log_file.flush()
        print(f"upd {update}/{n_updates} steps {global_step} ret20 {mean_ret:7.1f} "
              f"missed {mean_missed:4.1f} "
              f"laps {laps_finished} vloss {v_loss.item():8.2f} ent {ent.item():6.3f} "
              f"kl {np.mean(kls):.4f} sps {sps:.0f} (rollout {rollout_s:.1f}s update {update_s:.1f}s)"
              + (f" | val {val_time}s {val_return} test {test_time}s {test_return}"
                 if val_time else ""), flush=True)

        if update % args.save_every == 0 or update == n_updates or stop_requested:
            save_checkpoint(ckpt_path, {
                "update": update,
                "global_step": global_step,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "reward_scaler": dict(reward_scaler.__dict__),
                "recent_returns": list(recent_returns),
                "laps_finished": laps_finished,
                "best": best,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "args": vars(args),
            })
        if stop_requested:
            print(f"saved {ckpt_path} at update {update}/{n_updates}. Resume with:\n"
                  f"  uv run python -m carracing.train --run-name {args.run_name} --resume",
                  flush=True)
            break

    # After a stop, terminate the workers instead of asking them to close: a
    # `pkill -f` matching this command line may already have killed them.
    envs.close(terminate=stop_requested)
    eval_pool.close()
    log_file.close()


if __name__ == "__main__":
    main()
