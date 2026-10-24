# CarRacing AI Challenge

Training an AI driver for Gymnasium `CarRacing-v3` to complete laps as fast as possible.

See [journal.md](journal.md) for the development log.

## Approach (draft v0)

- **Input:** the last 4 frames, grayscale, stacked → `4 × 96 × 96`.
- **Network:** CNN (3 conv layers + 512 fully connected), with two heads:
  - **actor:** one Beta distribution per action;
  - **critic:** V(s), the expected discounted return.
- **Algorithm:** PPO (clip 0.2, GAE λ = 0.95, γ = 0.99) with Adam.
  Training uses 8 parallel envs on random tracks; 202 / 42 / 203 are used only for evaluation.
- **Action repeat:** each decision is held for 4 frames.
- **Reward shaping (training only):**
  - −0.5 per step when the car is on grass;
  - the episode ends with −20 after 50 decisions with no new tile.

| File | Role |
| --- | --- |
| `carracing/wrappers.py` | preprocessing, action repeat, frame stack, reward shaping |
| `carracing/model.py` | CNN actor-critic with Beta heads |
| `carracing/train.py` | PPO loop, logging (`runs/<run>/log.csv`), checkpoints |
| `carracing/driver.py` | the submitted model: raw frame in, `[steer, gas, brake]` out |
| `carracing/evaluate.py` | runs the official protocol on 202 / 42 / 203 |

## Usage

```bash
uv sync
uv run python -m carracing.train --run-name v0          # ~2M agent steps
uv run python -m carracing.evaluate runs/v0/best.pt     # official protocol
uv run python -m carracing.evaluate runs/v0/best.pt --render --seeds 202
```

`pyproject.toml` pins the ROCm 7.2 build of PyTorch, for training on an AMD RX 6750 XT
(`train.py` sets `HSA_OVERRIDE_GFX_VERSION=10.3.0` for it). Evaluation needs no GPU:
checkpoints are saved on CPU, and any PyTorch ≥ 2.x build can run them.

## Actions: how the model's output becomes a valid env action

The policy outputs **2** numbers in `[0, 1]` (Beta samples), not 3:

| Policy output | Mapping | Env action |
| --- | --- | --- |
| `a[0]` | `steer = 2·a[0] − 1` | steering in `[-1, 1]` |
| `a[1]` | `throttle = 2·a[1] − 1` | `gas = max(throttle, 0)`, `brake = max(−throttle, 0)` |

Gas and brake share one axis, so the car can never press both pedals at once.

**Is this allowed?** Yes. Section 6 of the guidelines requires that the env receives continuous `[steering, gas, brake]` with steering in `[-1, 1]` and gas and brake in `[0, 1]`. It explicitly allows a different internal action format, as long as it is converted at inference ("If you trained with discrete actions, convert them to the continuous format during inference"). Our conversion is a fixed formula, not a driving rule: every decision still comes from the network. It happens inside the submitted model (`Driver`), so the organizer always receives a valid 3-value action.

## How the organizer runs the submitted model

`best.pt` is a full `Driver` module. It does all the preprocessing itself: grayscale, the 4-frame stack, the 4-frame action repeat and the action mapping. The official loop works unchanged, with `action = model(obs)` on the raw `(96, 96, 3)` frame:

```python
import sys; sys.path.insert(0, "path/to/this/repo")   # so the Driver class can be unpickled
import gymnasium as gym, torch

model = torch.load("best.pt", weights_only=False)
model.eval()

env = gym.make("CarRacing-v3", max_episode_steps=3000)
obs, _ = env.reset(seed=202)
done, info = False, {}
while not done:
    action = model(obs)                    # numpy [steer, gas, brake]
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
lap_time = env.unwrapped.t if info.get("lap_finished", False) else None
```

Notes for the organizer:

- **`weights_only=False` is required** with PyTorch ≥ 2.6. Since 2.6, `torch.load` defaults to `weights_only=True` and refuses to load any full model, from any participant.
- **The model is stateful** (it keeps the last 4 frames). Load it fresh for each run, as the reference script does, or call `model.reset()` before a new run.
- The model only uses the image it receives. It never reads `env.unwrapped`.
