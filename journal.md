# Journal

## Approach

- **Algorithm:** PPO (on-policy actor-critic) written from scratch in PyTorch, no RL library.
- **Input:** last 4 frames, grayscale, stacked → `4 × 96 × 96`. Grayscale because colour adds
  nothing the network needs (road vs grass is visible in brightness); 4 frames because one frame
  has no speed or direction of motion.
- **Network:** 3 conv layers (32/64/64 filters, 8×8/4×4/3×3, strides 4/2/1) → 512 fully connected
  → two heads: actor (a Beta distribution per action) and critic (V(s)).
- **Actions:** 2 numbers in [0, 1]: steering, and one throttle axis that becomes gas (positive part)
  or brake (negative part), so the car never presses both pedals. Converted to the official
  `[steer, gas, brake]` inside the submitted model.
- **Beta distribution, not Gaussian:** actions are bounded; a Beta lives on [0, 1], so no clipping
  and no probability mass wasted outside the valid range.
- **Action repeat 4:** one decision every 4 frames. Fewer decisions per lap → easier credit
  assignment and 4× less network inference.
- **Training tracks:** random seeds only. 12 validation seeds (1000–1011) choose `best.pt`;
  202 / 42 / 203 are logged but never used to choose, so their score stays honest. This is how I
  prepare for the mystery track.

## Log

### 2026-10-23/24 — Setup and v0

- uv project, Python 3.12, `gymnasium[box2d]`, PyTorch ROCm build for my AMD RX 6750 XT
  (needs `HSA_OVERRIDE_GFX_VERSION=10.3.0`, the card is not on ROCm's official list).
- v0: PPO as above, γ = 0.99, raw rewards, 2M agent steps (~3 h at 180 steps/s).
- Reward shaping in v0: −0.5 per step on grass (pixel check under the car), and the episode ends
  with −20 after 50 decisions without a new tile (stops wasting time on a car stuck on grass).
- **Result:** mean return ~840 in training, but the greedy driver mostly failed the test tracks
  (eval time 55–61 s with DNF = 70 s).

**Problems found in v0:**
1. **Value loss of 130–190.** Returns reach ~900, so the critic regressed targets in the hundreds.
   Its gradient used most of the gradient-clipping budget of the shared CNN, leaving little for the
   policy. → Divide rewards by a running std of the discounted return (like SB3's VecNormalize).
2. **Learning rate decayed to 0:** the last ~15% of training changed nothing (KL ~1e-4).
   → Decay to 10% of the start value instead.
3. **The car never used full gas (max 0.76).** The driver uses the Beta *mean*, which never reaches
   0 or 1, and full gas needed exactly 1. → Throttle gain 1/0.7: full gas from 0.85.
4. **Choosing the checkpoint on the test tracks** makes the test score look better than it is.
   → Separate validation seeds.

### 2026-10-24/25 — v1

- All 4 fixes above. Speed-ups: the training wrapper skips drawing the 3 frames of each action
  repeat the agent never sees (same physics and rewards); evaluation runs in 8 processes on CPU,
  like the organizer.
- 7.4M agent steps (1816 updates, stopped before the planned 1953), ~400 steps/s.
- **Result:** return ~890, validation mean lap ~30 s, test (`best.pt`): 202 = 13.8 s,
  42 = 15.3 s, 203 = 39.0 s.

### 2026-10-27 — Watching v1 drive, and what was wrong

I recorded a video and measured every action:
1. **Weaving on straights:** steering changes sign on 32–40% of decisions; fewer than 20% of
   decisions are near-straight.
2. **Seed 203 took 39 s instead of ~20 s.** The car cut a 105° hairpin and missed 2 of 360 tiles,
   then had to drive a whole second lap to touch them.

**Why one missed tile costs a whole lap** (read in the CarRacing source): the guidelines say a lap
counts at 95% of tiles, but that rule only fires when the start tile is touched *for the first
time*, and the car spawns on it. So in practice the lap ends only when **every** tile is visited.

**Why the agent didn't learn this:** with γ = 0.99 per decision (4 frames), the agent looks
~100 decisions = ~8 s ahead. The cost of a cut corner (−0.1 per frame for a ~15 s extra lap)
comes far beyond that horizon. And my "no new tile for 50 decisions" rule ended the second lap
after 4 s, hiding most of its cost.

### 2026-10-27 — v2: penalties for the bad habits (fine-tune of v1)

- Added: −10 for each tile left behind unvisited, −0.2 × |steering change| per decision, γ = 0.995.
- Faster training: 4 PPO epochs × 512 minibatch instead of 10 × 256 (update phase 4.2 s → 1.8 s).
- **Faster renderer, proven identical:** profiling showed ~89% of an env step is pygame drawing,
  only ~10% Box2D physics. I vectorized the "is this polygon on screen" test with numpy and cached
  the HUD font, keeping exactly the same pygame draw calls. `scripts/check_fast_render.py` runs 22
  episodes with the official and the fast renderer: 19,061 frames, 0 differences. Training went
  from ~540 to ~625 steps/s (the CPU, 6P+4E cores, caps the gain at 16 processes).
- 1.24M agent steps. Validation ~26 s (v1 ~30 s).
- **Results on the test tracks:**

  | | 202 | 42 | 203 | mean |
  |---|---|---|---|---|
  | v1, torch 1 thread | 20.9 s | 15.2 s | DNF | 35.4 s |
  | v2, torch 1 thread | 14.1 s | 15.5 s | 32.4 s | 20.7 s |
  | v1, default threads | 13.8 s | 15.3 s | 39.0 s | 22.7 s |
  | v2, default threads | 13.7 s | 15.1 s | 39.1 s | 22.6 s |

**Lessons from v2:**
- **The same model drives differently with a different number of CPU threads.** Float rounding
  changes the actions slightly, and a slightly different line can mean a missed tile. v2 is much
  more robust than v1 to this.
- **My missed-tile rule had a bug:** a tile only counted as missed once the car was 8 tiles past it,
  and the counter stopped at the last tile, so the final corner before the finish was never
  checked. v2 learned to cut exactly that corner on 203.
- **The steering penalty did not stop the weaving** (flips still 0.32–0.44).
- Conclusion: hand-written reward rules are fragile and each one can be exploited.

### 2026-10-27 — v3: back to the env reward only

- All shaping removed: no grass penalty, no early episode end, no missed-tile or steering penalty.
  The env reward is +1000/N per new tile and −0.1 per frame: when every tile is visited, the return
  is `1000 − 0.1 × frames`, i.e. exactly the lap time the competition scores.
- γ = 0.998: horizon ~500 decisions ≈ 40 s, longer than a lap, so a cut corner (extra lap) and
  weaving (slower lap) are both visible as lost reward. The agent should drive straight and take
  every tile because it is faster, not because a rule says so.
- Fine-tune from v2 `best.pt`, 45 min, 370 updates (1.52M agent steps).
- **Result: no clear win.** `best.pt` (update 180) scored 19.6 s on the 12 validation seeds, the
  best so far, but on the test tracks it cut a corner on 202 and did not finish 203:

  | | 202 | 42 | 203 | mean |
  |---|---|---|---|---|
  | v3, torch 1 thread | 19.8 s | 14.7 s | DNF | 34.8 s |
  | v3, default threads | 20.7 s | 15.6 s | DNF | 35.4 s |

- To check whether that 19.6 s was luck, I evaluated every candidate on **50 new random tracks**
  (seeds 2000–2049, never used to choose anything), the best proxy for the mystery track:

  | | mean lap | median | DNF | laps > 30 s | mean return |
  |---|---|---|---|---|---|
  | v1 best | 25.8 s | 20.8 s | 3/50 | 8/50 | 864 |
  | v2 best | 23.6 s | 16.2 s | 3/50 | 5/50 | 872 |
  | v3 best | 22.2 s | 16.3 s | 2/50 | 6/50 | 890 |
  | v3 last | 25.6 s | 17.8 s | 5/50 | 4/50 | 877 |

- **Lessons:** 12 validation tracks are too few to pick a checkpoint: one lucky evaluation wins.
  Steering flips are unchanged (0.35–0.43): 45 minutes of the env reward did not remove the
  weaving either. Each version improves the average track a little, but every model still fails
  some sharp corners, and on a fixed track one failure costs a whole lap or a DNF.
- **Submission choice: v2 `best.pt`.** It is the best on the three known tracks (22.6 s vs 35.4 s
  for v3) and within ~1.4 s of v3 on the 50-track average, where the difference is about one DNF.

## Things I learned

- **Watch the agent drive.** The return (~890) looked fine; the video showed weaving and a cut
  hairpin. Measuring the actions told me what was wrong.
- **Read the environment's source.** The 95% lap rule from the docs never fires in practice.
- **γ is a horizon.** 1/(1−γ) decisions; with action repeat 4, γ = 0.99 is only ~8 s of driving.
  γ = 0 would only care about the next reward; γ = 0.999 looks ~80 s ahead but makes value
  estimates noisier and learning slower.
- **Reward shaping is a trade-off.** Each rule fixes one behaviour and opens a loophole elsewhere.
- **Profile before optimizing.** I assumed Box2D was the bottleneck; it was the renderer.
- **Evaluation noise is large:** one missed tile on one of 12 validation seeds moves the mean by
  ~1.3 s. Don't over-read a single evaluation.

## If I had one more day

- Train v3 longer and on more seeds, and evaluate on many random seeds to estimate mystery-track
  performance.
- Make the driver robust to thread-count rounding (e.g. train with small observation noise).
- Try action repeat 2 for finer steering on straights.
