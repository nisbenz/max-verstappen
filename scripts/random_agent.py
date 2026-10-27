import gymnasium as gym

env = gym.make("CarRacing-v3", render_mode="human")
obs, info = env.reset(seed=0)
total = 0.0
for step in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    total += reward
    if terminated or truncated:
        break
print(f"steps={step + 1} return={total:.1f}")
env.close()
