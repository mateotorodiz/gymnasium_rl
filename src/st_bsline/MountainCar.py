import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Create and wrap environment
def make_env():
    return Monitor(gym.make("MountainCarContinuous-v0"))

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)


# Instantiate PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=4096,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    learning_rate=3e-4,
    tensorboard_log="./ppo_mountaincar_tensorboard/",
    ent_coef= 0.05,
    clip_range=0.2
)


# Train
model.learn(total_timesteps=200_000)


# For evaluation and visualization, wrap the envs the same way (no normalization for rendering)
eval_env = DummyVecEnv([lambda: Monitor(gym.make("MountainCarContinuous-v0"))])
eval_env = VecNormalize.load("vecnormalize.pkl", eval_env) if hasattr(env, "save") else env

# Save normalization stats after training
if hasattr(env, "save"):
    env.save("vecnormalize.pkl")

# Evaluate
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

# Visualize
visual_env = gym.make("MountainCarContinuous-v0", render_mode="human")
obs, _ = visual_env.reset()
done, truncated = False, False
while not (done or truncated):
    # Use normalization for obs if needed
    norm_obs = obs
    if hasattr(env, "normalize_obs"):
        norm_obs = env.normalize_obs(obs)
    action, _ = model.predict(norm_obs, deterministic=True)
    obs, reward, done, truncated, info = visual_env.step(action)
    print(f"x={obs[0]:.3f}, reward={reward:.2f}")
    visual_env.render()

visual_env.close()
