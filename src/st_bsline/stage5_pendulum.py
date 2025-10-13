import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import time

log_dir = "./sac_logs/"
os.makedirs(log_dir, exist_ok=True)

def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessing creation of environments.
    
    :param rank: Index of the subprocess
    :param seed: Base seed for the environment
    """
    def _init():
        env = gym.make("Pendulum-v1")
        env.reset(seed=seed + rank)
        # Create separate monitor file for each environment
        env = Monitor(env, os.path.join(log_dir, f"env_{rank}"))
        return env
    return _init

def train(env, name="SAC"):
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    start = time.time()
    model.learn(total_timesteps=100_000, log_interval=10)
    elapsed = time.time() - start
    print(f"{name} training completed in {elapsed:.2f}s")
    return model
if __name__ == "__main__":
    num_envs = 4
    seed = 42
    
    # Create vectorized environment with proper seeding
    envs = SubprocVecEnv([make_env(i, seed) for i in range(num_envs)])
    model = train(envs, f"SAC with {num_envs} parallel envs")
    
    # Close training environments
    envs.close()

    # Evaluation
    eval_env = gym.make("Pendulum-v1")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"Final evaluation - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Optional: Compare with single environment training
    print("\n" + "="*50)
    print("For comparison, training with single environment:")
    single_env = Monitor(gym.make("Pendulum-v1"), os.path.join(log_dir, "single_env"))
    single_model = train(single_env, "SAC with single env")
    single_mean, single_std = evaluate_policy(single_model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"Single env evaluation - Mean reward: {single_mean:.2f} ± {single_std:.2f}")
    single_env.close()
    
    print(f"\nPerformance comparison:")
    print(f"Multi-env ({num_envs}): {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Single env: {single_mean:.2f} ± {single_std:.2f}")
    improvement = ((mean_reward - single_mean) / abs(single_mean)) * 100
    print(f"Improvement: {improvement:.1f}%")


    obs, _ = eval_env.reset()
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = eval_env.step(action)
        eval_env.render()
        if done or truncated:
            break
    eval_env.close()
