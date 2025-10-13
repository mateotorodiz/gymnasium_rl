import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from functools import partial
import os
import time

log_dir = "./sac_logs/"
os.makedirs(log_dir, exist_ok=True)

def make_env(rank):
    """Simple environment factory - no nested functions needed with partial"""
    env = gym.make("Pendulum-v1")
    # ONLY change: separate monitor files to avoid conflicts
    env = Monitor(env, os.path.join(log_dir, f"env_{rank}"))
    return env

def train(env, name="SAC"):
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    start = time.time()
    model.learn(total_timesteps=100_000, log_interval=10)
    elapsed = time.time() - start
    print(f"{name} training completed in {elapsed:.2f}s")
    return model

if __name__ == "__main__":
    num_envs = 4
    
    # Create vectorized environment using partial (cleaner than nested functions)
    envs = SubprocVecEnv([partial(make_env, i) for i in range(num_envs)])
    model = train(envs, f"SAC with {num_envs} parallel envs (optimized)")
    
    # Close training environments
    envs.close()

    # Evaluation
    eval_env = gym.make("Pendulum-v1")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"Final evaluation - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    eval_env.close()


    # TODO: try to find optimal (or better) solution to be both time and sample efficient
    # delayed multi-env learning seems necessary
    # TODO: what is functools.partial? Had no idea it existed