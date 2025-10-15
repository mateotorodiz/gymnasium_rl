import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import time

def make_env():
    return gym.make("CartPole-v1")

def train(env, name):
    model = PPO("MlpPolicy", env, verbose=0)
    start = time.time()
    model.learn(total_timesteps=100_000)
    print(f"{name} finished in {time.time() - start:.2f}s")

if __name__ == '__main__':
    # For sequential (in-process) environments:
    """
    n_envs = 8
    env_dummy = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # For parallel (subprocess) environments:
    env_subproc = SubprocVecEnv([make_env for _ in range(n_envs)])
    
    train(env_dummy, "DummyVecEnv (4)")
    train(env_subproc, "SubprocVecEnv (4)")
    """
    for n_envs in [1, 2, 4, 8]:
        envs = SubprocVecEnv([make_env for _ in range(n_envs)])
        train(envs,f"{n_envs} envs")

