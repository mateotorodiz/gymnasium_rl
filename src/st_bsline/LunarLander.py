import gymnasium as gym
import os
from stable_baselines3 import PPO

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("LunarLander-v3")
env.reset()

model = PPO("MlpPolicy", env, verbose = 1)
TIMESTEPS = 10000
for i in range(30):

    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False,tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs,reward,done,info = env.step(env.action_space.sample())