from stable_baselines3 import PPO
import os
from SnekEnv import SnekEnv
import time



models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = SnekEnv()
env.reset()

model = PPO('MlpPolicy',
			 env,
			   verbose=1,
				n_steps=2048,
				batch_size=64,
				n_epochs = 5,
				gamma = 0.99,
				learning_rate=5e-4,
				tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")