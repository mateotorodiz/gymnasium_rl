import json
import sys
import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

def train(config_path, log_dir):
    # Load config
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Create env
    env = make_vec_env("Pendulum-v1", n_envs=4, monitor_dir=log_dir)

    # Eval env
    eval_env = gym.make("Pendulum-v1")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=log_dir,
        log_path=log_dir, eval_freq=5000, deterministic=True
    )

    # Model
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, **cfg)
    model.learn(total_timesteps=200_000, callback=eval_callback, progress_bar=True)
    model.save(os.path.join(log_dir, "final_model"))

if __name__ == "__main__":
    # Configuration mapping: 0=default, 1=low_lr, 2=high_batch
    configs = {
        0: {
            "config_path": "configs/sac_default.json",
            "log_dir": "logs/run_default/"
        },
        1: {
            "config_path": "configs/sac_low_lr.json", 
            "log_dir": "logs/run_low_lr/"
        },
        2: {
            "config_path": "configs/sac_high_batch.json",
            "log_dir": "logs/run_high_batch/"
        }
    }
    
    # Hardcoded selection (change this number: 0=default, 1=low_lr, 2=high_batch)
    selection = 2
    
    # Get script directory for relative paths
    script_dir = os.path.dirname(__file__)
    
    # Get config and log paths
    config = configs[selection]
    config_file = os.path.join(script_dir, config["config_path"])
    log_dir = os.path.join(script_dir, config["log_dir"])
    
    print(f"Using configuration {selection}: {config['config_path']}")
    print(f"Logging to: {config['log_dir']}")
    
    os.makedirs(log_dir, exist_ok=True)
    train(config_file, log_dir)
