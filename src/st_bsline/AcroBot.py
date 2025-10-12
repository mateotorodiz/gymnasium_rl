import os
import json
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class HyperparamLogger(BaseCallback):
    """Logs hyperparameters to TensorBoard (as text and hparams) and saves them to hparams.json in the run folder."""

    def __init__(self, hparams: dict, verbose: int = 0):
        super().__init__(verbose)
        self.hparams = hparams

    def _on_training_start(self) -> None:
        # Try to get the TensorBoard writer
        tb_writer = None
        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                tb_writer = fmt.writer
                break

        # Ensure values are JSON serializable; convert non-serializable to strings
        flat_hparams = {}
        for k, v in self.hparams.items():
            try:
                json.dumps(v)
                flat_hparams[k] = v
            except TypeError:
                flat_hparams[k] = str(v)

        # Log as pretty JSON text so it's easy to read in TensorBoard
        if tb_writer is not None:
            try:
                tb_writer.add_text("hparams/json", json.dumps(flat_hparams, indent=2), global_step=0)
            except Exception:
                pass
            # Also try the TensorBoard HParams plugin (if available)
            if hasattr(tb_writer, "add_hparams"):
                try:
                    # HParams plugin requires at least one metric; we provide a dummy placeholder
                    tb_writer.add_hparams(flat_hparams, {"_init": 1})
                except Exception:
                    pass

        # Persist to file inside the TensorBoard run directory
        if self.logger.dir is not None:
            os.makedirs(self.logger.dir, exist_ok=True)
            try:
                with open(os.path.join(self.logger.dir, "hparams.json"), "w", encoding="utf-8") as f:
                    json.dump(flat_hparams, f, indent=2)
            except Exception:
                pass
        return None


env = gym.make("Acrobot-v1", render_mode=None)

# Define hyperparameters in one place so we can both pass them to A2C and log them
learning_rate = 1e-3
n_steps = 16
gamma = 0.99
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 2.0
use_rms_prop = True
policy_kwargs = {"net_arch": [64, 64]}
tensorboard_root = "./a2c_acrobot"

# Build a readable TensorBoard run name with key hyperparameters
tb_run_name = (
    f"A2C_Acrobot_lr{learning_rate}_ns{n_steps}_g{gamma}_ent{ent_coef}_vf{vf_coef}_mgn{max_grad_norm}"
)

# Prepare hparams dict for logging
hparams = {
    "algo": "A2C",
    "env_id": "Acrobot-v1",
    "policy": "MlpPolicy",
    "learning_rate": learning_rate,
    "n_steps": n_steps,
    "gamma": gamma,
    "ent_coef": ent_coef,
    "vf_coef": vf_coef,
    "max_grad_norm": max_grad_norm,
    "use_rms_prop": use_rms_prop,
    "policy_kwargs": policy_kwargs,
}

model = A2C(
    "MlpPolicy",
    env,
    learning_rate=learning_rate,
    n_steps=n_steps,
    gamma=gamma,
    ent_coef=ent_coef,
    vf_coef=vf_coef,
    max_grad_norm=max_grad_norm,
    use_rms_prop=use_rms_prop,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=tensorboard_root,
)

callback = HyperparamLogger(hparams)
model.learn(total_timesteps=200_000, tb_log_name=tb_run_name, callback=callback)
model.save("a2c_acrobot")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
# Visualization
env = gym.make("Acrobot-v1", render_mode="human")
obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        break
env.close()

