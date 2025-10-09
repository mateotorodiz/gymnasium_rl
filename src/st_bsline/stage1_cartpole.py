import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1")

obs, info = env.reset()
print("Observation shape:", env.observation_space)
print("Action space:", env.action_space)




model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=8e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)


n_updates = 12
timesteps_per_update = 1000
rewards = []

for i in range(n_updates):
    model.learn(total_timesteps=timesteps_per_update, reset_num_timesteps=False)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    rewards.append(mean_reward)
    print(f"Iteration {i+1}: mean reward = {mean_reward:.2f}")


plt.plot(np.arange(1, n_updates + 1), rewards, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Mean reward (5 episodes)")
plt.title("PPO Learning Curve on CartPole-v1")
plt.grid(True)
plt.show()


def clone_env_with_render_mode(env, render_mode):
    env_id = env.spec.id
    kwargs = env.spec.kwargs.copy()
    kwargs["render_mode"] = render_mode
    return gym.make(env_id, **kwargs)

eval_env = clone_env_with_render_mode(env, "human")

def run_agent(model,env,render=True):
    eval_env = clone_env_with_render_mode(env,"human")

    mean_reward,reward_variance = evaluate_policy(model,eval_env,n_eval_episodes=2)
    eval_env.render()

    eval_env.close()

run_agent(model,env)

