import gymnasium as gym
from tqdm import tqdm
from GridWorldAgent import GridWorldAgent
from GridWorldEnv import GridWorldEnv
import numpy as np
import matplotlib.pyplot as plt
from typing import List

class Trainer:
    def __init__(self,
                 env: gym.Env,
                 agent: GridWorldAgent,
                 n_episodes: int,
):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.Gt = 0
        self.episode_rewards = []

    
    def plot_learning_curve(self, rolling_window: int = 100, title: str = "Learning Curve"):
        """
        Compute and plot the rolling average of rewards over episodes.
        """
        rewards = np.array(self.episode_rewards)
        rolling_avg = np.convolve(rewards, np.ones(rolling_window)/rolling_window, mode='valid')

        plt.figure(figsize=(10,5))
        # Shift x-axis to start from the (rolling_window-1)-th episode
        plt.plot(range(rolling_window - 1, len(rewards)), rolling_avg, color='blue', label=f'Rolling avg (window={rolling_window})')
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()


    def train(self):
        for episode in tqdm(range(self.n_episodes)):
            obs,info  = self.env.reset()
            done = False
            self.Gt = 0
            
            while not done:
                action = self.agent.get_action(obs)
                next_obs,reward,terminated,truncated,info = self.env.step(action)
                self.Gt += reward
                self.agent.update(obs,action,reward,terminated,next_obs)
                done = terminated or truncated
                obs = next_obs
            self.agent.decay_epsilon()
            self.episode_rewards.append(self.Gt)
    
    def evaluate(self):
        successes = 0
        steps_list = []
        reward_list = []

        self.agent.epsilon = 0.0  # purely greedy

        for _ in range(10):
            obs, info = self.env.reset()
            done = False
            steps = 0
            Gt = 0
            while not done:
                action = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                obs = next_obs
                steps += 1
                Gt += reward
                     
            steps_list.append(steps)
            reward_list.append(Gt)

        print("Average returned reward:", reward_list)
        #print("Average steps (for successes):", np.mean([s for s,r in zip(steps_list, [reward]*100) if r==1]))



n_episodes = 100000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes/2)
final_epsilon = 0.1
from gymnasium.wrappers import TimeLimit
env = TimeLimit(GridWorldEnv(), max_episode_steps=1000)
agent = GridWorldAgent(
    env = env,
    lr = 0.001,
    initial_epsilon= 1.0,
    epsilon_decay = epsilon_decay,
    final_epsilon=final_epsilon
)
trainer = Trainer(env,agent,n_episodes)
trainer.train()
trainer.plot_learning_curve()
trainer.evaluate()