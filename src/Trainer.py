import gymnasium as gym
from tqdm import tqdm
from GridWorldAgent import GridWorldAgent
from GridWorldEnv import GridWorldEnv
import numpy as np

class Trainer:
    def __init__(self,
                 env: gym.Env,
                 agent: GridWorldAgent,
                 n_episodes: int,
):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes

    def train(self):
        for episode in tqdm(range(self.n_episodes)):
            obs,info  = self.env.reset()
            done = False
            
            while not done:
                action = self.agent.get_action(obs)

                next_obs,reward,terminated,truncated,info = self.env.step(action)
                self.agent.update(obs,action,reward,terminated,next_obs)
                done = terminated or truncated
                obs = next_obs
            self.agent.decay_epsilon()
    
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
                print(obs)
            
            
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
trainer.evaluate()