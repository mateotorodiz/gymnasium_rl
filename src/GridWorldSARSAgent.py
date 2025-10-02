from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from GridWorldEnv import GridWorldEnv

class GridWorldSARSAgent:
    def __init__(self,
                 env: gym.Env,
                 Nzero: int,
                 n_episodes: int,
                 discount_factor: float = 1.00,
                 ):
        """
        Initialize a Q-learning agent for my grid-world environment
        """
        self.env = env
        self.n_episodes = n_episodes

        # Variables holding MC-relevant info
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.N_states = defaultdict(int)
        self.N_states_actions = defaultdict(lambda: np.zeros(self.env.action_space.n))

        self.Nzero = Nzero
        self.discount_factor = discount_factor

        self.training_error = []

    def get_action(self,obs) -> int:
        if np.random.random() < 0.1:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def count_visits(self,obs,action):
        self.N_states[obs] +=1
        self.N_states_actions[obs][action] +=1

    def get_epsilon(self, obs):
        return self.Nzero/(self.Nzero + self.N_states[obs])

    def get_alpha(self,obs,action):
        return 1/self.N_states_actions[obs][action]


    def update(self, s,a,sp,ap,r,done):
        """
        Efficient every-visit MC update.
        Computes returns backward in one pass (O(n) instead of O(n^2)).
        """
        if not done:
            self.q_values[s][a] = self.q_values[s][a] + 0.1*(r+self.discount_factor*self.q_values[sp][ap] - self.q_values[s][a])
        else:
             self.q_values[s][a] = self.q_values[s][a] + 0.1*(r - self.q_values[s][a])

    def train(self):
        for episode in tqdm(range(self.n_episodes)):
            obs,info  = self.env.reset()
            done = False
            episode_info = []
            action = self.get_action(obs)
            while not done:
                next_obs,reward,terminated,truncated,info = self.env.step(action)
                next_action = self.get_action(next_obs)
                done = terminated or truncated
                self.update(obs,action,next_obs,next_action,reward,done)
                #sar
                episode_info.append((obs,action,reward))
                obs = next_obs
                action = next_action


    def evaluate(self):
        successes = 0
        steps_list = []
        reward_list = []
        info_list = []
        final_observation_list = []

        self.epsilon = 0.0  # purely greedy

        for _ in range(10):
            obs, info = self.env.reset()
            done = False
            steps = 0
            Gt = 0
            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                obs = next_obs
                steps += 1
                Gt += reward

            info_list.append(info)         
            steps_list.append(steps)
            reward_list.append(Gt)
            final_observation_list.append(obs)
        print("Evaluating the agent")
        print(obs)
        print("Returned reward:", reward_list)
        print("Info on whether agent fell of cliff: ", info_list)
        print("the final observation list is: ", final_observation_list)
        #print("Average steps (for successes):", np.mean([s for s,r in zip(steps_list, [reward]*100) if r==1]))    



n_episodes = 100000
Nzero = 100
env = GridWorldEnv()

agent = GridWorldSARSAgent(
    env = env,
    Nzero=Nzero,
    n_episodes=n_episodes
)
agent.train()
agent.evaluate()
