from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from GridWorldEnv import GridWorldEnv

class GridWorldMCAgent:
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
        if np.random.random() < self.get_epsilon(obs):
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


    def update(self, episode):
        """
        Efficient every-visit MC update.
        Computes returns backward in one pass (O(n) instead of O(n^2)).
        """
        j = 0
        for s, a, _ in episode: 
            Gt = sum([x[2]*(self.discount_factor**i) for i,x in enumerate(episode[j:])])
            
            self.N_states_actions[s][a] += 1
            
            error = Gt - self.q_values[s][a]
            self.q_values[s][a] += self.get_alpha(s, a) * error
            
            j += 1

    def train(self):
        for episode in tqdm(range(self.n_episodes)):
            obs,info  = self.env.reset()
            done = False
            episode_info = []
            
            while not done:
                action = self.get_action(obs)
                next_obs,reward,terminated,truncated,info = self.env.step(action)
                done = terminated or truncated
                episode_info.append((obs,action,reward))
                obs = next_obs
            self.update(episode_info)


    def evaluate(self):
        successes = 0
        steps_list = []
        reward_list = []
        info_list = []

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
        print("Evaluating the agent")
        print(obs)
        print("Returned reward:", reward_list)
        print("Info on whether agent fell of cliff: ", info_list)
        #print("Average steps (for successes):", np.mean([s for s,r in zip(steps_list, [reward]*100) if r==1]))    



n_episodes = 30000
Nzero = 100
env = GridWorldEnv()

agent = GridWorldMCAgent(
    env = env,
    Nzero=Nzero,
    n_episodes=n_episodes
)
agent.train()
agent.evaluate()
