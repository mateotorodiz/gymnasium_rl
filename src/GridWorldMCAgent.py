from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from GridWorldEnv import GridWorldEnv

class GridWorldMCAgent:
    def __init__(self,
                 env: gym.Env,
                 initial_epsilon: float,
                 Nzero: int,
                 n_episodes: int,
                 discount_factor: float = 0.95,
                 ):
        """
        Initialize a Q-learning agent for my grid-world environment
        """
        self.env = env
        self.n_episodes = n_episodes

        # Variables holding MC-relevant info
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.N_states = defaultdict(lambda: 0)
        self.N_states_actions = defaultdict(lambda: np.zeros(env.action_space.n))

        self.Nzero = Nzero
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon

        self.training_error = []
        self.episode_rewards = []


    def get_action(self,obs) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
            self,
            obs,
            action,
            Gt,
            alpha
    ):
        """
        Function to update the Q function based on every-visit MC. Algorithm is alright.
        """
        temporal_difference  = Gt - self.q_values[obs][action]
        self.q_values[obs][action] = self.q_values[obs][action] + alpha*temporal_difference
        self.training_error.append(temporal_difference)

    def count_visits(self,obs,action):
        self.N_states[obs] +=1
        self.N_states_actions[obs][action] +=1

    def decay_epsilon(self, obs):
        self.epsilon = self.Nzero/(self.Nzero + self.N_states[obs])

    def update_alpha(self,obs,action):
        return 1/self.N_states_actions[obs][action]


    def train(self):
        for episode in tqdm(range(self.n_episodes)):
            obs,info = self.env.reset()
            done = False
            s_a_r = []
            Gt = 0

            while not done: #Loop within an episode, agent does NOT learn during this time
                # Take an action
                action = self.get_action(obs)
                # Reaction from the environment
                next_obs,reward,terminated,truncated,info = self.env.step(action)
                # Register the results
                s_a_r.append((obs,action,reward))
                # Update the environment/episode
                done = terminated or truncated
                obs = next_obs

            for s,a,r in reversed(s_a_r):
                Gt += r
                self.count_visits(s,a)
                alpha = self.update_alpha(s,a)
                self.update(s,a,Gt,alpha)
                self.decay_epsilon(s)


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
        print("Returned reward:", reward_list)
        print("Info on whether agent fell of cliff: ", info_list)
        #print("Average steps (for successes):", np.mean([s for s,r in zip(steps_list, [reward]*100) if r==1]))    



n_episodes = 70000
Nzero = 100
env = GridWorldEnv()

agent = GridWorldMCAgent(
    env = env,
    initial_epsilon=1.0,
    Nzero=Nzero,
    n_episodes=n_episodes
)
agent.train()
agent.evaluate()
