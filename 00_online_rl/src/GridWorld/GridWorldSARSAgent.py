from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from GridWorldEnv import GridWorldEnv

class GridWorldSARSAgent:
    """
    SARSA agent for the GridWorld environment.
    Learns an action-value function using the SARSA algorithm with epsilon-greedy exploration.
    """
    def __init__(self,
                 env: gym.Env,
                 Nzero: int,
                 n_episodes: int,
                 discount_factor: float = 1.00,
                 ):
        """
        Initialize the SARSA agent.

        Args:
            env (gym.Env): The GridWorld environment.
            Nzero (int): Parameter for epsilon schedule (unused if fixed epsilon).
            n_episodes (int): Number of training episodes.
            discount_factor (float): Discount factor for future rewards.
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
        """
        Select an action using epsilon-greedy policy.

        Args:
            obs: Current observation.

        Returns:
            int: Selected action.
        """
        if np.random.random() < 0.1:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def count_visits(self,obs,action):
        """
        Increment visit counters for state and state-action pairs.

        Args:
            obs: Current observation.
            action (int): Action taken.
        """
        self.N_states[obs] +=1
        self.N_states_actions[obs][action] +=1

    def get_epsilon(self, obs):
        """
        Compute epsilon for epsilon-greedy policy based on state visits.

        Args:
            obs: Current observation.

        Returns:
            float: Epsilon value.
        """
        return self.Nzero/(self.Nzero + self.N_states[obs])

    def get_alpha(self,obs,action):
        """
        Compute learning rate alpha based on state-action visits.

        Args:
            obs: Current observation.
            action (int): Action taken.

        Returns:
            float: Alpha value.
        """
        return 1/self.N_states_actions[obs][action]


    def update(self, s,a,sp,ap,r,done):
        """
        Perform the SARSA update for Q-values.

        Args:
            s: Current state.
            a (int): Action taken.
            sp: Next state.
            ap (int): Next action.
            r (float): Reward received.
            done (bool): Whether the episode has terminated.
        """
        if not done:
            self.q_values[s][a] = self.q_values[s][a] + 0.1*(r+self.discount_factor*self.q_values[sp][ap] - self.q_values[s][a])
        else:
             self.q_values[s][a] = self.q_values[s][a] + 0.1*(r - self.q_values[s][a])

    def train(self):
        """
        Train the SARSA agent over multiple episodes.
        """
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
        """
        Evaluate the SARSA agent's performance over several episodes.
        Prints summary statistics.
        """
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
        print("Evaluating the SARSA agent")
        print("Returned reward:", reward_list)
        print("Info on whether agent fell of cliff: ", info_list)
        print("the final observation list is: ", final_observation_list)
        #print("Average steps (for successes):", np.mean([s for s,r in zip(steps_list, [reward]*100) if r==1]))    



n_episodes = 50000
Nzero = 100
env = GridWorldEnv()

agent = GridWorldSARSAgent(
    env = env,
    Nzero=Nzero,
    n_episodes=n_episodes
)
agent.train()
agent.evaluate()
