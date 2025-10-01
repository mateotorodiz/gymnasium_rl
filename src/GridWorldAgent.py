from collections import defaultdict
import gymnasium as gym
import numpy as np

class GridWorldAgent:
    def __init__(self,
                 env: gym.Env,
                 lr: float,
                 initial_epsilon: float,
                 epsilon_decay: float,
                 final_epsilon: float,
                 discount_factor: float = 0.95,
                 ):
        """
        Initialize a Q-learning agent for my grid-world environment
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = lr
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self,obs) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
            self,
            obs,
            action,
            reward,
            terminated,
            next_obs
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_value

        temporal_difference  = target - self.q_values[obs][action]

        self.q_values[obs][action] = self.q_values[obs][action] + self.lr*temporal_difference

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon,self.epsilon - self.epsilon_decay)