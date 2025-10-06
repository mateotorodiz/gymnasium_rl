from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from JobScheduleEnv import JobScheduleEnv
from JobScheduleGreedyAgent import JobScheduleGreedyAgent

class JobScheduleSARSAgent:
    """
    SARSA agent for the Job Scheduling environment.
    Learns an action-value function using the SARSA algorithm with epsilon-greedy exploration.
    """
    def __init__(self,
                 env: gym.Env,
                 n_episodes: int,
                 Nzero: int,
                 learning_rate:int = 0.1,
                 discount_factor: float = 1.00,
                 ):
        """
        Initialize the SARSA agent for Job Scheduling.

        Args:
            env (gym.Env): The Job Scheduling environment.
            Nzero (int): Parameter for epsilon schedule (unused if fixed epsilon).
            n_episodes (int): Number of training episodes.
            discount_factor (float): Discount factor for future rewards.
        """
        self.env = env
        self.n_episodes = n_episodes
        self.learning_rate = learning_rate

        # Variables holding MC-relevant info
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.N_states = defaultdict(int)
        self.N_states_actions = defaultdict(lambda: np.zeros(self.env.action_space.n))

        self.Nzero = Nzero
        self.discount_factor = discount_factor

        self.training_error = []

        # Monitoring progress
        self.q_change_history = []
        self.rolling_rewards = []
        self._last_q_values = None
        self._reward_buffer = []

    def get_action(self,obs) -> int:
        """
        Select an action using epsilon-greedy policy for job assignment.

        Args:
            obs: Current observation.

        Returns:
            int: Selected action (machine index).
        """
        if np.random.random() < self.get_epsilon(obs):
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def count_visits(self,obs,action):
        """
        Increment visit counters for state and state-action pairs in job scheduling.

        Args:
            obs: Current observation.
            action (int): Action taken.
        """
        self.N_states[obs] +=1
        self.N_states_actions[obs][action] +=1

    def get_epsilon(self, obs):
        """
        Compute epsilon for epsilon-greedy policy based on state visits in job scheduling.

        Args:
            obs: Current observation.

        Returns:
            float: Epsilon value.
        """
        return max(0.01,self.Nzero/(self.Nzero + self.N_states[obs]))

    def get_alpha(self,obs,action):
        """
        Compute learning rate alpha based on state-action visits in job scheduling.

        Args:
            obs: Current observation.
            action (int): Action taken.

        Returns:
            float: Alpha value.
        """
        return 1/self.N_states_actions[obs][action]


    def update(self, s,a,sp,ap,r,done):
        """
        Perform the SARSA update for Q-values in job scheduling.

        Args:
            s: Current state.
            a (int): Action taken.
            sp: Next state.
            ap (int): Next action.
            r (float): Reward received.
            done (bool): Whether the episode has terminated.
        """
        if not done:
            self.q_values[s][a] = self.q_values[s][a] + self.get_alpha(s,a)*(r+self.discount_factor*self.q_values[sp][ap] - self.q_values[s][a])
        else:
             self.q_values[s][a] = self.q_values[s][a] + self.get_alpha(s,a)*(r - self.q_values[s][a])

    def monitor_progress(self, episode_reward):
        """
        Track Q-value changes and rolling average of episode rewards.
        Call this at the end of each episode.
        """
        # Q-value change (L2 norm between previous and current Q-values)
        flat_q = np.concatenate([v for v in self.q_values.values()]) if self.q_values else np.array([0])
        if self._last_q_values is None:
            q_change = 0.0
        else:
            q_change = np.linalg.norm(flat_q - self._last_q_values)
        self.q_change_history.append(q_change)
        self._last_q_values = flat_q.copy()

        # Rolling average of rewards (window size 100)
        self._reward_buffer.append(episode_reward)
        if len(self._reward_buffer) > 100:
            self._reward_buffer.pop(0)
        rolling_avg = np.mean(self._reward_buffer)
        self.rolling_rewards.append(rolling_avg)

    def train(self):
        """
        Train the SARSA agent over multiple episodes in the Job Scheduling environment.
        """
        for episode in tqdm(range(self.n_episodes)):
            obs,info  = self.env.reset()
            done = False
            episode_info = []
            action = self.get_action(obs)
            Gt = 0
            while not done:
                next_obs,reward,terminated,truncated,info = self.env.step(action)
                next_action = self.get_action(next_obs)
                done = terminated or truncated
                self.count_visits(obs,action)
                self.update(obs,action,next_obs,next_action,reward,done)
                #sar
                episode_info.append((obs,action,reward))
                obs = next_obs
                action = next_action
                Gt += reward
            #self.monitor_progress(Gt)


    def evaluate(self):
        """
        Evaluate the SARSA agent's performance over several episodes in the Job Scheduling environment.
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
        print("Evaluating the SARSA agent on Job Scheduling environment")
        print("Returned reward per episode:", reward_list)
        #print("Info about the agent: ", info_list)
        #print("Average steps (for successes):", np.mean([s for s,r in zip(steps_list, [reward]*100) if r==1]))    


if __name__ == "__main__":
    n_episodes = 10000
    Nzero = 10
    env = JobScheduleEnv(n_machines=10)

    agent = JobScheduleSARSAgent(
        env = env,
        n_episodes=n_episodes,
        Nzero=Nzero,
        learning_rate=0.1
    )
    agent.train()
    agent.evaluate()

    agent = JobScheduleGreedyAgent(env)
    agent.evaluate(n_episodes=10)
