from collections import defaultdict

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from GridWorldEnv import GridWorldEnv


class GridWorldMCAgent:
    def __init__(
        self,
        env: gym.Env,
        Nzero: int,
        n_episodes: int,
        discount_factor: float = 0.95,
    ):
        """
        Initialize a Monte Carlo agent for grid-world environment
        """
        self.env = env
        self.n_episodes = n_episodes
        self.Nzero = Nzero
        self.discount_factor = discount_factor

        action_space_n = self.env.action_space.n
        self.q_values = defaultdict(lambda: np.zeros(action_space_n))
        self.state_visits = defaultdict(int)
        self.state_action_visits = defaultdict(
            lambda: np.zeros(action_space_n, dtype=int)
        )

    def get_action(self, obs) -> int:
        # Epsilon-greedy action selection
        # epsilon = self.get_epsilon(obs)
        if np.random.random() < 0.8:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[obs]))

    def count_visits(self, obs, action):
        self.state_visits[obs] += 1
        self.state_action_visits[obs][action] += 1

    # def get_epsilon(self, obs):
    #     # Decaying epsilon based on state visits
    #     return self.Nzero / (self.Nzero + self.state_visits[obs])

    # def get_alpha(self, obs, action):
    #     # Decaying alpha based on state-action visits
    #     return 1 / max(1, self.state_action_visits[obs][action])

    def update(self, episode):
        # Every-visit MC update, backward pass
        Gt = 0
        for s, a, r in reversed(episode):
            Gt = r + self.discount_factor * Gt
            self.count_visits(s, a)
            alpha = 1
            self.q_values[s][a] += alpha * (Gt - self.q_values[s][a])

    def train(self):
        pbar = tqdm(range(self.n_episodes))
        successful_episodes = 0
        recent_returns = []
        recent_lengths = []
        for episode in pbar:
            obs, _ = self.env.reset()
            done = False
            episode_info = []
            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_info.append((obs, action, reward))
                obs = next_obs
            self.update(episode_info)
            episode_return = sum([x[2] for x in episode_info])
            episode_length = len(episode_info)
            recent_returns.append(episode_return)
            recent_lengths.append(episode_length)
            if len(recent_returns) > 100:
                recent_returns.pop(0)
                recent_lengths.pop(0)

            # Track success (reaching goal = reward of 0 at end)
            if episode_info and episode_info[-1][2] == 0:
                successful_episodes += 1

            avg_return = sum(recent_returns) / len(recent_returns)
            avg_length = sum(recent_lengths) / len(recent_lengths)
            pbar.set_description(
                f"Ep: {episode:<5} Len: {episode_length:<4.0f} "
                f"AvgLen: {avg_length:<5.1f} Success: {successful_episodes}"
            )

    def evaluate(self):
        steps_list = []
        reward_list = []
        info_list = []
        # Purely greedy policy for evaluation
        for _ in range(10):
            obs, info = self.env.reset()
            done = False
            steps = 0
            Gt = 0
            trajectory = [obs]
            while not done:
                action = int(np.argmax(self.q_values[obs]))
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                trajectory.append(next_obs)
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
        print("Average steps:", sum(steps_list) / len(steps_list))

    def visualize_policy(self):
        """Visualize the learned policy on the grid"""
        print("\nLearned Policy (arrows show best action):")
        # (0=down,1=right,2=up,3=left)
        action_symbols = {
            0: "↓",  # down
            1: "→",  # right
            2: "↑",  # up
            3: "←",  # left
        }
        for row in range(self.env.size[0]):
            line = ""
            for col in range(self.env.size[1]):
                if (row, col) == (3, 0):
                    line += "S "  # Start
                elif (row, col) == (3, 11):
                    line += "G "  # Goal
                elif row == 3 and 0 < col < 11:
                    line += "X "  # Cliff
                else:
                    # Get best action for this state (with goal position)
                    obs = ((row, col), (3, 11))
                    best_action = int(np.argmax(self.q_values[obs]))
                    line += action_symbols[best_action] + " "
            print(line)
        print("\nQ-value statistics:")
        print(f"States visited: {len(self.q_values)}")
        print(f"Total state visits: {sum(self.state_visits.values())}")


if __name__ == "__main__":
    n_episodes = 100000
    Nzero = 500  #! Not used right now
    env = GridWorldEnv(max_steps=200)
    agent = GridWorldMCAgent(
        env=env, Nzero=Nzero, n_episodes=n_episodes, discount_factor=1
    )
    agent.train()
    agent.evaluate()
    agent.visualize_policy()
