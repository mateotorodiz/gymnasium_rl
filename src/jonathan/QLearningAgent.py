import gymnasium as gym
import numpy as np
from tqdm import tqdm

from GridWorldEnv import GridWorldEnv


class QLearningAgent:
    """
    Simple Q-Learning agent with Q-table for GridWorld environment.
    Uses TD(0) updates instead of Monte Carlo.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        n_episodes: int = 1000,
    ):
        """
        Initialize Q-Learning agent.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            n_episodes: Number of training episodes
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_episodes = n_episodes

        # Initialize Q-table as dictionary (state -> action values)
        self.q_table = {}

    def get_q_values(self, state) -> np.ndarray:
        """Get Q-values for a state, initializing if needed."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.env.action_space.n)
        return self.q_table[state]

    def get_action(self, state, training=True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return self.env.action_space.sample()
        else:
            # Exploit: best action
            q_values = self.get_q_values(state)
            return int(np.argmax(q_values))

    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning update rule:
        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        current_q = self.get_q_values(state)[action]

        if done:
            # Terminal state has no future rewards
            target_q = reward
        else:
            # Use max Q-value of next state
            next_q_values = self.get_q_values(next_state)
            max_next_q = np.max(next_q_values)
            target_q = reward + self.discount_factor * max_next_q

        # Q-learning update
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train(self):
        """Train the agent using Q-learning."""
        pbar = tqdm(range(self.n_episodes), desc="Training Q-Learning Agent")

        # Track statistics
        recent_returns = []
        recent_lengths = []
        successful_episodes = 0

        for episode in pbar:
            state, _ = self.env.reset()
            done = False
            episode_return = 0
            episode_length = 0

            while not done:
                # Select and take action
                action = self.get_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Update Q-table
                self.update_q_value(state, action, reward, next_state, done)

                # Update statistics
                episode_return += reward
                episode_length += 1
                state = next_state

            # Track success (reaching goal)
            if terminated and reward == 0:
                successful_episodes += 1

            # Decay epsilon after each episode
            self.decay_epsilon()

            # Track recent performance
            recent_returns.append(episode_return)
            recent_lengths.append(episode_length)
            if len(recent_returns) > 100:
                recent_returns.pop(0)
                recent_lengths.pop(0)

            # Update progress bar
            avg_length = sum(recent_lengths) / len(recent_lengths)
            pbar.set_description(
                f"Ep: {episode:<5} Len: {episode_length:<4} "
                f"AvgLen: {avg_length:<5.1f} ε: {self.epsilon:.3f} "
                f"Success: {successful_episodes}"
            )

    def evaluate(self, n_episodes=10):
        """
        Evaluate the trained agent.

        Args:
            n_episodes: Number of episodes to evaluate
        """
        steps_list = []
        reward_list = []
        success_count = 0

        print(f"\nEvaluating agent for {n_episodes} episodes...")

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            episode_return = 0
            episode_length = 0
            trajectory = [state]

            while not done:
                # Use greedy policy (no exploration)
                action = self.get_action(state, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                trajectory.append(next_state)
                episode_return += reward
                episode_length += 1
                state = next_state

            if terminated and reward == 0:
                success_count += 1

            steps_list.append(episode_length)
            reward_list.append(episode_return)

            print(
                f"Episode {episode + 1}: Return = {episode_return}, Steps = {episode_length}"
            )

        print("\n--- Evaluation Results ---")
        print(
            f"Success rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)"
        )
        print(f"Average return: {sum(reward_list) / len(reward_list):.2f}")
        print(f"Average steps: {sum(steps_list) / len(steps_list):.2f}")
        print(f"Min steps: {min(steps_list)}")
        print(f"Max steps: {max(steps_list)}")

    def visualize_policy(self):
        """Visualize the learned policy on the grid."""
        print("\n" + "=" * 50)
        print("Learned Policy Visualization")
        print("=" * 50)
        print("S = Start, G = Goal, X = Cliff")
        print("Arrows show best action at each position\n")

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
                    # Get best action for this state
                    state = ((row, col), (3, 11))  # Include goal position
                    q_values = self.get_q_values(state)
                    best_action = int(np.argmax(q_values))
                    line += action_symbols[best_action] + " "
            print(line)

        print(f"\nQ-table size: {len(self.q_table)} states")

    def show_q_values(self, state):
        """Display Q-values for a specific state."""
        q_values = self.get_q_values(state)
        action_names = ["Down", "Right", "Up", "Left"]
        print(f"\nQ-values for state {state}:")
        for action, name in enumerate(action_names):
            print(f"  {name:>6}: {q_values[action]:8.3f}")


if __name__ == "__main__":
    # Create environment
    env = GridWorldEnv(max_steps=300)

    # Create and train Q-learning agent
    agent = QLearningAgent(
        env=env,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        n_episodes=5000,
    )

    print("Starting Q-Learning training...")
    agent.train()

    # Evaluate the trained agent
    agent.evaluate(n_episodes=10)

    # Visualize the learned policy
    agent.visualize_policy()
