"""Create MDPDataset manually from raw logged data."""
import numpy as np
import gymnasium as gym
import d3rlpy
from d3rlpy.dataset import MDPDataset


class ManualMDPDatasetCreator:
    """
    Create MDPDataset by manually collecting observations, actions, rewards, and terminals.
    
    This approach gives full control over data collection, following d3rlpy's documentation
    for creating datasets from logged data.
    
    Attributes:
        env: Gymnasium environment for data collection.
        policy: Policy to use for action selection (must have .predict() method).
    """
    
    def __init__(self, env: gym.Env, policy) -> None:
        """
        Initialize the creator.
        
        Args:
            env: Gymnasium environment.
            policy: d3rlpy policy with .predict() method.
        """
        self.env = env
        self.policy = policy
    
    def create_dataset(self, n_episodes: int) -> MDPDataset:
        """
        Collect data and create MDPDataset.
        
        Args:
            n_episodes: Number of episodes to collect.
        
        Returns:
            MDPDataset containing collected transitions.
        """
        observations = []
        actions = []
        rewards = []
        terminals = []
        
        print(f"Collecting {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                # Store observation
                observations.append(obs)
                
                # Get action from policy
                action = self.policy.predict(np.array([obs]))[0]
                actions.append(action)
                
                # Step environment
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store reward and terminal flag
                rewards.append(reward)
                terminals.append(1.0 if terminated else 0.0)
                
                obs = next_obs
            
            if (episode + 1) % 10 == 0:
                print(f"Collected {episode + 1}/{n_episodes} episodes")
        
        # Convert to numpy arrays
        observations = np.array(observations, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        terminals = np.array(terminals, dtype=np.float32)
        
        print(f"\nDataset shape: {observations.shape[0]} transitions")
        
        # Create and return MDPDataset
        return d3rlpy.dataset.MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals
        )


if __name__ == "__main__":
    """Minimal example: collect random policy data and train CQL."""
    
    # Setup environment and random policy
    env = gym.make("CartPole-v1")
    random_policy = d3rlpy.algos.DiscreteRandomPolicyConfig().create()
    
    # Create dataset
    print("Creating dataset....")
    creator = ManualMDPDatasetCreator(env=env, policy=random_policy)
    dataset = creator.create_dataset(n_episodes=200000)
    
    # Train CQL on the dataset
    print("\nTraining CQL...")
    cql = d3rlpy.algos.DiscreteCQLConfig().create(device=False)
    cql.fit(
        dataset,
        n_steps=5000,
        n_steps_per_epoch=1000,
        show_progress=True
    )
    
    # Quick evaluation
    print("\nEvaluating trained policy...")
    eval_env = gym.make("CartPole-v1")
    total_reward = 0
    for _ in range(10):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = cql.predict(np.array([obs]))[0]
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_reward += episode_reward
    
    print(f"Average reward over 10 episodes: {total_reward / 10:.2f}")
    env.close()
    eval_env.close()
