"""Create MDPDataset manually from raw logged data."""
from typing import List
import numpy as np
import gymnasium as gym
import d3rlpy
from d3rlpy.dataset import ReplayBuffer


class ManualMDPDatasetCreator:
    """
    Create dataset by manually collecting data from multiple policies using append/clip_episode.
    
    This approach gives full control over data collection with incremental buffer writes,
    similar to MixedPolicyDatasetCreator but without using policy.collect().
    
    Attributes:
        env: Gymnasium environment for data collection.
        policies: List of policies to use for action selection (must have .predict() method).
        episodes_per_policy: List of episodes to collect from each policy.
    """
    
    def __init__(self, env: gym.Env, policies: List, episodes_per_policy: List[int]) -> None:
        """
        Initialize the creator.
        
        Args:
            env: Gymnasium environment.
            policies: List of d3rlpy policies with .predict() method.
            episodes_per_policy: List of episodes to collect from each policy.
        
        Raises:
            ValueError: If policies and episodes_per_policy have different lengths.
        """
        if len(policies) != len(episodes_per_policy):
            raise ValueError(
                f"Number of policies ({len(policies)}) must match "
                f"number of episodes_per_policy ({len(episodes_per_policy)})"
            )
        
        self.env = env
        self.policies = policies
        self.episodes_per_policy = episodes_per_policy
    
    def create_dataset(self, buffer_limit: int = 1000000) -> ReplayBuffer:
        """
        Collect data from all policies and create ReplayBuffer using incremental writes.
        
        Args:
            buffer_limit: Maximum buffer size (default: 1M transitions).
        
        Returns:
            ReplayBuffer containing collected transitions.
        """
        # Create empty buffer
        buffer = d3rlpy.dataset.create_fifo_replay_buffer(
            limit=buffer_limit,
            env=self.env
        )
        
        total_episodes = sum(self.episodes_per_policy)
        print(f"Collecting {total_episodes} episodes from {len(self.policies)} policies...")
        
        episode_count = 0
        
        # Collect from each policy sequentially
        for policy_idx, (policy, n_episodes) in enumerate(zip(self.policies, self.episodes_per_policy)):
            print(f"\nPolicy {policy_idx + 1}/{len(self.policies)}: collecting {n_episodes} episodes")
            
            for episode in range(n_episodes):
                obs, _ = self.env.reset()
                done = False
                
                while not done:
                    # Get action from policy
                    action = policy.predict(np.array([obs]))[0]
                    
                    # Step environment
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    # Append transition to buffer incrementally
                    buffer.append(obs, action, reward)
                    
                    obs = next_obs
                
                # Mark end of episode
                buffer.clip_episode(terminated)
                
                episode_count += 1
                if episode_count % 10 == 0:
                    print(f"Collected {episode_count}/{total_episodes} episodes")
        
        print(f"\nDataset created with {buffer.transition_count} transitions")
        
        return buffer


if __name__ == "__main__":
    """Minimal example: collect random policy data and train CQL."""
    
    # Setup environment and random policy
    env = gym.make("CartPole-v1")
    random_policy = d3rlpy.algos.DiscreteRandomPolicyConfig().create()
    
    # Create dataset (keeping same interface as before by wrapping in lists)
    print("Creating dataset....")
    creator = ManualMDPDatasetCreator(
        env=env, 
        policies=[random_policy],
        episodes_per_policy=[200000]
    )
    dataset = creator.create_dataset()
    
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
