"""Create datasets by collecting data from mixed policies."""
from typing import List, Union, Callable, Optional, Any
from pathlib import Path
import numpy as np
import gymnasium as gym
import d3rlpy
from d3rlpy.dataset import ReplayBuffer


class MixedPolicyDatasetCreator:
    """
    Create offline RL datasets by collecting episodes from multiple policies.
    
    This class allows mixing different policies (expert, random, trained models, etc.)
    at the episode level. Each episode is collected using one policy, selected based
    on the provided probability distribution.
    
    Supports both d3rlpy policies (with .predict() method) and arbitrary callables
    like Stable Baselines models.
    
    Attributes:
        env: The Gymnasium environment for data collection.
        policies: List of policies to use for data collection.
        policy_probs: Probability of selecting each policy (must sum to 1.0).
        buffer_size: Maximum size of the replay buffer.
        buffer: The replay buffer storing collected transitions.
    """
    
    def __init__(
        self,
        env: gym.Env,
        policies: List[Union[Any, Callable]],
        policy_probs: List[float],
        buffer_size: int = 100000
    ) -> None:
        """
        Initialize the MixedPolicyDatasetCreator.
        
        Args:
            env: Gymnasium environment for data collection.
            policies: List of policies. Can be d3rlpy models (with .predict()), 
                     callables that take observations and return actions,
                     or Stable Baselines models (with .predict()).
            policy_probs: List of probabilities for selecting each policy.
                         Must have same length as policies and sum to 1.0.
            buffer_size: Maximum number of transitions to store in buffer.
        
        Raises:
            ValueError: If policy_probs don't sum to 1.0 or lengths don't match.
        """
        if len(policies) != len(policy_probs):
            raise ValueError(
                f"Number of policies ({len(policies)}) must match "
                f"number of probabilities ({len(policy_probs)})"
            )
        
        if not np.isclose(sum(policy_probs), 1.0):
            raise ValueError(
                f"Policy probabilities must sum to 1.0, got {sum(policy_probs)}"
            )
        
        self.env = env
        self.policies = policies
        self.policy_probs = np.array(policy_probs)
        self.buffer_size = buffer_size
        
        # Create replay buffer
        self.buffer = d3rlpy.dataset.create_fifo_replay_buffer(
            limit=buffer_size,
            env=env
        )
    
    def _select_policy(self) -> Any:
        """
        Select a policy based on the probability distribution.
        
        Returns:
            Selected policy from the policies list.
        """
        policy_idx = np.random.choice(len(self.policies), p=self.policy_probs)
        return self.policies[policy_idx]
    
    def _get_action(self, policy: Any, observation: np.ndarray) -> np.ndarray:
        """
        Get action from a policy, supporting multiple policy types.
        
        Supports:
        - d3rlpy policies (have .predict() method)
        - Stable Baselines policies (have .predict() method, return tuple)
        - Callable functions (observation) -> action
        
        Args:
            policy: The policy to use for action selection.
            observation: Current observation from environment.
        
        Returns:
            Action to take in the environment.
        """
        # Check if it's a callable (function)
        if callable(policy) and not hasattr(policy, 'predict'):
            return policy(observation)
        
        # Handle policies with .predict() method (d3rlpy, Stable Baselines, etc.)
        if hasattr(policy, 'predict'):
            # d3rlpy expects batch input
            if hasattr(policy, '__class__') and 'd3rlpy' in str(type(policy)):
                obs_batch = np.asarray([observation], dtype=np.float32)
                action = policy.predict(obs_batch)[0]
            else:
                # Stable Baselines and similar (may return tuple)
                result = policy.predict(observation, deterministic=True)
                action = result[0] if isinstance(result, tuple) else result
            
            return action
        
        raise TypeError(
            f"Policy must be callable or have a .predict() method. "
            f"Got type: {type(policy)}"
        )
    
    def collect_episodes(
        self,
        n_episodes: int,
        seed: Optional[int] = None,
        show_progress: bool = True
    ) -> ReplayBuffer:
        """
        Collect data by running episodes with mixed policies.
        
        Each episode uses one policy selected according to policy_probs.
        
        Args:
            n_episodes: Number of episodes to collect.
            seed: Random seed for reproducibility.
            show_progress: Whether to show progress bar.
        
        Returns:
            ReplayBuffer containing collected transitions.
        """
        if seed is not None:
            np.random.seed(seed)
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(n_episodes), desc="Collecting episodes")
            except ImportError:
                iterator = range(n_episodes)
                print(f"Collecting {n_episodes} episodes...")
        else:
            iterator = range(n_episodes)
        
        total_steps = 0
        total_reward = 0.0
        
        for episode_idx in iterator:
            # Select policy for this episode
            policy = self._select_policy()
            
            # Reset environment
            observation, info = self.env.reset(
                seed=seed + episode_idx if seed is not None else None
            )
            
            # Collect episode data in lists
            observations = []
            actions = []
            rewards = []
            
            episode_reward = 0.0
            done = False
            
            while not done:
                # Get action from selected policy
                action = self._get_action(policy, observation)
                
                # Step environment
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition data
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                
                observation = next_observation
                episode_reward += reward
                total_steps += 1
            
            # Create Episode object and add to buffer
            episode = d3rlpy.dataset.Episode(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(actions),
                rewards=np.array(rewards, dtype=np.float32),
                terminated=True  # Episode ended naturally
            )
            self.buffer.append_episode(episode)
            
            total_reward += episode_reward
        
        avg_reward = total_reward / n_episodes
        print(f"\nCollected {total_steps} steps over {n_episodes} episodes")
        print(f"Average episode reward: {avg_reward:.2f}")
        
        return self.buffer
    
    def collect_steps(
        self,
        n_steps: int,
        seed: Optional[int] = None,
        show_progress: bool = True
    ) -> ReplayBuffer:
        """
        Collect a fixed number of steps with mixed policies.
        
        Collects episodes until n_steps is reached. Each episode uses one policy
        selected according to policy_probs.
        
        Args:
            n_steps: Total number of steps (transitions) to collect.
            seed: Random seed for reproducibility.
            show_progress: Whether to show progress bar.
        
        Returns:
            ReplayBuffer containing collected transitions.
        """
        if seed is not None:
            np.random.seed(seed)
        
        steps_collected = 0
        episode_count = 0
        total_reward = 0.0
        
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=n_steps, desc="Collecting steps")
            except ImportError:
                pbar = None
                print(f"Collecting {n_steps} steps...")
        else:
            pbar = None
        
        while steps_collected < n_steps:
            # Select policy for this episode
            policy = self._select_policy()
            
            # Reset environment
            observation, info = self.env.reset(
                seed=seed + episode_count if seed is not None else None
            )
            
            # Collect episode data in lists
            observations = []
            actions = []
            rewards = []
            
            episode_reward = 0.0
            done = False
            
            while not done and steps_collected < n_steps:
                # Get action from selected policy
                action = self._get_action(policy, observation)
                
                # Step environment
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition data
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                
                observation = next_observation
                episode_reward += reward
                steps_collected += 1
                
                if pbar:
                    pbar.update(1)
            
            # Create Episode object and add to buffer
            episode = d3rlpy.dataset.Episode(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(actions),
                rewards=np.array(rewards, dtype=np.float32),
                terminated=done  # Use actual termination status
            )
            self.buffer.append_episode(episode)
            
            total_reward += episode_reward
            episode_count += 1
        
        if pbar:
            pbar.close()
        
        avg_reward = total_reward / episode_count
        print(f"\nCollected {steps_collected} steps over {episode_count} episodes")
        print(f"Average episode reward: {avg_reward:.2f}")
        
        return self.buffer
    
    def save_buffer(self, filepath: str) -> None:
        """
        Save the replay buffer to disk.
        
        Args:
            filepath: Path where the buffer should be saved (e.g., "dataset.h5").
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w+b") as f:
            self.buffer.dump(f)
        
        print(f"Buffer saved to {path}")
    
    def load_buffer(self, filepath: str) -> ReplayBuffer:
        """
        Load a replay buffer from disk.
        
        Args:
            filepath: Path to the saved buffer file.
        
        Returns:
            Loaded ReplayBuffer.
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Buffer file not found: {path}")
        
        with open(path, "rb") as f:
            self.buffer = d3rlpy.dataset.load(f)
        
        print(f"Buffer loaded from {path}")
        return self.buffer


if __name__ == "__main__":
    # Example usage with CartPole
    import gymnasium as gym
    
    # Create environment
    env = gym.make("CartPole-v1")
    
    # Create a random policy (d3rlpy)
    random_policy = d3rlpy.algos.DiscreteRandomPolicyConfig().create()
    
    # Create another random policy as a simple callable
    def uniform_random_policy(observation):
        return env.action_space.sample()
    
    # Create dataset creator with 70% random d3rlpy policy, 30% callable policy
    creator = MixedPolicyDatasetCreator(
        env=env,
        policies=[random_policy, uniform_random_policy],
        policy_probs=[0.7, 0.3],
        buffer_size=100000
    )
    
    # Collect data
    buffer = creator.collect_episodes(n_episodes=50, seed=42)
    
    # Optionally save
    # creator.save_buffer("mixed_policy_dataset.h5")
    
    env.close()
    print("\nDataset creation completed!")
