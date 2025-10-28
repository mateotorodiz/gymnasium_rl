"""Create datasets by collecting data from mixed policies."""
from typing import List, Union, Callable, Optional, Any
from pathlib import Path
import numpy as np
import gymnasium as gym
import d3rlpy
from d3rlpy.dataset import ReplayBuffer


class MixedPolicyDatasetCreator:
    """
    Create offline RL datasets by collecting data from multiple policies sequentially.
    
    This class allows mixing different policies (expert, random, trained models, etc.)
    by collecting a specified number of steps from each policy into a shared buffer.
    
    Supports d3rlpy policies (with .predict() and .collect() methods).
    
    Attributes:
        env: The Gymnasium environment for data collection.
        policies: List of policies to use for data collection.
        steps_per_policy: List of steps to collect from each policy.
        buffer_size: Maximum size of the replay buffer (sum of steps_per_policy).
        buffer: The replay buffer storing collected transitions.
    """
    
    def __init__(
        self,
        env: gym.Env,
        policies: List,
        steps_per_policy: List[int]
    ) -> None:
        """
        Initialize the MixedPolicyDatasetCreator.
        
        Args:
            env: Gymnasium environment for data collection.
            policies: List of d3rlpy policies (with .collect() method).
            steps_per_policy: List of steps to collect from each policy.
                             The buffer_size will be set to sum(steps_per_policy).
        
        Raises:
            ValueError: If policies and steps_per_policy have different lengths.
        """
        if len(policies) != len(steps_per_policy):
            raise ValueError(
                f"Number of policies ({len(policies)}) must match "
                f"number of steps_per_policy ({len(steps_per_policy)})"
            )
        
        self.env = env
        self.policies = policies
        self.steps_per_policy = steps_per_policy
        self.buffer_size = sum(steps_per_policy)
        
        # Create replay buffer with total capacity
        self.buffer = d3rlpy.dataset.create_fifo_replay_buffer(
                limit=self.buffer_size,
                env=env
        )

    def create_dataset(self):
        """
        Collect data from all policies sequentially into the shared buffer.
        
        Each policy will collect the specified number of steps in order.
        
        Returns:
            The filled replay buffer.
        """
        print(f"Creating dataset with {len(self.policies)} policies...")
        print(f"Total buffer size: {self.buffer_size}")
        
        for i, (policy, n_steps) in enumerate(zip(self.policies, self.steps_per_policy)):
            print(f"\nCollecting {n_steps} steps from policy {i+1}/{len(self.policies)}...")
            policy.collect(self.env, self.buffer, n_steps=n_steps)
        
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
    
    # Create two policies: random and another random (in practice, one would be expert)
    random_policy = d3rlpy.algos.DiscreteRandomPolicyConfig().create()
    another_policy = d3rlpy.algos.DiscreteRandomPolicyConfig().create()

    # Create mixed dataset: 50k steps from each policy
    creator = MixedPolicyDatasetCreator(
        env=env,
        policies=[random_policy, another_policy],
        steps_per_policy=[50000, 50000]
    )
    dataset = creator.create_dataset()
    
    
    # Optionally save
    # creator.save_buffer("mixed_policy_dataset.h5")
    
    env.close()
    print("\nDataset creation completed!")
