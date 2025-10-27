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
        buffer_size: Maximum size of the replay buffer.
        buffer: The replay buffer storing collected transitions.
    """
    
    def __init__(
        self,
        env: gym.Env,
        policy,
        buffer_size: int = 100000
    ) -> None:
        """
        Initialize the MixedPolicyDatasetCreator.
        
        Args:
            env: Gymnasium environment for data collection.
            policy: Must be d3rlpy models (with .predict()), 
                     callables that take observations and return actions
            buffer_size: Maximum number of transitions to store in buffer.
        
        """        
        self.env = env
        self.policy = policy
        self.buffer_size = buffer_size
        
        # Create replay buffer
        self.buffer = d3rlpy.dataset.create_fifo_replay_buffer(
                limit=buffer_size,
                env=env
        )

    def create_dataset(self):
        self.policy.collect(self.env,self.buffer,n_steps=self.buffer_size)
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

    creator = MixedPolicyDatasetCreator(
        env=env,
        policy=random_policy,
        buffer_size=100000
    )
    dataset = creator.create_dataset()
    
    # Optionally save
    # creator.save_buffer("mixed_policy_dataset.h5")
    
    env.close()
    print("\nDataset creation completed!")
