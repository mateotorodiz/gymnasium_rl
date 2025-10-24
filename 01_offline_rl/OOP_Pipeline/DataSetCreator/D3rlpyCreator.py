"""D3rlpy dataset creator for offline RL experiments."""
from typing import Tuple, Any
import d3rlpy
from d3rlpy.dataset import ReplayBuffer
import gymnasium as gym


class D3rlpyCreator:
    """
    Creator class for loading d3rlpy datasets.
    
    This class provides a convenient interface for loading standard d3rlpy datasets
    based on Gymnasium environment names.
    
    Attributes:
        envname (str): The name of the Gymnasium environment (e.g., 'CartPole-v1', 'Pendulum-v1').
        discrete (bool): Whether the environment has discrete action space.
    """
    
    def __init__(self, envname: str, discrete: bool) -> None:
        """
        Initialize the D3rlpyCreator.
        
        Args:
            envname: Name of the Gymnasium environment (e.g., 'CartPole-v1', 'Pendulum-v1').
            discrete: Whether the environment has a discrete action space.
        """
        self.envname = envname
        self.discrete = discrete
    
    def get_dataset(self) -> Tuple[ReplayBuffer, gym.Env]:
        """
        Load the dataset and environment for the specified environment name.
        
        Returns:
            A tuple containing:
                - dataset: The d3rlpy ReplayBuffer containing offline trajectories.
                - env: The Gymnasium environment object.
        
        Raises:
            ValueError: If the environment name is not supported.
        """
        if self.envname == "CartPole-v1":
            dataset, env = d3rlpy.datasets.get_cartpole()
            return dataset, env
        elif self.envname == "Pendulum-v1":
            dataset, env = d3rlpy.datasets.get_pendulum()
            return dataset, env
        else:
            raise ValueError(
                f"Unsupported environment: {self.envname}. "
                f"Supported environments: 'CartPole-v1', 'Pendulum-v1'"
            )
