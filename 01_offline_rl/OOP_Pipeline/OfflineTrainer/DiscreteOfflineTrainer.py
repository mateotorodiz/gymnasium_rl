"""Discrete offline RL trainer using d3rlpy algorithms."""
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Any, List
from pathlib import Path
import d3rlpy
from d3rlpy.dataset import ReplayBuffer
import gymnasium as gym
import numpy as np


@dataclass
class DiscreteAlgoConfig:
    """
    Configuration for discrete offline RL algorithms.
    
    This dataclass encapsulates all the hyperparameters needed to configure
    a discrete d3rlpy algorithm like DiscreteCQL.
    
    Attributes:
        batch_size: Mini-batch size for training.
        gamma: Discount factor for future rewards.
        observation_scaler: Observation preprocessor (None, or d3rlpy scaler).
        action_scaler: Action scaler (optional).
        reward_scaler: Reward preprocessor (None, or d3rlpy reward scaler).
        compile_graph: Flag to enable JIT compilation and CUDAGraph.
        learning_rate: Learning rate for the optimizer.
        optim_factory: Optimizer factory (default uses AdamFactory).
        encoder_factory: Encoder factory for neural network architecture.
        q_func_factory: Q-function factory for value network architecture.
        n_critics: Number of Q-functions for ensemble.
        target_update_interval: Interval (in steps) to synchronize the target network.
        alpha: CQL alpha value for conservative penalty.
        device: Device to use for training (False, 'cpu', 'cuda:0', True for 'cuda:0', or int for 'cuda:<device>').
        enable_ddp: Flag to wrap models with DDP for Data Distributed Parallel training.
    """
    batch_size: int = 32
    gamma: float = 0.99
    observation_scaler: Optional[Any] = None
    action_scaler: Optional[Any] = None
    reward_scaler: Optional[Any] = None
    compile_graph: bool = False
    learning_rate: float = 6.25e-5
    optim_factory: Optional[Any] = None
    encoder_factory: Optional[Any] = None
    q_func_factory: Optional[Any] = None
    n_critics: int = 1
    target_update_interval: int = 8000
    alpha: float = 1.0
    device: Any = False  # False, True, int, or str
    enable_ddp: bool = False


@dataclass
class FitConfig:
    """
    Configuration for the fitting process.
    
    This dataclass contains all parameters related to the training loop,
    evaluation, logging, and checkpointing.
    
    Attributes:
        n_steps: Total number of gradient steps to train (mutually exclusive with n_epochs).
        n_steps_per_epoch: Number of gradient steps per epoch.
        experiment_name: Name for the experiment (used in logging directory).
        with_timestamp: Whether to add timestamp to directory name.
        show_progress: Whether to show progress bar for iterations.
        save_interval: Interval (in epochs) to save model parameters.
        evaluators: Dictionary of evaluator functions for monitoring performance.
    """
    n_steps: Optional[int] = 10000
    n_steps_per_epoch: int = 1000
    experiment_name: Optional[str] = None
    with_timestamp: bool = True
    show_progress: bool = True
    save_interval: int = 1
    evaluators: Dict[str, Any] = field(default_factory=dict)


class DiscreteOfflineTrainer:
    """
    Trainer for discrete action space offline RL algorithms using d3rlpy.
    
    This class provides a structured interface for training, evaluating, and managing
    discrete offline RL models. It handles the full lifecycle: initialization,
    training, model saving/loading, and evaluation.
    
    Attributes:
        env: The Gymnasium environment.
        dataset: The replay buffer containing offline trajectories.
        algo_config: Configuration dataclass for the algorithm hyperparameters.
        fit_config: Configuration dataclass for the training process.
        model_path: Path where the model will be saved/loaded.
        algo: The d3rlpy algorithm instance (DiscreteCQL, etc.).
    """
    
    def __init__(
        self,
        env: gym.Env,
        dataset: ReplayBuffer,
        algo_config: DiscreteAlgoConfig,
        fit_config: FitConfig,
        model_path: str
    ) -> None:
        """
        Initialize the DiscreteOfflineTrainer.
        
        Args:
            env: Gymnasium environment for the task.
            dataset: d3rlpy ReplayBuffer containing offline trajectories.
            algo_config: Configuration object for algorithm hyperparameters.
            fit_config: Configuration object for the training process.
            model_path: File path where the model will be saved/loaded.
        """
        self.env = env
        self.dataset = dataset
        self.algo_config = algo_config
        self.fit_config = fit_config
        self.model_path = Path(model_path)
        self.algo: Optional[Any] = None
        
        # Create the algorithm instance with the provided configuration
        self._create_algorithm()
    
    def _create_algorithm(self) -> None:
        """
        Create the d3rlpy algorithm instance from the configuration.
        
        Currently supports DiscreteCQL. This method instantiates the algorithm
        with all parameters from algo_config.
        """
        # Build kwargs for config, only including non-None optional parameters
        config_kwargs = {
            'batch_size': self.algo_config.batch_size,
            'gamma': self.algo_config.gamma,
            'learning_rate': self.algo_config.learning_rate,
            'n_critics': self.algo_config.n_critics,
            'target_update_interval': self.algo_config.target_update_interval,
            'alpha': self.algo_config.alpha,
            'compile_graph': self.algo_config.compile_graph,
        }
        
        # Add optional parameters only if they're not None
        if self.algo_config.observation_scaler is not None:
            config_kwargs['observation_scaler'] = self.algo_config.observation_scaler
        if self.algo_config.action_scaler is not None:
            config_kwargs['action_scaler'] = self.algo_config.action_scaler
        if self.algo_config.reward_scaler is not None:
            config_kwargs['reward_scaler'] = self.algo_config.reward_scaler
        if self.algo_config.optim_factory is not None:
            config_kwargs['optim_factory'] = self.algo_config.optim_factory
        if self.algo_config.encoder_factory is not None:
            config_kwargs['encoder_factory'] = self.algo_config.encoder_factory
        if self.algo_config.q_func_factory is not None:
            config_kwargs['q_func_factory'] = self.algo_config.q_func_factory
        
        # Create DiscreteCQL configuration
        config = d3rlpy.algos.DiscreteCQLConfig(**config_kwargs)
        
        # Create the algorithm instance
        self.algo = config.create(
            device=self.algo_config.device,
            enable_ddp=self.algo_config.enable_ddp
        )
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load a trained model from disk.
        
        This method builds the network architecture using the environment,
        then loads the saved weights from the specified path.
        
        Args:
            model_path: Path to the saved model file. If None, uses self.model_path.
        
        Raises:
            ValueError: If algorithm has not been created.
            FileNotFoundError: If the model file doesn't exist.
        """
        if self.algo is None:
            raise ValueError("Algorithm not created. Call _create_algorithm first.")
        
        path = Path(model_path) if model_path else self.model_path
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Build network shapes from the environment
        self.algo.build_with_env(self.env)
        
        # Load the trained weights
        self.algo.load_model(str(path))
        print(f"Model loaded from {path}")
    
    def fit_model(self) -> List[tuple[int, Dict[str, float]]]:
        """
        Train the offline RL algorithm on the dataset.
        
        This method runs the full training loop using the fit configuration.
        It handles logging, evaluation, and periodic model saving.
        
        Returns:
            List of tuples containing (epoch, metrics_dict) for each epoch.
        
        Raises:
            ValueError: If algorithm has not been created.
        """
        if self.algo is None:
            raise ValueError("Algorithm not created. Call _create_algorithm first.")
        
        # Train the algorithm
        results = self.algo.fit(
            self.dataset,
            n_steps=self.fit_config.n_steps,
            n_steps_per_epoch=self.fit_config.n_steps_per_epoch,
            experiment_name=self.fit_config.experiment_name,
            with_timestamp=self.fit_config.with_timestamp,
            show_progress=self.fit_config.show_progress,
            save_interval=self.fit_config.save_interval,
        )
        
        print(f"Training completed. Model will be saved to {self.model_path}")
        return results
    
    def save_model(self, model_path: Optional[str] = None) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path where the model should be saved. If None, uses self.model_path.
        
        Raises:
            ValueError: If algorithm has not been created or trained.
        """
        if self.algo is None:
            raise ValueError("Algorithm not created. Cannot save model.")
        
        path = Path(model_path) if model_path else self.model_path
        
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        self.algo.save_model(str(path))
        print(f"Model saved to {path}")
    
    def predict(self, observations: Any) -> Any:
        """
        Get greedy action predictions from the trained policy.
        
        Args:
            observations: Observations to predict actions for.
        
        Returns:
            Predicted actions (greedy policy).
        
        Raises:
            ValueError: If algorithm has not been created or trained.
        """
        if self.algo is None:
            raise ValueError("Algorithm not created. Cannot make predictions.")
        
        return self.algo.predict(observations)
    

def evaluate(algo, env, num_episodes=20, seed=42):
    returns = []
    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        ep_ret = 0.0
        while not done:
            # d3rlpy expects batch input; take greedy action
            action = int(algo.predict(np.asarray([obs], dtype=np.float32))[0])
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
        returns.append(ep_ret)

    mean = float(np.mean(returns)) if returns else 0.0
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    return mean, std


if __name__ == "__main__":
    dataset,env = d3rlpy.datasets.get_cartpole()
    algo_config = DiscreteAlgoConfig()
    fit_config = FitConfig()
    env_id = "CartPole-v1"
    model_path = "cql_cartpole.pt"  # adjust to your saved model path
    n_episodes = 20



    env = gym.make(env_id)
    discretetrainer = DiscreteOfflineTrainer(env=env,
                                             dataset=dataset,
                                            algo_config=algo_config,
                                            fit_config=fit_config,
                                            model_path=model_path)
    discretetrainer.fit_model()
    algo = discretetrainer.algo
    mean, std = evaluate(algo, env, n_episodes)
    print(f"Episodes: {n_episodes} | Mean reward: {mean:.2f} | Std: {std:.2f}")

    env.close()