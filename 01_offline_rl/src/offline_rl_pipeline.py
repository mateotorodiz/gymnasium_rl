"""
Offline RL Pipeline for Pendulum-v1
Demonstrates best practices for offline RL experimentation.

This modular pipeline separates concerns into distinct classes:
- ExpertPolicyTrainer: Trains/loads online RL expert
- OfflineDatasetGenerator: Collects expert demonstrations
- OfflineRLTrainer: Trains offline RL algorithms
- PolicyEvaluator: Evaluates and compares policies
- OfflineRLExperiment: Orchestrates the full pipeline
"""
import gymnasium as gym
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, asdict
import torch
import json
from datetime import datetime

from stable_baselines3 import SAC, PPO, A2C, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from d3rlpy.algos import (
    BC,
    CQL,
    BCConfig,
    CQLConfig,
    DiscreteBC,
    DiscreteCQL,
    DiscreteBCConfig,
    DiscreteCQLConfig,
)
from d3rlpy.dataset import MDPDataset
import d3rlpy


@dataclass
class ExperimentConfig:
    """Configuration for offline RL experiment."""
    # Environment
    env_name: str = "CartPole-v1"
    seed: int = 42
    # Expert algorithm for online data collection {"PPO", "A2C", "DQN", "SAC"}
    expert_algo: str = "PPO"
    
    # Expert training
    expert_timesteps: int = 200_000
    n_parallel_envs: int = 4
    
    # Dataset generation
    n_dataset_episodes: int = 100
    use_deterministic_expert: bool = False  # Set to True for pure expert data
    
    # Offline training
    offline_training_steps: int = 100_000
    
    # Evaluation
    eval_episodes: int = 10
    
    # Device
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Paths (set automatically)
    base_dir: Path = Path(__file__).parent.parent
    
    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"
    
    @property
    def datasets_dir(self) -> Path:
        return self.base_dir / "datasets"
    
    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"
    
    def save(self, path: Path):
        """Save configuration to JSON."""
        with open(path, 'w') as f:
            # Convert Path objects to strings for JSON serialization
            config_dict = asdict(self)
            config_dict['base_dir'] = str(self.base_dir)
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
            config_dict['base_dir'] = Path(config_dict['base_dir'])
            return cls(**config_dict)


class ExpertPolicyTrainer:
    """Handles training and loading of online RL expert policy."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        # Save per algorithm and env
        self.model_path = self.config.models_dir / f"{self.config.expert_algo.lower()}_{self.config.env_name}"
    
    def _make_env(self):
        """Environment factory for vectorized training."""
        return lambda: gym.make(self.config.env_name)
    
    def train(self):
        """Train expert with selected algorithm and appropriate env setup."""
        print(f"\n{'='*60}")
        print(f"Training {self.config.expert_algo.upper()} expert on {self.config.env_name}")
        print(f"{'='*60}")
        
        algo = self.config.expert_algo.lower()

        # Check action space compatibility
        tmp_env = gym.make(self.config.env_name)
        is_discrete = isinstance(tmp_env.action_space, gym.spaces.Discrete)
        tmp_env.close()

        if algo == "sac" and is_discrete:
            raise AssertionError("SAC requires a continuous (Box) action space. Choose PPO, A2C, or DQN for discrete envs.")

        # Use vectorized envs where sensible
        if algo in ("ppo", "a2c", "sac"):
            env = SubprocVecEnv([self._make_env() for _ in range(self.config.n_parallel_envs)])
        else:
            env = gym.make(self.config.env_name)

        # Instantiate algorithm
        if algo == "ppo":
            model = PPO("MlpPolicy", env, verbose=1, seed=self.config.seed)
        elif algo == "a2c":
            model = A2C("MlpPolicy", env, verbose=1, seed=self.config.seed)
        elif algo == "dqn":
            model = DQN("MlpPolicy", env, verbose=1, seed=self.config.seed)
        elif algo == "sac":
            model = SAC("MlpPolicy", env, verbose=1, seed=self.config.seed)
        else:
            raise ValueError(f"Unsupported expert_algo: {self.config.expert_algo}")

        model.learn(total_timesteps=self.config.expert_timesteps, progress_bar=True)
        model.save(str(self.model_path))
        print(f"\n✓ Expert saved to: {self.model_path}")
        env.close()
        return model
    
    def load_or_train(self):
        """Load existing expert or train new one."""
        if self.model_path.with_suffix('.zip').exists():
            print(f"\n✓ Loading existing expert from {self.model_path}")
            algo = self.config.expert_algo.lower()
            if algo == "ppo":
                return PPO.load(str(self.model_path))
            elif algo == "a2c":
                return A2C.load(str(self.model_path))
            elif algo == "dqn":
                return DQN.load(str(self.model_path))
            elif algo == "sac":
                return SAC.load(str(self.model_path))
            else:
                raise ValueError(f"Unsupported expert_algo: {self.config.expert_algo}")
        else:
            return self.train()


class OfflineDatasetGenerator:
    """Generates offline dataset from expert policy rollouts."""
    
    def __init__(self, config: ExperimentConfig, expert_policy: Any):
        self.config = config
        self.expert = expert_policy
        self.config.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.config.datasets_dir / f"{config.env_name}_dataset.h5"
    
    def generate(self) -> MDPDataset:
        """Roll out expert to collect offline dataset (always regenerate)."""
        print(f"\n{'='*60}")
        print(f"Generating offline dataset")
        print(f"{'='*60}")
        print(f"Episodes: {self.config.n_dataset_episodes}")
        print(f"Deterministic: {self.config.use_deterministic_expert}")

        env = gym.make(self.config.env_name)

        episodes: List[List[Tuple]] = []
        current_episode: List[Tuple] = []
        obs, _ = env.reset(seed=self.config.seed)

        episode_count = 0
        total_reward = 0.0
        episode_rewards: List[float] = []

        while len(episodes) < self.config.n_dataset_episodes:
            action, _ = self.expert.predict(obs, deterministic=self.config.use_deterministic_expert)
            # Cast action properly for discrete envs
            if isinstance(env.action_space, gym.spaces.Discrete):
                if isinstance(action, np.ndarray):
                    action = int(action.item())
                else:
                    action = int(action)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            current_episode.append((obs, action, reward, next_obs, terminated or truncated))
            total_reward += float(reward)

            if terminated or truncated:
                episodes.append(current_episode)
                episode_rewards.append(total_reward)
                current_episode = []
                total_reward = 0.0
                obs, _ = env.reset()
                episode_count += 1

                if episode_count % 20 == 0:
                    print(f"  Progress: {episode_count}/{self.config.n_dataset_episodes} episodes")
            else:
                obs = next_obs

        env.close()

        # Print dataset statistics
        print(f"\n{'='*60}")
        print(f"Dataset Statistics")
        print(f"{'='*60}")
        print(f"Episodes: {len(episode_rewards)}")
        if episode_rewards:
            print(f"Mean Return: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"Min/Max Return: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")

        dataset = self._episodes_to_dataset(episodes)
        return dataset
            
    def _episodes_to_dataset(self, episodes: List[List[Tuple]]) -> MDPDataset:
        """Convert episode list to MDPDataset."""
        transitions = [trans for ep in episodes for trans in ep]
        print(f"Total transitions: {len(transitions)}")
        
        observations = np.stack([t[0] for t in transitions])
        actions = np.stack([t[1] for t in transitions])
        rewards = np.array([t[2] for t in transitions])
        terminals = np.array([t[4] for t in transitions])
        
        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals
        )
        
        dataset.dump(str(self.dataset_path))
        print(f"✓ Dataset saved to: {self.dataset_path}")
        return dataset


class OfflineRLTrainer:
    """Trains offline RL algorithms on static dataset."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        self.bc_model_path = self.config.models_dir / f"bc_{config.env_name}.d3"
        self.cql_model_path = self.config.models_dir / f"cql_{config.env_name}.d3"
    
    def _is_discrete_dataset(self, dataset: MDPDataset) -> bool:
        """Infer whether the dataset represents a discrete action space.
        Uses action array shape/dtype as a robust heuristic across d3rlpy versions.
        """
        try:
            actions = dataset.actions
            if actions is None:
                return False
            # Discrete datasets are typically 1D ints; continuous are 2D floats
            if actions.ndim == 1:
                return True
            return np.issubdtype(actions.dtype, np.integer)
        except Exception:
            return False
    
    def train_bc(self, dataset: MDPDataset) -> BC:
        """Train Behavioral Cloning baseline."""
        # Load existing model if available
        if self.bc_model_path.exists():
            print(f"\n✓ Loading existing BC model from {self.bc_model_path}")
            return d3rlpy.load_learnable(str(self.bc_model_path))
        
        print(f"\n{'='*60}")
        print("Training BC (Behavioral Cloning)")
        print(f"{'='*60}")
        
        # Choose discrete or continuous BC based on dataset
        if self._is_discrete_dataset(dataset):
            bc_config = DiscreteBCConfig()
            bc = DiscreteBC(bc_config, device=self.config.device, enable_ddp=False)
        else:
            bc_config = BCConfig()
            bc = BC(bc_config, device=self.config.device, enable_ddp=False)
        bc.fit(dataset, n_steps=self.config.offline_training_steps)
        
        # Save the trained model
        bc.save(str(self.bc_model_path))
        print(f"✓ BC training complete, saved to {self.bc_model_path}")
        return bc
    
    def train_cql(self, dataset: MDPDataset) -> CQL:
        """Train Conservative Q-Learning."""
        # Load existing model if available
        if self.cql_model_path.exists():
            print(f"\n✓ Loading existing CQL model from {self.cql_model_path}")
            return d3rlpy.load_learnable(str(self.cql_model_path))
        
        print(f"\n{'='*60}")
        print("Training CQL (Conservative Q-Learning)")
        print(f"{'='*60}")
        
        # Choose discrete or continuous CQL based on dataset
        if self._is_discrete_dataset(dataset):
            cql_config = DiscreteCQLConfig()
            cql = DiscreteCQL(cql_config, device=self.config.device, enable_ddp=False)
        else:
            cql_config = CQLConfig()
            cql = CQL(cql_config, device=self.config.device, enable_ddp=False)
        cql.fit(dataset, n_steps=self.config.offline_training_steps)
        
        # Save the trained model
        cql.save(str(self.cql_model_path))
        print(f"✓ CQL training complete, saved to {self.cql_model_path}")
        return cql


class PolicyEvaluator:
    """Evaluates policies in the environment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def evaluate(self, policy: Any, name: str) -> Tuple[float, float]:
        """Evaluate policy performance."""
        env = gym.make(self.config.env_name)
        total_rewards = []
        
        for ep in range(self.config.eval_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Handle both d3rlpy and SB3 interfaces
                action = self._get_action(policy, obs)
                # Cast action properly for discrete envs
                if isinstance(env.action_space, gym.spaces.Discrete):
                    if isinstance(action, np.ndarray):
                        action = int(action.item())
                    else:
                        action = int(action)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
        
        env.close()
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        print(f"{name:25s} | Mean: {mean_reward:8.2f} | Std: {std_reward:7.2f}")
        return mean_reward, std_reward
    
    def _get_action(self, policy: Any, obs: np.ndarray) -> np.ndarray:
        """Get action from policy, handling different interfaces."""
        # Check if it's a d3rlpy policy
        if 'd3rlpy' in str(type(policy)):
            # d3rlpy expects (batch_size, obs_dim)
            return policy.predict(np.asarray([obs], dtype=np.float32))[0]
        
        # Check if it's a Stable Baselines3 policy
        elif hasattr(policy, 'predict') and callable(policy.predict):
            # SB3 expects (obs_dim,) - single observation
            action, _ = policy.predict(obs, deterministic=True)
            return action
        
        else:
            raise ValueError(f"Unknown policy type: {type(policy)}")


class OfflineRLExperiment:
    """Main experiment orchestrator."""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self._print_header()
    
    def _print_header(self):
        """Print experiment header."""
        print(f"\n{'='*60}")
        print(f"Offline RL Experiment")
        print(f"{'='*60}")
        print(f"d3rlpy version: {d3rlpy.__version__}")
        print(f"Device: {self.config.device}")
        print(f"Environment: {self.config.env_name}")
        print(f"Seed: {self.config.seed}")
        print(f"{'='*60}\n")
    
    def run(self) -> Dict[str, Tuple[float, float]]:
        """
        Execute full offline RL pipeline.
        
        Returns:
            Dictionary with evaluation results for each policy
        """
        # 1. Train or load expert
        expert_trainer = ExpertPolicyTrainer(self.config)
        expert = expert_trainer.load_or_train()
        
        # 2. Generate offline dataset
        dataset_gen = OfflineDatasetGenerator(self.config, expert)
        dataset = dataset_gen.generate()
        
        # 3. Train offline RL algorithms
        offline_trainer = OfflineRLTrainer(self.config)
        bc = offline_trainer.train_bc(dataset)
        cql = offline_trainer.train_cql(dataset)
        
        # 4. Evaluate all policies
        evaluator = PolicyEvaluator(self.config)
        print(f"\n{'='*60}")
        print("Policy Evaluation Results")
        print(f"{'='*60}")
        print(f"{'Policy':<25} | {'Mean':<8} | {'Std':<7}")
        print(f"{'-'*60}")
        
        bc_perf = evaluator.evaluate(bc, "BC (Imitation)")
        cql_perf = evaluator.evaluate(cql, "CQL (Offline RL)")
        expert_perf = evaluator.evaluate(expert, f"{self.config.expert_algo} Expert (Online)")
        
        print(f"{'='*60}\n")
        
        # Save results
        results = {
            'bc': bc_perf,
            'cql': cql_perf,
            'expert': expert_perf
        }
        
        self._save_results(results)
        return results
    
    def _save_results(self, results: Dict[str, Tuple[float, float]]):
        """Save experiment results to JSON."""
        results_path = self.config.logs_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_dict = {
            'config': asdict(self.config),
            'results': {
                k: {'mean': float(v[0]), 'std': float(v[1])} 
                for k, v in results.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert Path to string for JSON serialization
        results_dict['config']['base_dir'] = str(self.config.base_dir)
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"✓ Results saved to: {results_path}")


def main():
    """Main entry point for running experiments."""
    # Configure experiment
    config = ExperimentConfig(
        env_name="CartPole-v1",
        expert_timesteps=200_000,
        n_dataset_episodes=100,
        offline_training_steps=100_000,
        eval_episodes=10,
        seed=42
    )
    
    # Run experiment
    experiment = OfflineRLExperiment(config)
    results = experiment.run()
    
    print("\n✓ Experiment complete!")
    return results


if __name__ == "__main__":
    main()
