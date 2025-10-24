"""Simple example: Create dataset from random and callable policies."""
import sys
from pathlib import Path

basedir = Path(__file__).resolve().parent
if str(basedir) not in sys.path:
    sys.path.insert(0, str(basedir))

import gymnasium as gym
import numpy as np
import d3rlpy
from DataSetCreator import MixedPolicyDatasetCreator
from OfflineTrainer import OfflineTrainer, FitConfig, evaluate


def main():
    """Simple example with random policies and custom callable."""
    
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    
    # Policy 1: d3rlpy random policy
    random_policy = d3rlpy.algos.DiscreteRandomPolicyConfig().create()
    
    # Policy 2: Simple heuristic policy (callable)
    def simple_heuristic(observation):
        """Simple CartPole heuristic: move toward the direction pole is falling."""
        angle = observation[2]  # pole angle
        return 1 if angle > 0 else 0
    
    # Policy 3: Another callable - uniform random using gym's action space
    def gym_random(observation):
        return env.action_space.sample()
    
    # Create dataset with 50% d3rlpy random, 30% heuristic, 20% gym random
    print("Creating dataset with mixed policies...")
    creator = MixedPolicyDatasetCreator(
        env=env,
        policies=[random_policy, simple_heuristic, gym_random],
        policy_probs=[0.5, 0.3, 0.2],
        buffer_size=50000
    )
    
    # Collect 50 episodes
    buffer = creator.collect_episodes(n_episodes=50, seed=42)
    
    # Train offline RL
    print("\nTraining offline RL on mixed dataset...")
    algo_config = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=6.25e-5,
        batch_size=32,
        gamma=0.99,
    )
    
    fit_config = FitConfig(
        n_steps=5000,
        n_steps_per_epoch=1000,
        experiment_name="simple_mixed",
        show_progress=True,
        logger_adapter=d3rlpy.logging.TensorboardAdapterFactory(root_dir='logs')
    )
    
    trainer = OfflineTrainer(
        env=env,
        dataset=buffer,
        algo_config=algo_config,
        fit_config=fit_config,
        model_path="models/simple_mixed_cql.pt",
        device=False
    )
    
    trainer.fit_model()
    
    # Evaluate
    print("\nEvaluating trained policy...")
    mean_reward, std_reward = evaluate(trainer.algo, env, n_episodes=20)
    print(f"Trained policy reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
