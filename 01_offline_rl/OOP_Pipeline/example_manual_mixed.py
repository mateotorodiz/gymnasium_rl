"""Example: Manual mixed-policy dataset creation with multiple policies."""
import sys
from pathlib import Path

basedir = Path(__file__).resolve().parent
if str(basedir) not in sys.path:
    sys.path.insert(0, str(basedir))

import gymnasium as gym
import d3rlpy
from DataSetCreator import ManualMDPDatasetCreator


def main():
    """
    Demonstrate manual mixed-policy dataset creation.
    
    This shows how ManualMDPDatasetCreator can mix multiple policies
    using incremental buffer writes (append/clip_episode).
    """
    
    # Configuration
    env_name = "CartPole-v1"
    
    # Create environment
    env = gym.make(env_name)
    
    # Create two different policies to mix
    random_policy1 = d3rlpy.algos.DiscreteRandomPolicyConfig().create()
    random_policy2 = d3rlpy.algos.DiscreteRandomPolicyConfig().create()
    
    # Create mixed dataset: 30 episodes from policy1, 20 episodes from policy2
    print("Creating mixed-policy dataset manually...")
    creator = ManualMDPDatasetCreator(
        env=env,
        policies=[random_policy1, random_policy2],
        episodes_per_policy=[30, 20]
    )
    
    dataset = creator.create_dataset(buffer_limit=100000)
    
    # Train CQL on the mixed dataset
    print("\nTraining CQL on mixed dataset...")
    cql = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=6.25e-5,
        batch_size=32,
    ).create(device=False)
    
    cql.fit(
        dataset,
        n_steps=3000,
        n_steps_per_epoch=1000,
        show_progress=True
    )
    
    # Evaluate
    print("\nEvaluating trained policy...")
    eval_env = gym.make(env_name)
    total_reward = 0
    n_eval = 10
    
    for _ in range(n_eval):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            import numpy as np
            action = cql.predict(np.array([obs]))[0]
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_reward += episode_reward
    
    print(f"Average reward over {n_eval} episodes: {total_reward / n_eval:.2f}")
    
    env.close()
    eval_env.close()
    print("\nMixed-policy pipeline completed successfully!")


if __name__ == "__main__":
    main()
