import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from d3rlpy.algos import BC, CQL
import d3rlpy

# Set base directory relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "..", "models")
datasets_dir = os.path.join(base_dir, "..", "datasets")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(datasets_dir, exist_ok=True)

print(f"d3rlpy version: {d3rlpy.__version__}")


if __name__ == "__main__":
    def make_env():
        """Create a single environment instance."""
        return lambda: gym.make("Pendulum-v1")

    def train_model():
        """Train the SAC model using SubprocVecEnv."""
        env = SubprocVecEnv([make_env() for _ in range(4)])
        
        model = SAC("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=200_000, progress_bar=True)
        
        model_path = os.path.join(models_dir, "sac_pendulum")
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        env.close()
        return model

    # Load existing model or train new one
    model_path = os.path.join(models_dir, "sac_pendulum")
    if os.path.exists(f"{model_path}.zip"):
        print("Loading existing model...")
        model = SAC.load(model_path)
    else:
        print("Training new model...")
        model = train_model()

    # Create single environment for dataset generation
    env = gym.make("Pendulum-v1")

    # ---- Generate offline dataset with proper episode handling ----
    episodes = []
    current_episode = []
    
    obs, _ = env.reset()
    episode_count = 0
    
    while len(episodes) < 100:  # Collect 100 complete episodes
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Store transition
        current_episode.append((obs, action, reward, next_obs, terminated or truncated))
        
        if terminated or truncated:
            # Episode ended - add to episodes list
            episodes.append(current_episode)
            current_episode = []
            obs, _ = env.reset()
            episode_count += 1
            if episode_count % 10 == 0:
                print(f"Collected {episode_count} episodes")
        else:
            obs = next_obs

    print(f"Collected {len(episodes)} complete episodes")

    # Convert episodes to transitions
    transitions = []
    for episode in episodes:
        transitions.extend(episode)

    print("Total transitions:", len(transitions))

    from d3rlpy.dataset import MDPDataset

    observations = np.stack([t[0] for t in transitions])
    actions = np.stack([t[1] for t in transitions])
    rewards = np.array([t[2] for t in transitions])
    terminals = np.array([t[4] for t in transitions])

    # Create dataset with proper episode structure
    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals
    )
    
    dataset_path = os.path.join(datasets_dir, "pendulum_dataset.h5")
    dataset.dump(dataset_path)
    print(f"Dataset saved to: {dataset_path}")

    # Initialize algorithms with constructor args supported across versions
    bc = BC()  # avoid batch_size/learning_rate/use_gpu in __init__
    cql = CQL(alpha=5.0)  # keep only clearly supported arg

    def fit_algo(algo, name):
        print(f"Training {name}...")
        try:
            # v0.x / some 1.x support n_epochs
            algo.fit(dataset, n_epochs=30, batch_size=256, verbose=True)
        except TypeError:
            # v2.x uses n_steps
            algo.fit(dataset, n_steps=30_000, batch_size=256, verbose=True)

    fit_algo(bc, "BC")
    fit_algo(cql, "CQL")

    # Simple evaluation instead of OPE (to avoid scope_rl issues)
    def evaluate_policy(policy, env, n_episodes=10):
        """Evaluate policy performance."""
        total_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if hasattr(policy, 'predict') and hasattr(policy.predict, '__call__'):
                    # d3rlpy policy
                    action = policy.predict([obs])[0]
                else:
                    # stable_baselines3 policy
                    action, _ = policy.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards), np.std(total_rewards)

    # Evaluate all policies
    print("\n=== Policy Evaluation ===")
    
    bc_mean, bc_std = evaluate_policy(bc, env)
    print(f"BC Policy - Mean: {bc_mean:.2f}, Std: {bc_std:.2f}")
    
    cql_mean, cql_std = evaluate_policy(cql, env)
    print(f"CQL Policy - Mean: {cql_mean:.2f}, Std: {cql_std:.2f}")
    
    sac_mean, sac_std = evaluate_policy(model, env)
    print(f"Original SAC Policy - Mean: {sac_mean:.2f}, Std: {sac_std:.2f}")

    env.close()


