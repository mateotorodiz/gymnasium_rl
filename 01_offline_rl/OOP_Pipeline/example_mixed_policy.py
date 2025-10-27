"""Example: Create dataset from mixed policies and train offline RL agent."""
import sys
from pathlib import Path

# Add the parent directory to sys.path to enable imports
basedir = Path(__file__).resolve().parent
if str(basedir) not in sys.path:
    sys.path.insert(0, str(basedir))

import gymnasium as gym
import d3rlpy
from DataSetCreator import MixedPolicyDatasetCreator
from OfflineTrainer import OfflineTrainer, FitConfig, evaluate


def main():
    """
    Demonstrate creating a mixed-quality dataset and training offline RL.
    
    Workflow:
    1. Train a "expert" policy online (or load pretrained)
    2. Create dataset mixing expert (80%) and random (20%) policies
    3. Train offline RL algorithm on mixed dataset
    4. Evaluate the trained offline policy
    """
    
    # Configuration
    env_name = "CartPole-v1"
    n_collection_episodes = 100
    model_path = "models/mixed_policy_cql.pt"
    buffer_path = "datasets/mixed_cartpole.h5"
    
    # Step 1: Create or load an "expert" policy
    print("Step 1: Creating expert policy...")
    env = gym.make(env_name)
    
    # Option A: Train a simple DQN expert (quick for demo)
    print("Training expert DQN policy...")
    expert_config = d3rlpy.algos.DQNConfig(learning_rate=6.25e-5)
    expert_policy = expert_config.create(device=False)
    
    # Quick online training (just for demo - not a real expert)
    expert_buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=100000, env=env)
    explorer = d3rlpy.algos.ConstantEpsilonGreedy(0.3)
    
    print("Training expert for 5000 steps...")
    expert_policy.fit_online(
        env,
        expert_buffer,
        explorer=explorer,
        n_steps=5000,
        n_steps_per_epoch=1000,
        show_progress=True
    )
    
    # Option B: Use a pretrained model (uncomment if you have one)
    # expert_policy = d3rlpy.load_learnable("path/to/expert.d3")
    
    # Step 2: Create mixed-policy dataset
    print("\nStep 2: Creating mixed-policy dataset...")
    
    # Create random policy
    random_policy = d3rlpy.algos.DiscreteRandomPolicyConfig().create()
    
    # Create dataset creator with 80% expert, 20% random
    creator=MixedPolicyDatasetCreator(
        env=env,
        policy=expert_policy,
        buffer_size=100000
    )
    buffer = creator.create_dataset()

    # Step 3: Train offline RL on mixed dataset
    print("\nStep 3: Training offline RL (CQL) on mixed dataset...")
    
    algo_config = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=6.25e-5,
        batch_size=32,
        gamma=0.99,
    )
    
    fit_config = FitConfig(
        n_steps=10000,
        n_steps_per_epoch=1000,
        experiment_name="mixed_policy_cql",
        show_progress=True,
        logger_adapter=d3rlpy.logging.TensorboardAdapterFactory(root_dir='logs')
    )
    
    trainer = OfflineTrainer(
        env=env,
        dataset=buffer,
        algo_config=algo_config,
        fit_config=fit_config,
        model_path=model_path,
        device=False
    )
    
    trainer.fit_model()
    trainer.save_model()
    
    # Step 4: Evaluate all policies
    print("\nStep 4: Evaluating policies...")
    n_eval_episodes = 20
    
    # Evaluate expert
    mean_expert, std_expert = evaluate(expert_policy, env, n_eval_episodes)
    print(f"Expert policy: {mean_expert:.2f} ± {std_expert:.2f}")
    
    # Evaluate random
    mean_random, std_random = evaluate(random_policy, env, n_eval_episodes)
    print(f"Random policy: {mean_random:.2f} ± {std_random:.2f}")
    
    # Evaluate trained offline policy
    mean_offline, std_offline = evaluate(trainer.algo, env, n_eval_episodes)
    print(f"Offline CQL policy: {mean_offline:.2f} ± {std_offline:.2f}")
    
    env.close()
    print("\nPipeline completed successfully!")
    print(f"\nDataset saved to: {buffer_path}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
