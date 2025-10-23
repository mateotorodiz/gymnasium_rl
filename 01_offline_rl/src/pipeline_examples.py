"""
Example: Using the Offline RL Pipeline

This script shows various ways to use the offline_rl_pipeline.py module
for different use cases.
"""

from offline_rl_pipeline import OfflineRLExperiment, ExperimentConfig
from pathlib import Path


def example_1_basic_usage():
    """Example 1: Basic usage - run everything from scratch."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage - Full Pipeline")
    print("="*70)
    
    config = ExperimentConfig(
        env_name="Pendulum-v1",
        expert_timesteps=50_000,      # Reduced for faster demo
        n_dataset_episodes=50,        # Smaller dataset
        offline_training_steps=10_000, # Faster training
        eval_episodes=5
    )
    
    experiment = OfflineRLExperiment(config)
    results = experiment.run()
    
    print("\nResults:")
    for policy_name, (mean, std) in results.items():
        print(f"  {policy_name}: {mean:.2f} ± {std:.2f}")


def example_2_load_expert():
    """Example 2: Load pre-trained expert, only train offline algorithms."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Load Pre-trained Expert")
    print("="*70)
    
    config = ExperimentConfig(
        expert_timesteps=200_000,
        n_dataset_episodes=100,
        offline_training_steps=50_000
    )
    
    experiment = OfflineRLExperiment(config)
    
    # Skip expert training if model exists
    results = experiment.run(skip_expert_training=True)
    
    print("\nResults:")
    for policy_name, (mean, std) in results.items():
        print(f"  {policy_name}: {mean:.2f} ± {std:.2f}")


def example_3_evaluation_only():
    """Example 3: Load everything - just evaluate."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Evaluation Only")
    print("="*70)
    
    config = ExperimentConfig(eval_episodes=20)
    experiment = OfflineRLExperiment(config)
    
    # Paths to pre-trained models
    bc_path = r"c:\Users\tomt886\PythonProjects\gymnasium_rl\d3rlpy_logs\BC_20251021152546\model_100000.d3"
    
    results = experiment.run(
        skip_expert_training=True,
        skip_dataset_generation=True,
        bc_load_path=bc_path if Path(bc_path).exists() else None,
        cql_load_path=None  # Will train CQL if no path provided
    )
    
    print("\nResults:")
    for policy_name, (mean, std) in results.items():
        print(f"  {policy_name}: {mean:.2f} ± {std:.2f}")


def example_4_custom_environment():
    """Example 4: Use different environment."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Environment (MountainCarContinuous)")
    print("="*70)
    
    config = ExperimentConfig(
        env_name="MountainCarContinuous-v0",
        expert_timesteps=100_000,
        n_dataset_episodes=100,
        offline_training_steps=50_000,
        seed=123
    )
    
    experiment = OfflineRLExperiment(config)
    results = experiment.run()
    
    print("\nResults:")
    for policy_name, (mean, std) in results.items():
        print(f"  {policy_name}: {mean:.2f} ± {std:.2f}")


def example_5_hyperparameter_sweep():
    """Example 5: Run multiple experiments with different configs."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Hyperparameter Sweep")
    print("="*70)
    
    dataset_sizes = [50, 100, 200]
    all_results = {}
    
    for n_episodes in dataset_sizes:
        print(f"\n--- Testing with {n_episodes} episodes ---")
        
        config = ExperimentConfig(
            expert_timesteps=50_000,
            n_dataset_episodes=n_episodes,
            offline_training_steps=20_000,
            eval_episodes=5
        )
        
        experiment = OfflineRLExperiment(config)
        results = experiment.run(skip_expert_training=True)
        
        all_results[f"dataset_{n_episodes}"] = results
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON ACROSS DATASET SIZES")
    print("="*70)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        for policy_name, (mean, std) in results.items():
            print(f"  {policy_name}: {mean:.2f} ± {std:.2f}")


def example_6_deterministic_vs_stochastic():
    """Example 6: Compare deterministic vs stochastic expert data."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Deterministic vs Stochastic Expert Data")
    print("="*70)
    
    # Experiment 1: Stochastic expert (with exploration noise)
    print("\n--- Stochastic Expert Data ---")
    config_stochastic = ExperimentConfig(
        use_deterministic_expert=False,
        n_dataset_episodes=100,
        offline_training_steps=30_000
    )
    
    exp_stochastic = OfflineRLExperiment(config_stochastic)
    results_stochastic = exp_stochastic.run(skip_expert_training=True)
    
    # Experiment 2: Deterministic expert (pure greedy)
    print("\n--- Deterministic Expert Data ---")
    config_deterministic = ExperimentConfig(
        use_deterministic_expert=True,
        n_dataset_episodes=100,
        offline_training_steps=30_000
    )
    
    exp_deterministic = OfflineRLExperiment(config_deterministic)
    results_deterministic = exp_deterministic.run(skip_expert_training=True)
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON: Stochastic vs Deterministic")
    print("="*70)
    
    print("\nStochastic Expert Data:")
    for policy_name, (mean, std) in results_stochastic.items():
        print(f"  {policy_name}: {mean:.2f} ± {std:.2f}")
    
    print("\nDeterministic Expert Data:")
    for policy_name, (mean, std) in results_deterministic.items():
        print(f"  {policy_name}: {mean:.2f} ± {std:.2f}")


def example_7_configuration_management():
    """Example 7: Save and load configurations."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Configuration Management")
    print("="*70)
    
    # Create custom configuration
    config = ExperimentConfig(
        env_name="Pendulum-v1",
        expert_timesteps=150_000,
        n_dataset_episodes=200,
        offline_training_steps=100_000,
        seed=999
    )
    
    # Save configuration
    config_path = Path("my_experiment_config.json")
    config.save(config_path)
    print(f"✓ Saved configuration to {config_path}")
    
    # Load configuration
    loaded_config = ExperimentConfig.load(config_path)
    print(f"✓ Loaded configuration from {config_path}")
    
    print(f"\nLoaded config:")
    print(f"  Environment: {loaded_config.env_name}")
    print(f"  Expert timesteps: {loaded_config.expert_timesteps}")
    print(f"  Dataset episodes: {loaded_config.n_dataset_episodes}")
    print(f"  Seed: {loaded_config.seed}")
    
    # Run experiment with loaded config
    experiment = OfflineRLExperiment(loaded_config)
    # results = experiment.run()  # Uncomment to run
    
    # Clean up
    if config_path.exists():
        config_path.unlink()
        print(f"\n✓ Cleaned up {config_path}")


def example_8_quick_test():
    """Example 8: Quick test with minimal resources."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Quick Test (Minimal Resources)")
    print("="*70)
    
    config = ExperimentConfig(
        expert_timesteps=10_000,       # Very fast
        n_dataset_episodes=20,         # Small dataset
        offline_training_steps=5_000,  # Quick training
        eval_episodes=3,               # Fast eval
        n_parallel_envs=2              # Less CPU
    )
    
    experiment = OfflineRLExperiment(config)
    results = experiment.run()
    
    print("\nResults (may not be optimal due to limited training):")
    for policy_name, (mean, std) in results.items():
        print(f"  {policy_name}: {mean:.2f} ± {std:.2f}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Offline RL Pipeline Examples")
    print("="*70)
    print("\nChoose an example to run:")
    print("1. Basic Usage - Full Pipeline")
    print("2. Load Pre-trained Expert")
    print("3. Evaluation Only")
    print("4. Custom Environment (MountainCarContinuous)")
    print("5. Hyperparameter Sweep")
    print("6. Deterministic vs Stochastic Expert Data")
    print("7. Configuration Management")
    print("8. Quick Test (Minimal Resources)")
    print("\nOr modify this file to run specific examples!")
    
    # Uncomment the example you want to run:
    
    # example_1_basic_usage()
    # example_2_load_expert()
    # example_3_evaluation_only()
    # example_4_custom_environment()
    # example_5_hyperparameter_sweep()
    # example_6_deterministic_vs_stochastic()
    # example_7_configuration_management()
    example_8_quick_test()  # Fastest option for testing
