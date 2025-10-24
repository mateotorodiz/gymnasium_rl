"""Example pipeline using DataSetCreator and OfflineTrainer packages."""
import sys
from pathlib import Path

# Add the parent directory to sys.path to enable imports
basedir = Path(__file__).resolve().parent
if str(basedir) not in sys.path:
    sys.path.insert(0, str(basedir))

import gymnasium as gym
import d3rlpy
from DataSetCreator import D3rlpyCreator
from OfflineTrainer import (
    OfflineTrainer,
    FitConfig,
    evaluate
)


def main():
    """Run a complete offline RL pipeline."""
    # Configuration
    env_name = "CartPole-v1"
    model_path = "models/cql_cartpole_pipeline.pt"
    n_eval_episodes = 20
    
    # Step 1: Create dataset using D3rlpyCreator
    print(f"Loading dataset for {env_name}...")
    dataset_creator = D3rlpyCreator(envname=env_name, discrete=True)
    dataset, _ = dataset_creator.get_dataset()
    
    # Step 2: Create environment for training
    env = gym.make(env_name)
    
    # Step 3: Configure algorithm and training
    # Use d3rlpy config directly (supports any algorithm: DiscreteCQL, CQL, BC, IQL, etc.)
    algo_config = d3rlpy.algos.DiscreteCQLConfig(
        batch_size=32,
        learning_rate=6.25e-5,
        n_critics=1,
    )
    
    fit_config = FitConfig(
        n_steps=10000,
        n_steps_per_epoch=1000,
        experiment_name="pipeline_example",
        show_progress=True,
        logger_adapter=d3rlpy.logging.TensorboardAdapterFactory(root_dir='logs')
    )
    
    # Step 4: Create trainer and train
    print("Creating trainer and starting training...")
    trainer = OfflineTrainer(
        env=env,
        dataset=dataset,
        algo_config=algo_config,
        fit_config=fit_config,
        model_path=model_path,
        device=False  # Use CPU, change to True for cuda:0
    )
    
    trainer.fit_model()
    
    # Step 5: Save model
    trainer.save_model()
    
    # Step 6: Evaluate
    print(f"\nEvaluating trained policy over {n_eval_episodes} episodes...")
    mean_reward, std_reward = evaluate(trainer.algo, env, n_eval_episodes)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    env.close()
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
