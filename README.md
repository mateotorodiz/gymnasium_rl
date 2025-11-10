# Gymnasium RL - Reinforcement Learning Experiments

A comprehensive repository for learning and experimenting with reinforcement learning algorithms using the Gymnasium API (formerly OpenAI Gym). This repository contains implementations of both **online RL** and **offline RL** algorithms across various custom and standard environments.

## 🎯 Repository Purpose

This repository serves as a personal learning platform for reinforcement learning, prioritizing rapid experimentation and algorithm understanding over production-grade code quality. It demonstrates practical implementations of RL concepts from tabular methods to deep RL.

## 📁 Repository Structure

```
gymnasium_rl/
├── 00_online_rl/          # Online RL experiments
│   └── src/
│       ├── GridWorld/      # Custom grid navigation environment
│       ├── JobSchedule/    # Custom job scheduling environment
│       ├── st_bsline/      # Stable Baselines3 experiments
│       └── stage6_experiments/  # SAC configuration experiments
│
├── 01_offline_rl/         # Offline RL experiments
│   ├── src/               # Basic offline RL examples
│   ├── OOP_Pipeline/      # Object-oriented offline RL framework
│   └── datasets/          # Offline datasets
│
└── d3rlpy_logs/           # Training logs and results
```

## 🚀 Key Components

### 1. Online Reinforcement Learning (`00_online_rl/`)

Online RL involves learning from interactions with the environment in real-time.

#### Custom Environments

##### **GridWorld Environment**
- **Description**: A flexible grid-based navigation environment where an agent learns to reach a target while avoiding a cliff region
- **Features**:
  - Configurable grid size (default: 4×20)
  - Cliff region with large negative rewards (-100)
  - Target location with zero reward upon reaching
  - Step penalty to encourage efficient paths
  - Optional stochastic transitions (20% chance of random action)
- **Agents Implemented**:
  - **SARSA Agent** (`GridWorldSARSAgent.py`): On-policy TD learning with epsilon-greedy exploration
  - **Monte Carlo Agent** (`GridWorldMCAgent.py`): First-visit MC for episodic tasks
  - **Q-Learning Agent** (`jonathan/QLearningAgent.py`): Off-policy TD learning
- **Key Features**:
  - Adaptive learning rate based on state-action visit counts
  - Epsilon scheduling (decreases with state visits)
  - Training progress monitoring with rolling reward averages
  - Learning curve visualization via `Trainer.py`

##### **JobSchedule Environment**
- **Description**: A job scheduling simulation where an agent assigns jobs to machines to maximize efficiency
- **Features**:
  - Multiple machines (configurable count)
  - Jobs with random duration (1-10 time units)
  - Maximum 20 jobs per episode
  - Rewards:
    - +10 for completing a job
    - -1 for invalid job assignment (to busy machine)
    - -1 per timestep as step penalty
- **Agents Implemented**:
  - **SARSA Agent** (`JobScheduleSARSAgent.py`): Learns optimal scheduling policy
  - **Greedy Agent** (`JobScheduleGreedyAgent.py`): Baseline greedy heuristic
- **State Space**: Machine states (idle/busy) + time remaining + jobs remaining
- **Action Space**: Discrete (wait or assign to specific machine)

#### Stable Baselines3 Experiments (`st_bsline/`)

Implementations using the popular Stable Baselines3 library for deep RL:

- **CartPole-v1** (`stage1_cartpole.py`): PPO agent learning balance control
- **MountainCar** (`MountainCar.py`): Classic control problem with A2C
- **Pendulum** (`stage5_pendulum.py`, `stage5_pendulum_optimized.py`): Continuous control with optimizations
- **AcroBot** (`AcroBot.py`): Underactuated double pendulum swing-up
- **Multi-Environment Training** (`stage4_multi_cartpole.py`): Parallel environment training with SubprocVecEnv
- **Snake Environment** (`SnekEnv.py`, `gymnasium_snake.py`): Custom Snake game implementation with PPO/SAC agents

**Key Learning Points** (from README):
- Hyperparameter tuning (entropy coefficient, clipping, learning rate)
- Tensorboard integration for training visualization
- Callback systems (checkpoints, evaluation, early stopping)
- Multi-environment parallel training
- Debugging catastrophic forgetting

#### Advanced SAC Experiments (`stage6_experiments/`)

Soft Actor-Critic (SAC) algorithm with configurable hyperparameters:
- JSON-based configuration system
- Multiple experimental setups (default, low learning rate, high batch size)
- Pendulum-v1 environment testing
- Tensorboard logging for all experiments
- Evaluation callbacks with best model saving

### 2. Offline Reinforcement Learning (`01_offline_rl/`)

Offline RL involves learning from pre-collected datasets without environment interaction.

#### Basic Offline RL (`src/`)

- **`cartpole-example.py`**: Conservative Q-Learning (CQL) on CartPole
  - Uses d3rlpy library
  - Trains on pre-collected CartPole dataset
  - 10,000 training steps with environment evaluation
  - CUDA support for GPU acceleration
  - Model saving and Tensorboard logging

- **`cartpole-evaluator.py`**: Evaluation script for trained offline models
- **`cuda.py`**: CUDA/GPU availability checker

#### OOP Pipeline Framework (`OOP_Pipeline/`)

A well-structured, reusable framework for offline RL experiments:

##### **DataSetCreator Package**
- **`D3rlpyCreator`**: Unified interface for loading/creating d3rlpy-compatible datasets
  - Supports both discrete and continuous action spaces
  - Built-in dataset loaders for standard environments
  - Extensible for custom dataset creation

##### **OfflineTrainer Package**
- **`OfflineTrainer`**: Main training orchestrator
  - Flexible algorithm configuration (CQL, BC, IQL, etc.)
  - Integrated evaluation during training
  - Model saving and loading
  - Tensorboard logging
  - Progress bar support
- **`FitConfig`**: Training configuration dataclass
  - Episode/step control
  - Logging and experiment naming
  - Evaluation frequency settings

##### **Example Pipeline** (`example_pipeline.py`)
Complete end-to-end workflow demonstrating:
1. Dataset loading with `D3rlpyCreator`
2. Algorithm configuration (DiscreteCQL)
3. Training with `OfflineTrainer`
4. Model saving
5. Policy evaluation

**Supported Algorithms** (via d3rlpy):
- Conservative Q-Learning (CQL, DiscreteCQL)
- Behavioral Cloning (BC)
- Implicit Q-Learning (IQL)
- And all d3rlpy algorithms

## 🛠️ Technologies & Libraries

### Core RL Libraries
- **Gymnasium**: Modern environment API (successor to OpenAI Gym)
- **Stable Baselines3**: State-of-the-art deep RL algorithms
- **d3rlpy**: Offline RL algorithms implementation

### Deep Learning & Utilities
- **NumPy**: Numerical computations
- **Matplotlib**: Plotting and visualization
- **Tensorboard**: Training monitoring and visualization
- **tqdm**: Progress bars
- **OpenCV (cv2)**: Image processing for Snake environment

### Algorithms Implemented

#### Tabular Methods (Custom Implementations)
- **SARSA**: On-policy temporal difference learning
- **Q-Learning**: Off-policy temporal difference learning
- **Monte Carlo**: First-visit MC for episodic tasks

#### Deep RL Methods (via Stable Baselines3)
- **PPO**: Proximal Policy Optimization
- **SAC**: Soft Actor-Critic
- **A2C**: Advantage Actor-Critic

#### Offline RL Methods (via d3rlpy)
- **CQL**: Conservative Q-Learning
- **BC**: Behavioral Cloning
- **IQL**: Implicit Q-Learning

## 📊 Key Features

### Training & Evaluation
- **Adaptive Learning**: Learning rates that adjust based on state-action visit counts
- **Epsilon Scheduling**: Exploration rate decay based on state visits
- **Progress Monitoring**: Real-time tracking of Q-value changes and reward averages
- **Learning Curves**: Automated plotting and visualization of training progress
- **Tensorboard Integration**: Comprehensive training metrics and visualization

### Environment Features
- **Flexible Configuration**: Easily adjustable environment parameters (grid size, number of machines, etc.)
- **Stochastic Transitions**: Optional noise in environments for robustness testing
- **Custom Reward Shaping**: Carefully designed reward structures for efficient learning
- **Gymnasium-Compatible**: All custom environments follow Gymnasium API standards

### Code Organization
- **Modular Design**: Separate agent and environment implementations
- **Reusable Components**: OOP pipeline for offline RL experiments
- **Configuration Management**: JSON-based experiment configuration
- **Parallel Training**: Multi-environment support for faster training

## 🎓 Learning Objectives & Insights

From the repository's documented lessons:

### Hyperparameter Insights
- **Entropy Coefficient**: Keep at 0.0 by default; increase only if stuck at local maxima (but beware catastrophic forgetting)
- **Clipping Parameters**: Balance between preventing catastrophic forgetting and enabling learning
- **Learning Rate**: Critical for stability vs. speed trade-off
- **Batch Size (n_steps)**: Affects training stability and speed

### Best Practices
- Start with Tensorboard for debugging and monitoring
- Use progress bars (`progress_bar=True`) for training visibility
- Implement checkpoint callbacks for long training runs
- Use evaluation callbacks for periodic testing
- Monitor training times to catch inefficient code
- Be careful with multi-environment training (use proper Monitor wrapper with ranks)

### Multi-Environment Training
- Single environment → slow training
- Too many environments → excessive overhead
- Prefer `SubprocVecEnv` for multi-core utilization
- Use proper Monitor CSV file handling with ranks

## 🚦 Getting Started

### Prerequisites
```bash
pip install gymnasium
pip install stable-baselines3
pip install d3rlpy
pip install numpy matplotlib tqdm opencv-python
```

### Running Experiments

#### GridWorld SARSA Training
```python
from GridWorld.GridWorldEnv import GridWorldEnv
from GridWorld.GridWorldSARSAgent import GridWorldSARSAgent

env = GridWorldEnv(size=(4, 20))
agent = GridWorldSARSAgent(env, Nzero=10, n_episodes=5000)
agent.train()
agent.evaluate()
```

#### Job Scheduling with SARSA
```python
from JobSchedule.JobScheduleEnv import JobScheduleEnv
from JobSchedule.JobScheduleSARSAgent import JobScheduleSARSAgent

env = JobScheduleEnv(n_machines=10)
agent = JobScheduleSARSAgent(env, n_episodes=10000, Nzero=10, learning_rate=0.1)
agent.train()
agent.evaluate()
```

#### Stable Baselines3 CartPole
```python
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

#### Offline RL Pipeline
```python
from OOP_Pipeline.DataSetCreator import D3rlpyCreator
from OOP_Pipeline.OfflineTrainer import OfflineTrainer, FitConfig
import d3rlpy

# Load dataset
creator = D3rlpyCreator(envname="CartPole-v1", discrete=True)
dataset, _ = creator.get_dataset()

# Configure and train
algo_config = d3rlpy.algos.DiscreteCQLConfig()
fit_config = FitConfig(n_steps=10000, n_steps_per_epoch=1000)
trainer = OfflineTrainer(env=env, dataset=dataset, algo_config=algo_config, 
                        fit_config=fit_config, model_path="model.pt")
trainer.fit_model()
```

## 📈 Training Artifacts

The repository generates various training artifacts:
- **Model Checkpoints**: Saved models in `.zip` or `.pt` format
- **Tensorboard Logs**: Training metrics in `logs/` and `d3rlpy_logs/`
- **Monitor Files**: Episode statistics in `Monitor.csv`
- **Learning Curves**: Matplotlib plots showing training progress

## ⚠️ Known Limitations

- **Code Quality**: Prioritizes learning speed over production best practices
- **Documentation**: Minimal inline documentation (intentional for learning)
- **Testing**: No formal test suite
- **Datasets**: Offline RL datasets not included in repository (generated on-demand)

## 🎯 Use Cases

This repository is ideal for:
- **Learning RL Fundamentals**: From tabular methods to deep RL
- **Algorithm Comparison**: Testing different approaches on same environment
- **Hyperparameter Experimentation**: Systematic tuning and evaluation
- **Custom Environment Development**: Templates for Gymnasium-compatible environments
- **Offline RL Exploration**: Understanding batch RL from fixed datasets
- **Research Prototyping**: Quick iteration on RL ideas

## 📝 Notes

- The repository intentionally maintains a "learning-focused" structure rather than production code
- Many experiments are documented through code and Tensorboard rather than formal documentation
- The `jonathan/` subfolder contains alternative implementations for comparison
- GPU support available for offline RL (d3rlpy) experiments

## 🤝 Contact

This repository is maintained for personal learning purposes. For questions or collaboration, reach out via:
- University email (for coworkers)
- LinkedIn (for personal contacts)

---

**Repository Philosophy**: "Minimum structure for maximum learning" - focusing on rapid experimentation and algorithm understanding rather than software engineering perfection.
