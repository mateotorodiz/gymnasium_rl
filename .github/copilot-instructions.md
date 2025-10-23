# Copilot Instructions for gymnasium_rl
## Behavior
Do not rewrite code for no reason if I dont ask you to. Justify your decisions. Provide me first with the understanding of the problem, then propose changes.
## Project Overview
This repository contains reinforcement learning experiments in custom environments using the Gymnasium API. The main environments are:
- **GridWorld**: Flexible grid-based navigation with cliff and target, supporting SARSA, Monte Carlo, and Q-Learning agents.
- **JobSchedule**: Simulates job assignment to machines, with SARSA and greedy agents optimizing scheduling.

## Architecture & Key Components
- `src/GridWorld/` and `src/JobSchedule/`: Each contains environment (`*Env.py`) and agent (`*Agent.py`) implementations. Agents use tabular RL (Q-table) and epsilon-greedy exploration.
- `Trainer.py` (GridWorld): Centralizes training and evaluation, including reward plotting.
- `jonathan/` subfolder: Contains alternative agent/environment implementations for GridWorld.

## Patterns & Conventions
- **Agent Design**: Agents expect a Gymnasium environment and implement `train()` and `evaluate()` methods. Q-values are stored in `defaultdict` keyed by state.
- **Epsilon-Greedy**: Epsilon is often scheduled by state visit counts (see `get_epsilon()`), not fixed decay.
- **Learning Rate**: Alpha adapts per state-action visit (see `get_alpha()`), not fixed.
- **Episode Loop**: Training loops follow the SARSA pattern: select action, step, update Q, repeat until done.
- **Monitoring**: Agents track rolling average rewards and Q-value changes for progress (see `monitor_progress()`).

## Workflows
- **Training**: Instantiate environment and agent, then call `agent.train()`. Example (JobSchedule):
  ```python
  env = JobScheduleEnv(n_machines=10)
  agent = JobScheduleSARSAgent(env, n_episodes=10000, Nzero=10, learning_rate=0.1)
  agent.train()
  agent.evaluate()
  ```
- **GridWorld**: Use `Trainer.py` for training and plotting learning curves.
- **Evaluation**: Set agent epsilon to 0 for greedy evaluation.

## Integration & Dependencies
- **Gymnasium**: All environments inherit from `gym.Env` and use `gym.spaces` for observation/action spaces.
- **Numpy, tqdm**: Used for numerical ops and progress bars.
- **Matplotlib**: Used for plotting in GridWorld.

## Project-Specific Notes
- **State Representation**: States are tuples (agent position, target position) for GridWorld, and machine/job status for JobSchedule.
- **Reward Structure**: Negative rewards for undesirable actions (falling off cliff, invalid job assignment), positive for success.
- **No external config/build system**: All parameters are set in code, not via CLI or config files.
- **No test suite detected**: Validation is manual via `evaluate()` and reward statistics.

## Examples
- See `src/GridWorld/GridWorldSARSAgent.py` and `src/JobSchedule/JobScheduleGreedyAgent.py` for agent patterns.
- See `src/GridWorld/Trainer.py` for training workflow and plotting.

---
_If any section is unclear or missing important project-specific details, please provide feedback to improve these instructions._
