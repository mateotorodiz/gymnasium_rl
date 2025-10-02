# GridWorld SARSA Experiment

This project demonstrates reinforcement learning in a flexible grid-world environment using the SARSA algorithm.

## Environment

- **GridWorldEnv** simulates a grid where an agent must reach a target while avoiding a cliff region.
- The grid size is configurable.
- The agent receives a large negative reward for falling off the cliff and a zero reward for reaching the target; otherwise, it receives a small negative reward for each step.

## Agent

- **GridWorldSARSAgent** implements the SARSA algorithm with epsilon-greedy exploration.
- The agent learns an action-value function to navigate the grid efficiently.
- Training and evaluation routines are provided.

## Usage

1. Configure the environment and agent parameters in `GridWorldSARSAgent.py`.
2. Run the script to train and evaluate the agent.

## Purpose

The experiment aims to illustrate how SARSA learns to avoid the cliff and reach the target efficiently, demonstrating temporal-difference learning in a classic RL benchmark.

---