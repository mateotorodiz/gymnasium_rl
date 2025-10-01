from GridWorldAgent import GridWorldAgent
from GridWorldEnv import GridWorldEnv
from tqdm import tqdm
import numpy as np

from collections import defaultdict

if __name__ == "__main__":
    n_episodes = 100000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes/2)
    final_epsilon = 0.1
    env = GridWorldEnv(size = 15)
    obs,info = env.reset()
    episode_rewarsd = []
    steps_in_episode = []
    agent = GridWorldAgent(
        env = env,
        lr = 0.001,
        initial_epsilon= 1.0,
        epsilon_decay = epsilon_decay,
        final_epsilon=final_epsilon
    )

    for episode in tqdm(range(n_episodes)):
        obs,info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)

            next_obs,reward,terminated,truncated,info = env.step(action)

            agent.update(obs,action,reward,terminated,next_obs)

            done = terminated or truncated
            obs = next_obs
        agent.decay_epsilon()
    

    successes = 0
    steps_list = []

    agent.epsilon = 0.0  # purely greedy

    for _ in range(100):
        obs, info = env.reset()
        done = False
        steps = 0
        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        successes += reward
        steps_list.append(steps)

    print("Success rate:", successes/100)
    print("Average steps (for successes):", np.mean([s for s,r in zip(steps_list, [reward]*100) if r==1]))