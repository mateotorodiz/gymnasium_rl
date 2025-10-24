import numpy as np
import gymnasium as gym
import d3rlpy


def evaluate(algo, env, num_episodes=20, seed=42):
    returns = []
    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        ep_ret = 0.0
        while not done:
            # d3rlpy expects batch input; take greedy action
            action = int(algo.predict(np.asarray([obs], dtype=np.float32))[0])
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
        returns.append(ep_ret)

    mean = float(np.mean(returns)) if returns else 0.0
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    return mean, std


if __name__ == "__main__":
    env_id = "CartPole-v1"
    model_path = "cql_cartpole.pt"  # adjust to your saved model path
    n_episodes = 20

    env = gym.make(env_id)
    algo = d3rlpy.algos.DiscreteCQLConfig().create(device="cpu")
    # Build network shapes from the environment, then load weights
    algo.build_with_env(env)
    algo.load_model(model_path)

    mean, std = evaluate(algo, env, n_episodes)
    print(f"Episodes: {n_episodes} | Mean reward: {mean:.2f} | Std: {std:.2f}")

    env.close()