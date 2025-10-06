
import gymnasium as gym
from JobScheduleEnv import JobScheduleEnv

class JobScheduleGreedyAgent:
    """
    Greedy agent for the Job Scheduling environment.
    Uses the 'earliest available machine' heuristic: assigns each job to the machine that will be available the soonest.
    """
    def __init__(self, env: gym.Env):
        self.env = env

    def get_action(self, obs):
        """
        Selects the machine with the lowest remaining processing time (earliest available).
        Args:
            obs: Current observation (tuple of machines, jobs_remaining)
        Returns:
            int: Selected action (machine index)
        """
        machines, jobs_remaining = obs
        # Find the machine with the lowest time remaining
        min_time = float('inf')
        best_machine = 0
        for idx, (state, time_remaining) in enumerate(machines):
            if state == 0:
                # Idle machine, available now
                return idx + 1  # action 0 is 'wait', actions 1..N are machine assignments
            if time_remaining < min_time:
                min_time = time_remaining
                best_machine = idx
        # If all busy, pick the idle option
        return 0

    def evaluate(self, n_episodes=10):
        print("Evaluating GreedyAgent on Job Scheduling environment")
        reward_list = []
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            Gt = 0
            while not done:
                action = self.get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                Gt += reward
            reward_list.append(Gt)
        print("Returned reward per episode:", reward_list)


if __name__ == "__main__":
    n_machines = 10
    env = JobScheduleEnv(n_machines=n_machines)
    agent = JobScheduleGreedyAgent(env)
    agent.evaluate(n_episodes=10)
