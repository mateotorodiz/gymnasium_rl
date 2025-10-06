import gymnasium as gym
import random
random.seed(42)

class JobScheduleEnv(gym.Env):
    def __init__(self, n_machines):
        # create a list of machines, each represented as (state, time_remaining)
        self.machines = [(0,0)] * n_machines
        self.jobs_remaining = -1
        self.max_time = 10
        self.max_jobs = 20
        self.time_step = 0

        self.idle_reward = 0
        self.finish_reward = 10
        self.invalid_assignment_reward = -1

        self.idle_steps = 0
        self.finished_jobs = 0
        self.incorrect_asignments = 0

        # Define observation space for each machine and jobs remaining
        obs_spaces = []
        for _ in range(n_machines):
            obs_spaces.append(
                gym.spaces.Tuple((
                    gym.spaces.Discrete(2),
                    gym.spaces.Discrete(self.max_time)
                ))
            )
        obs_spaces.append(gym.spaces.Discrete(self.max_jobs))

        self.observation_space = gym.spaces.Tuple(tuple(obs_spaces))
        self.action_space = gym.spaces.Discrete(n_machines)

    def reset(self, *, seed = None, options = None):
        # Reset all machines to idle state
        for i in range(len(self.machines)):
            self.machines[i] = (0,0)

        self.jobs_remaining = self.max_jobs
        self.time_step = 0
        self.finished_jobs = 0
        self.idle_steps  = 0
        self.incorrect_asignments = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info
        
    def step(self, action):
        # Every step is penalized with -1, will only be different if something happens in job scheduling
        reward = -1
        terminated = False
        self.time_step +=1
        if action == 0:
            for i in range(len(self.machines)):
                if not self.is_idling(i):
                    state, time = self.machines[i]
                    self.machines[i] = (state, max(0, time - 1))
                    if self.check_finished_job(i): # if the machine then finishes its job
                        self.machines[i] = (0, 0)
                        reward += self.finish_reward
                        self.finished_jobs += 1
                else: # machine has idled
                    reward += self.idle_reward
                    self.idle_steps += 1

        else:
            # the action number matches the machine, at which we want to execute the job
            # ensure that the machine is idling (or not)
            if self.is_idling(action-1):
                self.machines[action-1] = (1,self.generate_job_time())
                self.jobs_remaining -= 1
                # assign a job to the machine
            else:
                reward -= self.invalid_assignment_reward
                self.incorrect_asignments += 1
                # machine is already working, cannot assign a job to it
        if self.jobs_remaining == 0:
            reward += 100
            terminated = True
        obs = self._get_obs()
        info = self._get_info()
        truncated = False

        return obs,reward,terminated,truncated,info
    
    def _get_obs(self):
        return (tuple(self.machines),self.jobs_remaining)

    def _get_info(self):
        return {"time_step": self.time_step,
                "idle_steps": self.idle_steps,
                "finished_jobs": self.finished_jobs,
                "incorrect_assignments": self.incorrect_asignments}
    
    def check_finished_job(self,machine_index):
        return self.machines[machine_index][1] == 0
        
    def is_idling(self,machine_index):
        return self.machines[machine_index][0] == 0

    def generate_job_time(self):
        return self.max_time #random.randint(1,self.max_time)

    
if __name__ == "__main__":
    n_machines = 5
    env = JobScheduleEnv(n_machines = n_machines)
    print("Type of machine state:", type(env.machines[0][0]))
    Gt = 0
    rewards = list()
    infos = list()
    for i in range(200):
        done = False
        obs,info = env.reset()
        Gt = 0
        while not done:
            action = random.randint(0,n_machines)
            obs,reward,terminated,truncated,info = env.step(action)
            done = terminated or truncated
            Gt += reward
        rewards.append(Gt)
        infos.append(info)
    
    print("Total rewards per episode (job scheduling):")
    print(rewards)
    print("Episode info (job scheduling):")
    print(infos)
    #mini test bench