import gymnasium as gym
import random

class JobScheduleEnv(gym.Env):
    def __init__(self, n_machines):
        # create a list of machines
        self.machines = [(0,0)] * n_machines
        self.jobs_remaining = -1
        self.max_time = 10
        self.max_jobs = 10
        self.time_step = 0

        self.idle_reward = 0
        self.finish_reward = 10
        self.invalid_assignment_reward = -1


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
        for i in range(len(self.machines)):
            self.machines[i] = (0,0)

        self.jobs_remaining = random.randint(1,self.max_jobs)
        self.time_step = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info
        
    def step(self, action):
        # Every step is penalized with -1, will only be different if something happens
        reward = -1
        terminated = False
        self.time_step +=1
        if action == 0:
            for i in range(len(self.machines)):
                if not self.is_idling(i):
                    state, time = self.machines[i]
                    self.machines[i] = (state, max(0, time - 1))
                    if self.check_finished_job(i): # if the machine then finishes
                        self.machines[i] = (0, 0)
                        reward += self.finish_reward
                else: # machine has idled
                    reward += self.idle_reward

        else:
            # the action number matches the machine, at which we want to execute the job
            # ensure that the machine is idling (or not)
            if self.is_idling(action):
                self.machines[action] = (1,self.generate_job_time())
                self.jobs_remaining -= 1
                # assign a job to the machine
            else:
                reward -= self.invalid_assignment_reward
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
        return {"time_step": self.time_step}
    
    def check_finished_job(self,machine_index):
        return self.machines[machine_index][1] == 0
        
    def is_idling(self,machine_index):
        return self.machines[machine_index][0] == 0

    def generate_job_time(self):
        return random.randint(1,self.max_time)

    
if __name__ == "__main__":
    n_machines = 2
    env = JobScheduleEnv(n_machines = n_machines)
    Gt = 0
    done = False
    for i in range(200):
        while not done:
            obs,info = env.reset()
            obs,reward,terminated,truncated,info = env.step(random.randint(0,n_machines))
            done = terminated or truncated
            Gt += reward
    
    print(reward)
    #mini test bench