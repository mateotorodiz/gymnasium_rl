from typing import Optional,Tuple
import numpy as np
import gymnasium as gym

import random

class GridWorldEnv(gym.Env):

    def __init__(self, size: Tuple = (4,12)):
        # The size of the square grid (5x5 by default)
        self.size = size

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = (-1,-1)
        self._target_location = (-1,-1)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.MultiDiscrete([self.size[0], self.size[1]]),  # agent (row, col)
            gym.spaces.MultiDiscrete([self.size[0], self.size[1]])   # target (row,col)
        )
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: (1,0),   # Move right (positive x)
            1: (0,1),   # Move up (positive y)
            2: (-1,0),  # Move left (negative x)
            3: (0,-1),  # Move down (negative y)
        }
    
    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            tuple: Observation with agent and target positions
        """
        agent = tuple(int(x) for x in self._agent_location)
        target = tuple(int(x) for x in self._target_location)
        return (agent, target)

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return self.fall_of_cliff()
    
    def randomized_action(self,action):
        if np.random.random() <= 0.8:
            return action
        else:
            return self.action_space.sample()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Place fixed agent
        self._agent_location = (2,random.randint(0,self.size[1] - 3))

        # Place fixed target
        self._target_location = (3,self.size[1]-1)


        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def fall_of_cliff(self):
        """
        Check if the agent has fallen off a cliff
        If so, reward will be very negative
        Agent should be returned to starting position
        """
        if self._agent_location[0] == 3 and self._agent_location[1] > 1 and self._agent_location[1] < self.size[1] - 2:
            return True
        else:
            return False
        
    def found_target(self):
        if self._agent_location[0] == self._target_location[0] and self._agent_location[1] == self._target_location[1]:
            return True
    
    def step(self,action):
        action = self.randomized_action(action)
        direction= self._action_to_direction[action]
        x_clipped = np.clip(self._agent_location[0]+direction[0], 0, self.size[0]-1)
        y_clipped = np.clip(self._agent_location[1]+direction[1],0,self.size[1]-1)
        self._agent_location = (x_clipped,y_clipped)
        
        if self.fall_of_cliff():
            self._agent_location = (3,0)
            terminated = True
            truncated = False
            reward = -100
        
        else:           
            # Check if agent reached target
            terminated = self.found_target()
            truncated = False
            # Assign reward
            reward = 0 if terminated else -1
        observation = self._get_obs()
        info = self._get_info()

        return observation,reward,terminated,truncated,info
        
if __name__ == "__main__":
    test_env = GridWorldEnv()
    test_env.reset()
    terminated = False
    steps = 0
    max_steps = 50000
    while not terminated:
        observation, reward, terminated, truncated, info = test_env.step(random.randint(0, 3))
        steps += 1
    print("Reward:", reward, "Steps:", steps, "Observation:", observation)
