import gymnasium as gym
from typing import Optional
import random
import numpy as np

class SnakeGame(gym.env):
    def __init__(self):
        ...
        # create observation space
        self.obervation_space = gym.spaces.Box(...)
        self.action_space = gym.spaces.Discrete(4)
        self.snake_head = [1,1]
        self._action_to_direction = {
            0: [-10,0],
            1: [10,0],
            2: [0,10],
            3: [0,-10],
        }

    def _get_obs(self):
        ...
        # create the gym.spaces.Box from the snake position
    
    def _get_info(self):
        ...
        #create and return a dict
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Initial Snake and Apple position
        self.snake_position = [[250,250],[240,250],[230,250]]
        self.apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250,250]

        obs = self._get_obs()
        info = self._get_info()

        return obs,info
    
    def step(self,action):
        terminated = False
        truncated = False # change for time condition here
        # Decode the action
        self.snake_head += self._action_to_direction(action)

        # Check for the consequences of the action
        if self.snake_head == self.apple_position:
            self.collision_with_apple()
            self.snake_position.insert(0,list(self.snake_head))
            reward = 10

        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()
            reward = -1

            if self.collision_with_boundaries() or self.collision_with_self():
                terminated = True
                truncated = False
                reward = -100

        observation = self._get_obs()
        info = self._get_info()
        return observation,reward,terminated,truncated,info

        

    def collision_with_apple(self):
        self.apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
        self.score += 1

    def collision_with_boundaries(self):
        if self.snake_head[0]>=500 or self.snake_head[0]<0 or self.snake_head[1]>=500 or self.snake_head[1]<0 :
            return 1
        else:
            return 0

    def collision_with_self(self):
        self.snake_head = self.snake_position[0]
        if self.snake_head in self.snake_position[1:]:
            return 1
        else:
            return 0
    