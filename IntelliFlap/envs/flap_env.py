import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import sys
sys.path.append('../FlappyGym/FlappyBird/')
from flappy import FlappyBird
import math

class FlapEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FlapEnv, self).__init__()
        self.width = 288
        self.height = 512
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.flappy = FlappyBird(self.width, self.height)
        self.game_obs = self.flappy.showWelcomeAnimation()
        self.observation = self._format_observation(self.game_obs)
        
    def step(self, action):
        self.game_obs, reward, info = self.flappy.game_step(action)
        self.observation = self._format_observation(self.game_obs)
        done = self.game_obs['dead']
        #print(self.observation)
        return self.observation, reward, done, info
    
    def _format_observation(self, output):
        #print(output)
        #pipe_x = output['upperPipes'][0]['x'] / self.width if output['upperPipes'][0]['x'] / self.width <= 1 else 1
        """
        if output['upperPipes'][0]['x'] != 0:
            y_over_x_0 = (output['lowerPipes'][0]['y'] + 46) / output['upperPipes'][0]['x']
            if y_over_x_0 > 1:
                y_over_x_0 = 1
        else:
            y_over_x_0 = 1
        if output['upperPipes'][1]['x'] != 0:
            y_over_x_1 = (output['lowerPipes'][1]['y'] + 46) / output['upperPipes'][1]['x']
            if y_over_x_1 > 1:
                y_over_x_1 = 1
        else:
            y_over_x_1 = 1
        """
        dist_to_goal_0 = math.sqrt(((output['upperPipes'][0]['x'] - output['playerx']) ** 2) + (((output['lowerPipes'][0]['y'] + 46) -  output['playery']) ** 2))
        dist_to_goal_1 = math.sqrt(((output['upperPipes'][1]['x'] - output['playerx']) ** 2) + (((output['lowerPipes'][1]['y'] + 46) -  output['playery']) ** 2))
        observation = [
            output['playery'] / (self.height * 2),
            output['upperPipes'][0]['x'] / (self.width * 2.25), 
            (output['lowerPipes'][0]['y'] + 46) / (self.height * 2),
            dist_to_goal_0 / (self.width * 2.25),
            output['upperPipes'][1]['x'] / (self.width * 2.25), 
            (output['lowerPipes'][1]['y'] + 46) / ( self.height * 2),
            dist_to_goal_1 / (self.width * 2.25)
        ]   
        #print(observation)
        return np.asarray(observation)
    
    def reset(self):
        self.game_obs = self.flappy.showWelcomeAnimation()
        self.observation = self._format_observation(self.game_obs)
        return self.observation
    
    def render(self, mode='human'):
        self.flappy.enable_rendering()