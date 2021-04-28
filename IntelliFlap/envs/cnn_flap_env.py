import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import sys
sys.path.append('../FlappyGym/FlappyBird/')
from flappy import FlappyBird

class CnnFlapEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CnnFlapEnv, self).__init__()
        width = 288
        height = 512
        self.observation_space = spaces.Box(low=0.0, high=255.0, shape=(width, height, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)

        self.flappy = FlappyBird(width, height)
        self.flappy.showWelcomeAnimation()
        self.observation = self.flappy.image_observation()
        
    def step(self, action):
        self.observation, reward, info = self.flappy.game_step(action)
        done = info['dead']

        return self.observation, reward, done, info
    
    def reset(self):
        self.game_obs = self.flappy.showWelcomeAnimation()
        self.observation = self.flappy.image_observation()
        return self.observation
    
    def render(self, mode='human'):
        self.flappy.enable_rendering()
