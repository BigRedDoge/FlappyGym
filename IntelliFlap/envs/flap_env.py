import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import sys
sys.path.append('../FlappyGym/FlappyBird/')
from flappy import FlappyBird

class FlapEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        #super(FlapEnv, self).__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.flappy = FlappyBird()
        self.game_obs = self.flappy.init_game()
        self.observation = self._format_observation(self.game_obs)
        
    def step(self, action):
        self.game_obs, reward, info = self.flappy.game_step(action, self.game_obs)
        self.observation = self._format_observation(self.game_obs)
        done = self.game_obs['dead']

        return self.observation, reward, done, info
    
    def _format_observation(self, output):
        width = self.flappy.SCREENWIDTH
        height = self.flappy.SCREENHEIGHT
        observation = [
            output['playery'] / height,
            output['playerVelY'] / 10, 
            output['playerRot'] / 90, 
            output['upperPipes'][0]['x'] / width, 
            output['upperPipes'][0]['y'] / height,
            output['lowerPipes'][0]['x'] / width, 
            output['lowerPipes'][0]['y'] / height
        ]   

        return np.asarray(observation)
    
    def reset(self):
        movementInfo = self.flappy.showWelcomeAnimation()
        self.game_obs = self.flappy.mainGame(movementInfo)
        self.observation = self._format_observation(self.game_obs)
        return self.observation
    
    def render(self):
        self.flappy.enable_rendering()