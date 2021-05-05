import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import sys
sys.path.append('../FlappyGym/FlappyBird/')
from flappy import FlappyBird
from skimage import color
from skimage.transform import rescale, resize

class CnnFlapEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CnnFlapEnv, self).__init__()
        width = 288
        height = 512
        self.rgb = True
        self.high_score = 0
        self.div = 4
        self.observation_space = spaces.Box(low=0.0, high=255.0, shape=(width // self.div, height // self.div, 3), dtype=np.uint8)
        if not self.rgb:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1, width, height), dtype=np.uint8)
            
        self.action_space = spaces.Discrete(2)
        #self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.uint8)

        self.flappy = FlappyBird(width, height)
        self.flappy.showWelcomeAnimation()
        self.observation = self.flappy.image_observation()
        self.preprocess_observation()
        
    def step(self, action):
        self.observation, reward, info = self.flappy.game_step(action)
        self.preprocess_observation()
        done = info['dead']
        if info['score'] > self.high_score:
            self.high_score = info['score']
            print("New High Score: ", self.high_score)
 
        return self.observation, reward, done, info
    
    def reset(self):
        self.game_obs = self.flappy.showWelcomeAnimation()
        self.observation = self.flappy.image_observation()
        self.preprocess_observation()
        return self.observation
    
    def render(self, mode='human'):
        self.flappy.enable_rendering()

    def preprocess_observation(self):
        if not self.rgb:
            self.observation = color.rgb2gray(np.asarray(self.observation))
        self.observation = resize(self.observation, (self.observation.shape[0] // self.div, self.observation.shape[1] // self.div), anti_aliasing=True)
        if not self.rgb:
            self.observation = np.expand_dims(self.observation, 0)
