import gym
from collections import defaultdict
import numpy as np
import statistics
import IntelliFlap
from IntelliFlap.envs.flap_env import FlapEnv
from IntelliFlap.envs.cnn_flap_env import CnnFlapEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.bit_flipping_env import BitFlippingEnv
from stable_baselines3 import HER, DDPG, DQN, SAC, TD3, PPO, A2C
from stable_baselines3.ppo import CnnPolicy
import random
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def main():
    #env = gym.make('cnnflap-v0')
    #env = CnnFlapEnv

    #env = DummyVecEnv([CnnFlapEnv])
    #check_env(env)
    env = SubprocVecEnv([CnnFlapEnv for _ in range(16)])
    #envs = [make_env('cnnflap-v0', i)() for i in range(8)]
    env.env_method('render')
    #env.render()
    #model = DQN.load("flappy_dqn1", env)
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        activation_fn=nn.LeakyReLU,
        net_arch=[128, dict(pi=[32], vf=[32])]
    )
    #model = DQN('CnnPolicy', env, verbose=1, buffer_size=35000, learning_starts=1000, policy_kwargs=policy_kwargs)
    #model = A2C('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
    model = PPO('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
    #model = DQN.load("flappy_dqn3", env)
    model.learn(total_timesteps=500000)
    model.save("flappy_ppo1")

    env = DummyVecEnv([CnnFlapEnv])
    model = PPO.load("flappy_ppo1", env)

    obs = env.reset()
    env.render()
    rewards = []
    scores = []
    reward = 0
    n_steps = 1000

    wait = input()

    for step in range(n_steps):
        action, _ = model.predict(obs)
        obs_next, reward, done, info = env.step(action)
        print("Reward: ", reward[0])
        if done:
            rewards.append(reward[0])
            scores.append(info[0]['score'])
            reward = 0
            obs = env.reset()
        else:
            obs = obs_next

    print(scores)
    print(rewards)
    print("Average Score: ", statistics.mean(scores))
    print("Highest Score: ", max(scores))
    print("Average Reward: ", statistics.mean(rewards))
    print("Highest Reward: ", max(rewards))


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    #set_random_seed(seed)
    return _init


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.MaxPool2d(3, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.LeakyReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))




if __name__ == "__main__":
    main()