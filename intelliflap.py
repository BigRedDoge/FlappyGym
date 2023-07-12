import random
import statistics
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
#from stable_baselines3.common.bit_flipping_env import BitFlippingEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.ppo import CnnPolicy

import IntelliFlap
from custom_arch import CustomActorCriticPolicy, CustomCNN
from IntelliFlap.envs.cnn_flap_env import CnnFlapEnv
from IntelliFlap.envs.flap_env import FlapEnv


def main():
    #env = gym.make('cnnflap-v0')
    #env = CnnFlapEnv

    #env = DummyVecEnv([CnnFlapEnv])
    #check_env(env)
    env = SubprocVecEnv([CnnFlapEnv for _ in range(1)])
    #envs = [make_env('cnnflap-v0', i)() for i in range(8)]
    env.env_method('render')
    #env.render()
    #model = DQN.load("flappy_dqn1", env)
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256)
    )
    #eval_callback = EvalCallback(env[0], best_model_save_path='./saved_models/',
    #                         eval_freq=100000, deterministic=True, render=False)
    #model = DQN('CnnPolicy', env, verbose=1, buffer_size=35000, learning_starts=1000, policy_kwargs=policy_kwargs)
    #model = PPO('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
    #model = DQN.load("flappy_dqn3", env)
    
   # model = A2C(CustomActorCriticPolicy, env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./logs/")
    #model = PPO(CustomActorCriticPolicy, env, verbose=1, n_steps=512, policy_kwargs=policy_kwargs, tensorboard_log="./logs/", batch_size=32)
    #model.load("saved_models/flappy_a2c_custom2mil")
    #model.learn(total_timesteps=1000000)
    #model.save("saved_models/flappy_a2c_custom2_1mil")

    #env = DummyVecEnv([CnnFlapEnv])
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/progress_tensorboard/")
    #model = A2C.load("saved_models/flappy_a2c_custom2_1mil", env)

    obs = env.reset()
    env.render('human')
    rewards = []
    scores = []
    reward = 0
    n_steps = 1000

    wait = input()

    for step in range(n_steps):
        action, _ = model.predict(obs)
        obs_next, reward, done, info = env.step(action)
        #print("Reward: ", reward[0])
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


if __name__ == "__main__":
    main()
