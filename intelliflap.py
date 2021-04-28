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
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.dqn import CnnPolicy
import random


def main():
    #env = gym.make('cnnflap-v0')
    #check_env(env)
    #env = CnnFlapEnv
    env = DummyVecEnv([CnnFlapEnv])
    #env = SubprocVecEnv([CnnFlapEnv for _ in range(4)])
    #env = SubprocVecEnv([make_env('cnnflap-v0', i) for i in range(2)])
    env.render()

    #model = DQN.load("flappy_dqn5", env)
    model = DQN(CnnPolicy, env, verbose=1, buffer_size=7500, optimize_memory_usage=True, train_freq=30, learning_starts=250)
    model.learn(total_timesteps=25000)
    model.save("flappy_dqn6")


    obs = env.reset()
    #env.render()
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


if __name__ == "__main__":
    main()