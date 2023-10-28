import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.registration import register
import time
import os

try:
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name':'4x4', 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=0.78, # optimum = .8196
    )
except:
    print('Already Registered')

env = gym.make('FrozenLakeNotSlippery-v0' , render_mode="human")
env.reset()

for step in range(5):

    env.render()
    action = env.action_space.sample()
    observation = env.step(action)
    time.sleep(0.1)
    # print(observation)
    if observation[2]:
        env.reset()

env.close()

qTable = np.zeros([env.observation_space.n, env.action_space.n])




