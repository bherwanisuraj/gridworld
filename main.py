import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.registration import register
import time
import os

EPOCHS = 20000
ALPHA = 0.8
GAMMA = 0.95

epsilon = 1.0
min_epsilon = 0.1
max_epsilon = 1.0
decay_val = 0.001

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

env = gym.make('FrozenLakeNotSlippery-v0', render_mode="human")
# env.reset()
#
# for step in range(5):
#
#     env.render()
#     action = env.action_space.sample()
#     observation = env.step(action)
#     time.sleep(0.1)
#     # print(observation)
#     if observation[2]:
#         env.reset()
#
# env.close()

qTable = np.zeros([env.observation_space.n, env.action_space.n])

def egas(epsilon, qTable, discrete_state):
    rn = np.random.random()
    # exploitation
    if rn > epsilon:
        state = qTable[discrete_state, :]
        action = np.argmax(state)
    else:
        action = env.action_space.sample()

    return action

def computeQValue(currentQvalue, reward, nextOptimalQValue):
    return currentQvalue + ALPHA*(reward+GAMMA*nextOptimalQValue-currentQvalue)

def decay(epsilon, EPOCH):
    return min_epsilon+(max_epsilon-min_epsilon)*np.exp(-(decay_val)*EPOCH)

rewards = []
interval = 1000

for episode in range(EPOCHS):
    state = env.reset()
    terminated = False
    totalRewards = 0

    while not terminated:
        #ACTION
        action = egas(epsilon, qTable, state)


        #Get values such as state, reward, done, info
        obs, reward, terminated, truncated, info = env.step(action)
        print()

        #Get Current Q Value

        currentQValue = qTable[obs, action]

        #Get Optimal q value
        nextOptimalQValue = np.max(qTable[obs, :])

        #Compute next Q Value
        nextQValue = computeQValue(currentQValue, reward, nextOptimalQValue)

        #Update the table
        qTable[obs, action] = nextQValue

        #Track Rewards
        totalRewards = totalRewards+reward

        #Update State
        state = obs

    episode+=1
    epsilon = decay(epsilon, episode)
    rewards.append(totalRewards)
    # if episode % interval == 0:
    #     print(np.sum(rewards))
    print(np.sum(rewards))

env.close()
