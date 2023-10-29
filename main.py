import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.registration import register
import time
import os

EPOCHS = 20000
ALPHA = 0.8
GAMMA = 0.9

epsilon = 1.0
min_epsilon = 0.01
max_epsilon = 1.0
decay_val = 0.0045

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

env = gym.make('FrozenLakeNotSlippery-v0')#render_mode="human"rgb_array
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
        # print(discrete_state)
        # print('Exploit')
        state_row = qTable[discrete_state, :]
        action = np.argmax(state_row)
        # print(qTable)
    else:
        # print('random')
        action = env.action_space.sample()

    return action

def computeQValue(currentQvalue, reward, nextOptimalQValue):
    # if reward>0:
    #
    #     print(f"computeQValue : {currentQvalue + ALPHA*(reward+GAMMA*nextOptimalQValue-currentQvalue)}")

    return currentQvalue + ALPHA*(reward+GAMMA*nextOptimalQValue-currentQvalue)

def decay(epsilon, EPOCH):
    return min_epsilon+(max_epsilon-min_epsilon)*np.exp(-decay_val*EPOCH)

rewards = []
interval = 1000

for episode in range(EPOCHS):
    # print(episode)
    state = env.reset()
    # print(f"first state {state}")
    terminated = False
    totalRewards = 0

    while not terminated:
        #ACTION
        if type(state) is not int:
            state = state[0]
        action = egas(epsilon, qTable, state)

        #Get values such as state, reward, done, info
        new_state, reward, terminated, truncated, info = env.step(action)
        # if reward>0:
        #     print(f"new_state: {new_state}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")

        #Get Current Q Value

        currentQValue = qTable[state, action]
        # print(currentQValue)

        #Get Optimal q value
        nextOptimalQValue = np.max(qTable[new_state, :])
        # if nextOptimalQValue > 0:
        #     print(f"nextOptimalQValue : {nextOptimalQValue}")

        #Compute next Q Value
        nextQValue = computeQValue(currentQValue, reward, nextOptimalQValue)
        # if reward>0:
        #     print(reward)
        # if nextQValue > 0:
        #
        #     print(nextQValue)

        #Update the table
        qTable[state, action] = nextQValue
        # if nextQValue > 0:
        #     print(qTable)

        #Track Rewards
        totalRewards = totalRewards+reward
        # print(totalRewards)

        #Update State

        state = new_state
        # print(state)
        # print("-----------------------------------------------------------")
    # print(f"this obs {state}")
    episode+=1
    epsilon = decay(epsilon, episode)
    # print(epsilon)
    rewards.append(totalRewards)
    if episode % interval == 0:
        print(np.sum(rewards))
        # pass
    # print(f"reward {np.sum(rewards)}")

env.close()
