import random
from _collections import deque
import gym
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

env = gym.make('CartPole-v1')
# env = gym.make('CartPole-v1', render_mode = 'human')
# env.reset()
# env.close()
NO_OBSERVATIONS = env.observation_space.shape[0]

env.reset()
# for _ in range(100):
#     env.render()
#     observation, check, done, info, done  = env.step(env.action_space.sample())
#     if done:
#         env.reset()
# env.close()

EPOCHS = 1000
EPSILON = 1.0
EPSILON_REDUCE = 0.995
LR = 0.001
GAMMA = 0.95

num_of_obs = env.observation_space.shape[0]
num_of_act = env.action_space.n
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation = 'relu', input_shape=(1, num_of_obs)))
model.add(tf.keras.layers.Dense(32, activation = 'relu'))
model.add(tf.keras.layers.Dense(num_of_act, activation = 'linear'))
model.compile(loss = 'mse', optimizer = (Adam(learning_rate=LR)))

# print(model.summary())
target_model = clone_model(model)



def egas(model, epsilon, obs):
    if np.random.random() > epsilon:
        # print(f"Inside {obs.shape}")
        # print(obs.shape)
        prediction = model.predict(obs, verbose=0)
        # print(f"prediction {prediction}")
        # print("-------------------------------------------------------------------------------------------------------------------")

        action = np.argmax(prediction)
    else:
        action = np.random.randint(0, env.action_space.n)

    return action

REPLAY_BUFFER = deque(maxlen=20000)
UPDATE_TARGET = 10
BATCH_SIZE = 32

def replay(buffer, batch_size, model, target_model):
    if len(buffer)<batch_size:
        return
    # print(f"Lenght {len(buffer)}")
    samples = random.sample(buffer, batch_size)
    target_batch = []
    zipped_list = list(zip(*samples))
    states, actions, rewards, new_states, dones = zipped_list
    # print(f"states {states}, actions {actions}, rewards {rewards}, new_states {new_states}, dones {dones}")
    targets = target_model.predict(np.array(states), verbose=0)
    # print(f"targets {targets}")
    # print("-------------------------------------------------------------------------------------------------------------------")

    q_values = model.predict(np.array(new_states), verbose=0)
    # print(f"q_values {q_values}")
    # print("-------------------------------------------------------------------------------------------------------------------")

    for i in range(batch_size):
        q_value = max(q_values[i][0])
        # print(f"q_value single {q_value}")
        # print("-------------------------------------------------------------------------------------------------------------------")
        target = targets[i].copy()
        # print(f"target single {target}")
        # print("-------------------------------------------------------------------------------------------------------------------")

        if dones[i]:
            target[0][actions[i]] = rewards[i]
            # print(target[0][actions[i]])
            # print("-------------------------------------------------------------------------------------------------------------------")


        else:
            target[0][actions[i]] = rewards[i]+q_value*GAMMA

        target_batch.append(target)
    model.fit(np.array(states), np.array(target_batch), epochs=1, verbose=0)


def update_model_handler(epoch, update_target_model, model, target_model):
    if epoch >0 and epoch % update_target_model == 0:
        target_model.set_weights(model.get_weights())



best_so_far = 0
for epoch in range(EPOCHS):
    obs = env.reset()
    # print(obs)
    obs = obs[0].reshape([1,4])
    # print(obs)
    done = False
    points = 0
    while not done:
        action = egas(model, EPSILON, obs)
        # print(env.step(action))
        next_obs, reward, done, info, check = env.step(action)
        next_obs = next_obs.reshape([1,4])
        # print(f"next_obs {next_obs}, reward {reward}, done {done}, info {info}, check {check}")
        # print("-------------------------------------------------------------------------------------------------------------------")


        REPLAY_BUFFER.append((obs, action, reward, next_obs, done))
        # print(f"REPLAY BUFFER: obs {obs}, action {action}, reward {reward}, next_obs {next_obs}, done {done}")
        # print(REPLAY_BUFFER)
        # print("-------------------------------------------------------------------------------------------------------------------")
        obs = next_obs
        points+=1
        replay(REPLAY_BUFFER, 32, model, target_model)

    EPSILON = EPSILON * EPSILON_REDUCE
    update_model_handler(epoch, UPDATE_TARGET, model, target_model)
    if points > best_so_far:
        best_so_far = points

    if epoch%25 == 0:
        print(f" epoch - {epoch}  |  points - {points}  |  epsilon - {EPSILON}  |  bfs - {best_so_far}")
