import random
import time
import math
import os
import numpy as np
import tensorflow.python.keras as keras
import matplotlib.pyplot as plt
from scipy.io import savemat
from missile import Missile, pi
from ddpg import DDPG

RAD = 180 / pi  # 弧度转角度

TRAIN_EPISODES = 100  # total number of episodes for training
TEST_EPISODES = 100  # total number of episodes for testing
GAMMA = 0.99
REWARD_SAVE_CASE = 0
model_tren = keras.models.load_model("./model/model_tren.h5")


class AGENT(Missile):
    def __init__(self):
        super().__init__(k=(3.0, 1.5, 1.0))
        x = np.dot(self.state[1:], np.diag([0.05, 10, 1e-3, 1e-3]))
        self.tgo = float(model_tren.predict(x[np.newaxis, :], verbose=0, use_multiprocessing=True))

    def get_tgo(self):
        x = np.dot(self.state[1:], np.diag([0.05, 10, 1e-3, 1e-3]))
        self.tgo = float(model_tren.predict(x[np.newaxis, :], verbose=0, use_multiprocessing=True))
        return self.tgo

    def get_state(self, td):
        tgo = self.get_tgo()
        s = [self.v * max(td - self.t - tgo, 0) / tgo]
        return np.array(s)

    def get_reward(self, td, ab):
        tgo = self.tgo
        err = (td - self.t - tgo) / tgo

        energy = 9.81 * self.y + 0.5 * self.v ** 2  # 机械能

        vy = self.v * math.sin(self.theta)  # y向速度
        zem = self.y / tgo + vy

        b1, b2, b3 = 0.98, 0.01, 0.01
        s1, s2, s3 = 1., 1000., 100.
        r = b1 * math.exp(-(err / s1) ** 2) + \
            b2 * math.tanh((energy / s2) ** 2) + \
            b3 * math.exp(-(zem / s3) ** 2)
        return np.array(r)


if __name__ == '__main__':
    env = AGENT()

    # set the init parameter
    s_dim = 1
    a_dim = 1
    a_bound = 9.81 * 2.0
    MEMORY_CAPACITY = 2000  # size of replay buffer

    t0 = time.time()
    model_num = 0

    dict_reward = {'episode_reward': [], 'target_time': [], 'actual_time': []}

    agent = DDPG(a_dim, s_dim, a_bound)
    for episode in range(int(TRAIN_EPISODES)):
        desired_tgo = []  # 期望的tgo
        actual_tgo = []  # 实际的tgo
        episode_reward = 0

        env.modify()  # 0.时间,1.速度,2.弹道倾角,3.导弹x,4.导弹y,5.质量
        td = env.get_tgo() * random.uniform(1.2, 1.3)
        state = env.get_state(td)
        action = agent.choose_action(state)  # get new action with old state
        # 弹道模型
        done = False
        while done is False:
            # collect state, action and reward
            action = agent.choose_action(state)  # get new action with old state
            done = env.step(h=0.1, ab=float(action))
            state_ = env.get_state(td)  # get new state with new action
            reward = env.get_reward(td, float(action))  # get new reward with new action
            agent.store_transition(state, action, reward, state_)  # train with old state
            state = state_  # update state
            episode_reward += reward

            desired_tgo.append(td - env.t)
            actual_tgo.append(env.tgo)

            # update ppo
            # 第一次数据满了，就可以开始学习
            if agent.pointer > MEMORY_CAPACITY:
                agent.learn()

        episode_reward -= env.R + (td - env.t) ** 2

        # print the result
        episode_reward = episode_reward / env.t  # calculate the average episode reward
        print('Training | Episode: {}/{} | Average Episode Reward: {:.2f} | Running Time: {:.2f} | '
              'Target Time: {:.2f} | Actual Time: {:.2f} | Error Time: {:.2f}'
              .format(episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0, td, env.t, td - env.t))

        # save the episode data
        dict_reward['episode_reward'].append(episode_reward)
        dict_reward['target_time'].append(td)
        dict_reward['actual_time'].append(env.t)

    agent.save_ckpt()
    savemat('./ddpg_reward.mat', dict_reward)
