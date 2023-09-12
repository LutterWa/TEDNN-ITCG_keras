import os
import numpy as np
from scipy.io import loadmat, savemat
from missile import Missile
from multiprocessing import Pool
from functools import partial

pi = 3.141592653589793
RAD = 180 / pi  # 弧度转角度
g = 9.81
cpu_counts = 10  # 并行采样使用的CPU数
samples_num = 100  # 采样弹道数


def generate(miss):
    h = 0.001  # 仿真步长
    while True:
        miss.modify()
        done = False
        while done is False:
            done = miss.step(h)
        if miss.R < 200 * h and miss.t > 1:
            break
    missile = np.array(miss.record["state"])
    x = np.dot(missile[:, 1:], np.diag([0.05, 10, 1e-3, 1e-3]))
    y = missile[-1, 0] - missile[:, 0]
    return x, y


def process(k, count):
    miss = Missile(k=k)  # 创建导弹对象
    x, y = generate(miss)  # 生成随机样本
    for itr in range(int(samples_num // cpu_counts)):
        print("==========迭代次数 {}==========".format(itr + 1))
        batch_x, batch_y = generate(miss)  # 生成随机样本
        x = np.concatenate([x, batch_x])
        y = np.concatenate([y, batch_y])
    flight_data = {"x": x, "y": y}
    if not os.path.exists('mats_{:.1f}_{:.1f}_{:.1f}'.format(*k)):
        os.makedirs('mats_{:.1f}_{:.1f}_{:.1f}'.format(*k))
    savemat('./mats_{:.1f}_{:.1f}_{:.1f}/anti_flight_data_{}.mat'.format(*k, count), flight_data)


def collect_data(k=(1., 1., 1.)):
    try:
        data_raw = loadmat('./anti_flight_data_{:.1f}_{:.1f}_{:.1f}.mat'.format(*k))
        print("all flight data load done!")
        x = data_raw["x"]
        y = data_raw["y"]
    except FileNotFoundError:
        pool = Pool(cpu_counts)
        pool.map(partial(process, k), range(cpu_counts))  # 创建多个线程

        data_raw = loadmat('./mats_{:.1f}_{:.1f}_{:.1f}/anti_flight_data_0.mat'.format(*k))
        print("simulate data collect done!")

        x = data_raw["x"]
        y = data_raw["y"].T

        for _ in range(1, cpu_counts):
            data_raw = loadmat('./mats_{:.1f}_{:.1f}_{:.1f}/anti_flight_data_{}.mat'.format(*k, _))
            x = np.concatenate([x, data_raw["x"]])
            y = np.concatenate([y, data_raw["y"].T])
        flight_data = {"x": x, "y": y}
        savemat('mats/itcg_train_data_{:.1f}_{:.1f}_{:.1f}.mat'.format(*k), flight_data)
    return x, y


if __name__ == "__main__":
    K = [0.2, 0.5, 1.0, 2.0, 4.0]
    for i in K:
        k = (i, i, i)
        collect_data(k=k)
    k = (3.0, 1.5, 1.0)
    collect_data(k=k)
