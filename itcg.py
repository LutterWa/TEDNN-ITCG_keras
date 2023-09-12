import os
import sys
import keras
import random
import numpy as np
from scipy.io import loadmat, savemat
from missile import Missile, pi

env = Missile(k=(3.0, 1.5, 1.0))
model_tren = keras.models.load_model("./model/model_tren.h5")

if not os.path.exists('itcg_figs'):
    os.makedirs('itcg_figs')

task = ["monte", "single", "compare"][0]


def progress_bar(i):
    i *= 100  # 转换为百分比
    print("\r", end="")
    print("Data in progress: {:.2f}% ".format(i),
          "[" + "=" * (int(min(i, 100) / 2)) + ">" + "." * (50 - int(min(i, 100) / 2)) + "]", end="")
    sys.stdout.flush()


if task == "monte":
    # result = loadmat('itcg_figs/sim_ddpg_monte.mat')
    result = {}
    N = 500
    for i in range(1, N + 1):
        while True:
            env.modify()
            x = np.dot(env.state[1:], np.diag([0.05, 10, 1e-3, 1e-3]))
            tgo = float(model_tren.predict(x[np.newaxis, :], verbose=0, use_multiprocessing=True))
            td = tgo * 1.1
            done = False
            while done is False:
                x = np.dot(env.state[1:], np.diag([0.05, 10, 1e-3, 1e-3]))
                tgo = float(model_tren.predict(x[np.newaxis, :], verbose=0, use_multiprocessing=True))
                te = td - tgo - env.t
                done = env.step(h=0.1, ab=0.03 * env.v * te)
                progress_bar(env.t / td)
                if env.t / td > 1.1:
                    done = True
            if env.R < 50 and abs(td - env.t) < 1:
                break
        print("{}/{}脱靶量={:.4f} 飞行时间误差={:.4f}".format(i, N, env.R, td - env.t))
        result["sim_{}".format(i)] = env.record
        result["time_{}".format(i)] = [td, env.t, td - env.t]
        savemat('itcg_figs/sim_ddpg_monte.mat', result)
elif task == "single":
    tds = [70, 80, 90, 100]
    for td in tds:
        env.modify([0., 600, 0, -20000, 10000])
        result = {"td": [], "tgo": []}
        done = False
        while done is False:
            x = np.dot(env.state[1:], np.diag([0.05, 10, 1e-3, 1e-3]))
            tgo = float(model_tren.predict(x[np.newaxis, :], verbose=0, use_multiprocessing=True))
            te = td - tgo - env.t
            done = env.step(h=0.01, ab=0.02 * env.v * te)
            result["td"].append(td - env.t)
            result["tgo"].append(tgo)
            progress_bar(env.t / td)
        print("脱靶量={:.4f} 飞行时间={:.4f}".format(env.R, env.t))
        savemat('itcg_figs/sim_ddpg_td_{}.mat'.format(td), dict(env.record, **result))
elif task == "compare":
    td = 80
    for compared_itcg in ["jeon", "tahk"]:
        env.modify([0., 600, 0, -20000, 10000])
        result = {"td": [], "tgo": []}
        done = False
        while done is False:
            eta = env.theta - env.q
            tgo = env.R / env.v * (1 + 0.1 * eta ** 2)
            te = td - tgo - env.t
            if compared_itcg == "jeon":
                ab = -120 * env.v ** 4 / (3 * env.qdot * env.R ** 3) * te
            elif compared_itcg == "tahk":
                ab = 100 * (env.v ** 2) / (env.R * tgo * eta) * te
            else:
                ab = 0

            done = env.step(h=0.01, ab=ab)
            result["td"].append(td - env.t)
            result["tgo"].append(tgo)
            progress_bar(env.t / td)
        print("脱靶量={:.4f} 飞行时间={:.4f}".format(env.R, env.t))
        savemat('itcg_figs/sim_{}_td_{}.mat'.format(compared_itcg, td), dict(env.record, **result))
