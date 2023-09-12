import os
import numpy as np
import keras
import keras.optimizers.optimizer_v2.adam as adam
from keras.layers import Input, Dense, concatenate
from keras.activations import relu
from keras.callbacks import TensorBoard, LearningRateScheduler, EarlyStopping
from scipy.io import loadmat, savemat


def load_data(train_set):
    data_raw = loadmat(train_set)
    x = data_raw["x"]
    y = data_raw["y"]
    print("training data load done!")
    return x, y


def init_network(units, n):
    # 创建网络
    x = Input(shape=[4], name="n{}_x".format(n))
    l1 = Dense(units=units, activation='relu', name="n{}_d1".format(n))(x)
    l1 = Dense(units=units, activation='relu', name="n{}_d2".format(n))(l1)
    l1 = Dense(units=units, activation='relu', name="n{}_d3".format(n))(l1)
    y = Dense(1, name="n{}_y".format(n))(l1)
    model = keras.Model(inputs=x, outputs=y, name="m{}".format(n))
    model.compile(loss="mse", optimizer=adam.Adam(learning_rate=0.001))
    return model


def train_network(train_set, n):
    if not os.path.exists("model"):
        os.makedirs("model")
    model = init_network(units=100, n=n)
    x, y = load_data(train_set)

    def scheduler(epoch):
        if epoch % 5 == 0 and epoch != 0:
            lr = keras.backend.get_value(model.optimizer.lr)
            keras.backend.set_value(model.optimizer.lr, lr * 0.2)
        return keras.backend.get_value(model.optimizer.lr)

    sc = LearningRateScheduler(scheduler)
    tb = TensorBoard(log_dir='logs/log_{}'.format(n), write_images=True)
    model.fit(x, y, batch_size=500, epochs=10, validation_split=0.02, callbacks=[sc, tb])
    model.save("./model/model_{}.h5".format(n))
    return model


def transfer_ensemble(learn_rate):
    """
    迁移-集成学习
    :param learn_rate: 迁移学习率
    :return:
    """
    x = Input(shape=[4], name="tren_x")
    y_ = []
    for n in range(5):
        model = keras.models.load_model("model/model_{}.h5".format(n))  # 加载各个模型
        for layer in model.layers:  # 冻结参数
            layer.trainable = False
            y_.append(model(x))
    y_ = concatenate(y_, name="tren_mid")
    y = Dense(units=1, kernel_initializer=keras.initializers.Constant(value=1 / 5), name="tren_y")(y_)
    model_en = keras.Model(inputs=x, outputs=y)
    model_en.compile(loss="mse", optimizer=adam.Adam(learning_rate=learn_rate))
    return model_en


def train_tren(tdata, vdata, lr, bs, ep):
    """
    迁移-集成学习
    :param tdata: 迁移数据训练集
    :param vdata: 迁移数据测试集
    :param lr: 学习率
    :param bs: 迁移样本尺寸
    :param ep: 迁移样本使用次数
    :return:
    """
    x_trans = tdata[0]
    y_trans = tdata[1]
    model = transfer_ensemble(lr)
    tb = TensorBoard(log_dir='logs/log_tren', write_images=True)
    el = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
    model.fit(x_trans, y_trans, batch_size=bs, epochs=ep, validation_data=vdata, callbacks=[el, tb])
    model.save("./model/model_tren.h5")
    return model


if __name__ == '__main__':
    # 训练预训练模型
    K = [0.2, 0.5, 1.0, 2.0, 4.0]
    for i in range(5):
        k = (K[i], K[i], K[i])
        train_network('mats/itcg_train_data_{:.1f}_{:.1f}_{:.1f}.mat'.format(*k), i)

    # 训练迁移-集成模型
    k = (3.0, 1.5, 1.0)
    x, y = load_data('mats/itcg_train_data_{:.1f}_{:.1f}_{:.1f}.mat'.format(*k))
    tl0 = 86451  # 第一条轨迹的数据量
    tl1 = 150898  # 前两条轨迹的数据量
    train_tren((x[:tl0, :], y[:tl0, :]), (x[tl0:tl1, :], y[tl0:tl1, :]), 0.001, 50, 1)  # 训练第一条轨迹，测试第二条轨迹

    # 测试所有模型在验证集的性能
    y_hat = np.zeros([tl1 - tl0, 6])
    for i in range(5):
        model = keras.models.load_model("model/model_{}.h5".format(i))
        y_hat[:, i] = np.squeeze(model.predict(x[tl0:tl1, :]))

    model_tren = keras.models.load_model("model/model_tren.h5")
    y_hat[:, -1] = np.squeeze(model_tren.predict(x[tl0:tl1, :]))
    savemat("itcg_figs/en_test.mat", {"y": y[tl0:tl1, :], "y_hat": y_hat})

    s = int(x.shape[0] * 0.3)
    savemat("itcg_figs/dnn_test.mat",
            {"y": y[-s:, :], "y_hat": model_tren.predict(x[-s:, :], use_multiprocessing=True)})
