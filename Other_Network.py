import numpy as np
from Linear_Layers import Linear, Dropout, BatchNormalization
from Convolution_Layer import Conv2D, Flatten, MaxPooling, AvgPooling
from ActivationFunction import Relu, Softmax, Sigmoid, Tanh
from Loss import CategoricalCrossEntropy, Mse
from tqdm import tqdm
import tensorflow as tf
from Datasets import DataLoader, create_data
from Utils import one_hot, computeAccuracy, evaluate
import JohnDL as Jh
from optimizers import Fixed, Adam
import math
import time
from functools import wraps


# @fn_timer放在函数前
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.__name__, str(t1 - t0))
              )
        return result

    return function_timer


# 搭建全连接神经网络模型
class FullConnectionModel(Jh.Model):
    def __init__(self, input_size, class_dim):
        self.ConvUnit = Jh.Unit(
            Conv2D(1, 4, kernel_size=(3, 3), padding=1),
            MaxPooling((2, 2)),
            BatchNormalization((4, 14, 14), 4),
            Relu(),
            Conv2D(4, 4, kernel_size=(3, 3), padding=1, stride=(1, 1)),
            MaxPooling((2, 2)),
            Flatten(),
        )
        self.LinearUint = Jh.Unit(
            Relu(),
            Linear(7 * 7 * 4, 128),
            # Dropout(0.1),
            Relu(),
            Linear(128, class_dim),
        )

    def forward(self, X):
        if len(X.shape) == 3:
            X = X.reshape((X.shape[0], 1, X.shape[-2], X.shape[-1]))
        X = self.ConvUnit(X)
        res = self.LinearUint(X)
        return res


# 训练模型和寻优
def train(x_train, y_train, x_validation, y_validation):
    epochs = 2
    learning_rate = 0.0001
    batch_size = 64

    print("Start training...\n")
    # 模型
    model = FullConnectionModel(28 * 28, 10)
    # 损失函数
    criterion = CategoricalCrossEntropy()
    # 优化器
    optimizer = Adam(model, learning_rate)

    total_size = math.ceil(y_train.shape[0] / batch_size)
    for epoch in range(epochs):
        data_loader = DataLoader(x_train, y_train, batch_size, True)
        # each batch
        with tqdm(total=total_size) as _tqdm:  # 使用需要的参数对tqdm进行初始化
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, epochs))  # 设置前缀 一般为epoch的信息
            for x, y in data_loader():
                start = time.time()
                pre_y = model.forward(x)
                forward_time = time.time()
                # 计算误差
                loss = criterion(y, pre_y)
                # 后向传播
                model.backwards(criterion.gradient)
                backward_time = time.time()
                # 调整参数
                optimizer.step()
                # acc
                acc = computeAccuracy(pre_y, y)
                _tqdm.set_postfix(
                    {"loss": '{:.6f}'.format(loss), "acc": '{:.6f}'.format(acc)})  # 输入一个字典
                _tqdm.update(1)  # 设置你每一次想让进度条更新的iteration 大小
                end = time.time()
                # 打印各部分时间，forward:0.33, backward: 0.87, whole: 1.193
                # 优化后:forward:0.0249, backward: 0.0324, whole: 0.058
                # print("forward:", forward_time-start,
                #       "backward:", backward_time-forward_time,
                #       "all:", end-start)
        model.eval()
        validation_accuracy = evaluate(model, x_validation, y_validation)
        model.train()
        print(f"validation_accuracy={validation_accuracy}.\n")

    return model


# 逻辑与
def AndTest():
    # create data
    x_train, y_train = create_data(900)
    x_test, y_test = create_data(10)
    y_train = one_hot(y_train, 2)
    y_test = one_hot(y_test, 2)

    model = train(x_train, y_train, x_test, y_test)
    pre_y = model.forward(x_test)

    for index, each in enumerate(x_test):
        print(x_test[index], y_test[index], pre_y[index], np.argmax(pre_y[index]))
    accuracy = evaluate(model, x_train, y_train)
    print(f'Evaluate the best model, test accuracy={accuracy:0<8.6}.')
    path = "Models/test1.John"
    model.save(path)
    print("save models at:", path)


# Minst
def hand_write():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X_train = x_train / x_train.mean()
    X_test = x_test / x_test.mean()
    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)

    model = train(X_train, y_train, X_test, y_test)
    accuracy = evaluate(model, X_test, y_test)
    print(f'Evaluate the best model, test accuracy={accuracy:0<8.6}.')
    path = "Models/Minst2.John"
    model.save(path)
    print("save models at:", path)


if __name__ == '__main__':
    # AndTest()
    hand_write()
