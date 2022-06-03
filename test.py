from Native_neural_network import MyModel
from Datasets import create_data
from Utils import one_hot
import numpy as np
from Utils import computeAccuracy
import tensorflow as tf
from LeNet import LeNet5


# 与逻辑测试
def testAnd():
    x_test, y_test = create_data(500)
    y_test = one_hot(y_test, 2)
    model = MyModel.load("Models/test1.John")
    pre_y = model.predict(x_test)
    # for index, each in enumerate(x_test):
    #     print(x_test[index], pre_y[index], pre_y[index], np.argmax(pre_y[index]))
    acc = computeAccuracy(pre_y, y_test)
    print("acc=", acc)


# 手写数字测试
def test_hand_write():
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    X_test = x_test[10:20]
    X_test = X_test / X_test.mean()
    Y_test = y_test[10:20]
    # 加载模型
    model = LeNet5.load("Models/LeNet_Minst1.John")
    model.eval()
    # 预测部分并打印
    pre_y = model.predict(X_test)
    pre_y_arg = []
    for index, each in enumerate(X_test):
        pre_y_arg.append(np.argmax(pre_y[index]))
    print("预测数字:", pre_y_arg)
    print("真实数字:", Y_test)
    # 计算测试集上的ACC
    pre_y = model.predict(x_test)
    acc = computeAccuracy(pre_y, one_hot(y_test, 10))
    print("acc=", acc)


if __name__ == "__main__":
    test_hand_write()
