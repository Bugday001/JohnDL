from Native_neural_network import FullConnectionModel
from Datasets import create_data
from Utils import one_hot
import numpy as np
from Utils import computeAccuracy
import tensorflow as tf


def testAnd():
    x_test, y_test = create_data(500)
    y_test = one_hot(y_test, 2)
    model = FullConnectionModel.load("Models/test1.John")
    pre_y = model.predict(x_test)
    # for index, each in enumerate(x_test):
    #     print(x_test[index], pre_y[index], pre_y[index], np.argmax(pre_y[index]))
    acc = computeAccuracy(pre_y, y_test)
    print("acc=", acc)


def test_hand_write():
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test[10:60]
    X_test = x_test / x_test.mean()
    y_test = y_test[10:60]
    Y_test = one_hot(y_test, 10)
    model = FullConnectionModel.load("Models/Minst2.John")
    model.eval()
    pre_y = model.predict(X_test)
    pre_y_arg = []
    for index, each in enumerate(x_test):
        pre_y_arg.append(np.argmax(pre_y[index]))
    print(pre_y_arg)
    print(y_test)
    acc = computeAccuracy(pre_y, Y_test)
    print("acc=", acc)


if __name__ == "__main__":
    test_hand_write()
