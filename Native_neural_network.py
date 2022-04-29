import numpy as np
from Layers import Linear, Dropout
from ActivationFunction import Relu, Softmax, Sigmoid, Tanh
from Loss import CrossEntropy, CategoricalCrossEntropy, Mse
from tqdm.std import trange
import tensorflow as tf
from Datasets import DataLoader, create_data
from Utils import one_hot, computeAccuracy, evaluate
import JohnDL as John
from optimizers import Fixed


# 搭建全连接神经网络模型
class FullConnectionModel(John.Model):
    def __init__(self, input_size, class_dim):
        super().__init__(None)
        self.linear1 = Linear(input_size, 512)
        self.relu1 = Relu()
        self.linear2 = Linear(512, 128)
        self.dropout = Dropout(0.1)
        self.relu2 = Relu()
        self.linear3 = Linear(128, class_dim)

    def forward(self, X):
        # X = X.reshape((-1, 28*28))
        X = self.linear1(X)
        X = self.relu1(X)
        X = self.linear2(X)
        X = self.dropout(X)
        X = self.relu2(X)
        res = self.linear3(X)
        return res


# 训练模型和寻优
def train(x_train, y_train, x_validation, y_validation):
    epochs = 50
    learning_rate = 0.01
    batch_size = 64

    print("Start training...\n")
    # 模型
    model = FullConnectionModel(2, 2)
    # 损失函数
    criterion = CategoricalCrossEntropy()
    # 优化器
    optimizer = Fixed(model, learning_rate)

    # 使用 tqdm 第三方库，调用 tqdm.std.trange 方法给循环加个进度条
    bar = trange(epochs)
    for epoch in bar:
        data_loader = DataLoader(x_train, y_train, batch_size, True)
        # each batch
        for x, y in data_loader():
            Y = y
            pre_y = model.forward(x)
            # 计算误差
            loss = criterion(pre_y, Y)
            # 后向传播
            model.backward(criterion.gradient)
            # 调整参数
            optimizer.step()
        # 显示
        accuracy = computeAccuracy(pre_y, Y)
        bar.set_description(f'epoch={epoch + 1: <3}, loss={loss: <10.8}, accuracy={accuracy: <8.6}')  # 给进度条加个描述
    bar.close()

    validation_accuracy = evaluate(model, x_validation, y_validation)
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
    path = "Models/Minst1.John"
    model.save(path)
    print("save models at:", path)


if __name__ == '__main__':
    AndTest()
    # hand_write()

