import numpy as np
from Layers import Linear
from ActivationFunction import Relu, Softmax, Sigmoid, Tanh
from Loss import CrossEntropy, CategoricalCrossEntropy, Mse
from tqdm.std import trange
import tensorflow as tf
from Datasets import Dataloader, create_data
from Utils import one_hot, computeAccuracy
import JohnBP as John


# 搭建全连接神经网络模型
class FullConnectionModel(John.Model):
    def __init__(self, input_size, class_dim):
        super().__init__()
        self.linear1 = Linear(input_size, 48)
        self.relu1 = Relu()
        self.linear2 = Linear(48, class_dim)
        self.cross_en = CategoricalCrossEntropy()

    def forward(self, X, y):
        X = X.reshape((-1, 28*28))
        X = self.linear1.forward(X)
        X = self.relu1.forward(X)
        X = self.linear2.forward(X)
        self.loss = self.cross_en(X, y)
        return X

    def backward(self, y):
        self.loss_grad = self.cross_en.gradient
        self.h2_grad, self.W2_grad = self.linear2.backward(self.loss_grad)

        self.h1_relu_grad = self.relu1.backward(self.h2_grad)
        self.h1_grad, self.W1_grad = self.linear1.backward(self.h1_relu_grad)

    def predict(self, X, y):
        pre_y = self.forward(X, y)

        return pre_y


# 训练一次模型
def trainOneStep(model, x_train, y_train, learning_rate=1e-5):
    pre_y = model.forward(x_train, y_train)
    model.backward(y_train)
    model.linear1.W += learning_rate * model.W1_grad[0]
    model.linear1.b += learning_rate * model.W1_grad[1]
    model.linear2.W += learning_rate * model.W2_grad[0]
    model.linear2.b += learning_rate * model.W2_grad[1]
    # model.linear3.W += -learning_rate * model.W3_grad[0]
    # model.linear3.b += -learning_rate * model.W3_grad[1]
    loss = model.loss

    return loss, pre_y


# 训练模型和寻优
def train(x_train, y_train, x_validation, y_validation):
    epochs = 500
    learning_rate = 0.01
    batch_size = 64
    # 在验证集上寻优
    print("Start seaching the best parameter...\n")
    model = FullConnectionModel(28*28, 10)

    bar = trange(epochs)  # 使用 tqdm 第三方库，调用 tqdm.std.trange 方法给循环加个进度条
    for epoch in bar:
        x, y = Dataloader(x_train, y_train, batch_size)
        for i in range(y_train.shape[0]//batch_size):
            loss, pre_y = trainOneStep(model, x[i], y[i].reshape((y[i].shape[0], y[i].shape[-1])), learning_rate)
        accuracy = computeAccuracy(pre_y, y[-2].reshape((y[-2].shape[0], y[-2].shape[-1])))
        bar.set_description(f'epoch={epoch + 1: <3}, loss={loss: <10.8}, accuracy={accuracy: <8.6}')  # 给进度条加个描述
    bar.close()

    validation_loss, validation_accuracy = evaluate(model, x_validation, y_validation)
    print(f"validation_loss={validation_loss}, validation_accuracy={validation_accuracy}.\n")

    return model


# 评估模型
def evaluate(model, x, y):
    pre_y = model.forward(x, y)
    loss = model.loss
    accuracy = computeAccuracy(pre_y, y)
    return loss, accuracy

# 逻辑与
def AndTest():
    # create data
    x_train, y_train = create_data(900)
    x_test, y_test = create_data(10)
    y_train = one_hot(y_train, 2)
    y_test = one_hot(y_test, 2)

    # x_test = np.array([[1, 0]])
    # y_test = np.array([[0, 1]])
    model = train(x_train, y_train, x_test, y_test)
    pre_y = model.forward(x_test, y_test)

    for index, each in enumerate(x_test):
        print(x_test[index], y_test[index], pre_y[index], np.argmax(pre_y[index]))
    loss, accuracy = evaluate(model, x_train, y_train)
    print(f'Evaluate the best model, test loss={loss:0<10.8}, accuracy={accuracy:0<8.6}.')
    path = "Models/test2.John"
    model.save(path)
    print("save models at:", path)


def hand_write():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X_train = x_train / x_train.mean()
    X_test = x_test / x_test.mean()
    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)

    model = train(X_train, y_train, X_test, y_test)

    loss, accuracy = evaluate(model, X_test, y_test)
    print(f'Evaluate the best model, test loss={loss:0<10.8}, accuracy={accuracy:0<8.6}.')
    path = "Models/Minst2.John"
    model.save(path)
    print("save models at:", path)


if __name__ == '__main__':
    # AndTest()
    hand_write()
