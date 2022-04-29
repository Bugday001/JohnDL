import numpy as np


def one_hot(y, dim):
    y_shape = y.shape
    y = y.reshape((-1, 1))
    m = y.shape[0]

    a = np.ones((m, 1))
    b = np.arange(dim).reshape((1, dim))
    c = a @ b
    res = (y == c).astype(int)
    res = res.reshape(y_shape + (dim,))

    return res


# 评估模型
def evaluate(model, x, y):
    pre_y = model.predict(x)
    accuracy = computeAccuracy(pre_y, y)
    return accuracy


# 计算精确度
def computeAccuracy(prob, labels):
    predicitions = np.argmax(prob, axis=1)
    truth = np.argmax(labels, axis=1)
    return np.mean(predicitions == truth)


if __name__ == "__main__":
    y = np.array([1, 2, 1, 2])
    Y = one_hot(y.reshape((-1, 1)), 3)
    print(y, "\n", Y)
