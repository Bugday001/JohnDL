import numpy as np
import Functions as F
from JohnDL import Model


# ReLU
class Relu(Model):
    name = 'relu'

    def __init__(self):
        super().__init__(self)
        self.mem = {}

    def __call__(self, X):
        self.mem["X"] = X
        return np.where(X > 0, X, np.zeros_like(X))

    def backward(self, grad_y):
        X = self.mem["X"]
        return (X > 0).astype(np.float32) * grad_y


# Softmax
class Softmax(Model):
    name = 'softmax'

    def __init__(self):
        super().__init__(self)
        self.mem = {}
        self.epsilon = 1e-12  # 防止求导后分母为 0

    def __call__(self, p):
        # 1
        # p_exp = np.exp(p)
        # denominator = np.sum(p_exp, axis=1, keepdims=True)
        # s = p_exp / (denominator + self.epsilon)
        # 2
        # assert (len(p.shape) == 2)
        # row_max = np.max(p).reshape(-1, 1)
        # p -= row_max
        # p_exp = np.exp(p)
        # s = p_exp / np.sum(p_exp, axis=1, keepdims=True)
        # 3
        s = .5 * (1 + np.tanh(.5 * p))
        self.mem["s"] = s
        # self.mem["p_exp"] = p_exp
        return s

    def backward(self, grad_s):
        s = self.mem["s"]
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))
        tmp = np.matmul(np.expand_dims(grad_s, axis=1), sisj)
        tmp = np.squeeze(tmp, axis=1)
        grad_p = -tmp + grad_s * s
        return grad_p


# Sigmoid
class Sigmoid(Model):
    name = 'sigmoid'

    def __init__(self):
        super().__init__(self)
        self.__grad = None

    def __call__(self, in_batch):
        out = F.sigmoid(in_batch)
        self.__grad = out * (1 - out)
        return out

    def backward(self, gradient):
        return gradient * self.__grad


'''
tanh 激活函数
'''


class Tanh:
    name = 'tanh'

    def __init__(self):
        self.__grad = None

    def __call__(self, in_batch):
        out = F.tanh(in_batch)
        self.__grad = 1 - out ** 2
        return out

    def backward(self, gradient):
        return gradient * self.__grad
