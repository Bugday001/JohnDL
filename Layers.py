import numpy as np
from JohnBP import Model


class Layer(Model):
    name = ""

    def __init__(self, layer):
        super(Layer, self).__init__(layer)


# 全连接
class Linear(Layer):
    name = "linear"
    # 实例化次数
    count = 0
    i = 0

    def __init__(self, input_dim, output_dim):
        super().__init__(self)

        # 参考Pytorch初始化参数，均匀分布
        k = 1 / input_dim
        self.W = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k), size=[input_dim, output_dim])
        self.b = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k), size=[output_dim, ])

        # 初始化梯度
        self.grad_W = 0
        self.grad_b = 0
        self.shape = (input_dim, output_dim)
        Linear.count += 1
        self.mem = {}

    @classmethod
    def get_counts(cls):
        return Linear.count

    # 前向
    def __call__(self, X):
        self.mem["X"] = X
        X_shape = X.shape
        if len(X_shape) > 2:
            X = X.reshape((-1, self.shape[0]))
        W = self.W
        b = self.b
        # print(Linear.i)
        Linear.i += 1
        out = X @ W + b
        if len(X_shape) > 2:
            out = out.reshape((-1, self.shape[1]))
        return out

    # 反向
    def backward(self, grad):
        W = self.W
        grad_shape = grad.shape
        if len(grad_shape) > 2:
            grad = grad.reshape((-1, self.shape[0]))
        # 参数梯度
        # (inshape, outshape) = (inshape, m) @ (m, outshape)
        self.grad_W = self.mem["X"].T @ grad
        self.grad_b = grad.sum(axis=0)
        # 数据梯度 (m,inshape) = (m,outshape) @ (outshape, inshape)
        out_grad = grad @ W.T
        if len(grad_shape) > 2:
            out_grad = out_grad.reshape(-1, self.shape[1])
        return out_grad


# Dropout
class Dropout(Layer):
    tag = 'dropout'

    '''
    drop_ratio: 保留概率取值区间为(0, 1], 默认0.5.
                probability of an element to be zeroed
    '''

    def __init__(self, drop_ratio=0.5):
        self.__drop_ratio = drop_ratio

        super().__init__(self)
        self.__mark = None

    def __call__(self, in_batch):
        kp = self.__drop_ratio
        if not Dropout.is_training:
            return in_batch
        elif kp <= 0 or kp >= 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(kp))

        # 生成[0, 1)直接的均价分布
        tmp = np.random.uniform(size=in_batch.shape)
        # 保留/丢弃索引
        mark = (tmp >= kp).astype(int)
        # 丢弃数据, 并拉伸保留数据
        out = (mark * in_batch) / kp

        self.__mark = mark

        return out

    def backward(self, gradient):
        if self.__mark is None:
            return gradient

        out = (self.__mark * gradient) / (1.0 - self.__drop_ratio)

        return out

    def reset(self):
        self.__mark = None
