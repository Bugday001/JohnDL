import numpy as np
from JohnDL import Layer, LayerParam


# 全连接
class Linear(Layer):
    name = "linear"
    # 实例化次数
    count = 0

    def __init__(self, input_dim, output_dim):
        super().__init__(self)

        # 参考Pytorch初始化参数，均匀分布
        k = 1 / input_dim
        W = self.generate_param(low=-np.sqrt(k), high=np.sqrt(k), shape=[input_dim, output_dim])
        b = self.generate_param(low=-np.sqrt(k), high=np.sqrt(k), shape=[output_dim, ])
        self.W = LayerParam(Linear.name, "omega", W)
        self.b = LayerParam(Linear.name, "bias", b)

        # 初始化梯度
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
        W = self.W.value
        b = self.b.value
        out = X @ W + b
        if len(X_shape) > 2:
            out = out.reshape((-1, self.shape[1]))
        return out

    # 反向
    def backward(self, grad):
        W = self.W.value
        grad_shape = grad.shape
        if len(grad_shape) > 2:
            grad = grad.reshape((-1, self.shape[0]))
        # 参数梯度
        # (inshape, outshape) = (inshape, m) @ (m, outshape)
        self.W.gradient = self.mem["X"].T @ grad
        self.b.gradient = grad.sum(axis=0)
        # 数据梯度 (m,inshape) = (m,outshape) @ (outshape, inshape)
        out_grad = grad @ W.T
        if len(grad_shape) > 2:
            out_grad = out_grad.reshape(-1, self.shape[1])
        return out_grad

    # 返回参数，用于更新参数值
    @property
    def params(self):
        return [self.W, self.b]


# Dropout
class Dropout(Layer):
    name = 'dropout'

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


class BatchNormalization(Layer):
    name = "BN"
    """Batch normalization.
    # 特征维度（全连接层中神经元的个数或者卷积中特征图通道的个数）
    # 用于区分当前是对全连接层标准化还是卷积层
    eps：平滑处理
    """
    def __init__(self, num_features=None, num_dims=4, momentum=0.99, eps=1e-5):
        self.momentum = momentum
        self.running_mean = None
        self.running_var = None

        super(BatchNormalization, self).__init__(self)

        self.inshape = [1, num_features]
        if num_dims == 4:
            self.inshape = [1, num_features, 1, 1]
            # 由于后面标准化时是以每个特征图为单位进行计算，因此会用到广播机制，所以需要保持4个维度
        self.momentum = momentum
        self.num_features = num_features
        self.eps = eps
        self.gamma = LayerParam(BatchNormalization.name, "gamma", np.ones(self.inshape))
        self.beta = LayerParam(BatchNormalization.name, "beta", np.zeros(self.inshape))

    def __call__(self, X, training=True):
        # Initialize running mean and variance if first run
        if self.running_mean is None:
            self.running_mean = np.mean(X, axis=0)
            self.running_var = np.var(X, axis=0)

        if BatchNormalization.is_training:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Statistics saved for backward pass
        self.X_centered = X - mean
        self.stddev_inv = 1 / np.sqrt(var + self.eps)

        X_norm = self.X_centered * self.stddev_inv
        output = self.gamma.value * X_norm + self.beta.value

        return output

    def backward(self, accum_grad):

        # Save parameters used during the forward pass
        gamma = self.gamma.value

        # If the layer is trainable the parameters are updated
        X_norm = self.X_centered * self.stddev_inv
        self.gamma.gradient = np.sum(accum_grad * X_norm, axis=0)
        self.beta.gradient = np.sum(accum_grad, axis=0)

        batch_size = accum_grad.shape[0]

        # The gradient of the loss with respect to the layer inputs (use weights and statistics from forward pass)
        accum_grad = (1 / batch_size) * gamma * self.stddev_inv * (
            batch_size * accum_grad
            - np.sum(accum_grad, axis=0)
            - self.X_centered * self.stddev_inv**2 * np.sum(accum_grad * self.X_centered, axis=0)
            )

        return accum_grad

    # 返回参数，用于更新参数值
    @property
    def params(self):
        return [self.gamma, self.beta]
