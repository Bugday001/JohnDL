"""
卷积层实现
class Padding
"""
from JohnDL import Model, Layer, LayerParam
import numpy as np
from Utils import convolution2d, calculate_HW


# 填充
class Padding(Model):
    name = "padding"
    """
    params:
    padding 填充大小(目前默认四边填充，相同范围)
    value 填充值，默认0
    """

    def __init__(self, pad, value=0):
        super().__init__(self)
        self.__pad = pad
        self.__value = value

    def __call__(self, X):
        # 直接利用numpy的pad
        pad = self.__pad
        # 目前只支持四面填充constant
        X_pad = np.zeros(X.shape[:2] + (X.shape[2] + pad*2, X.shape[3] + pad*2))
        for b in range(X.shape[0]):
            for c in range(X.shape[1]):
                X_pad[b, c] = np.pad(X[b, c], pad_width=pad, mode="constant", constant_values=self.__value)
        return X_pad

    def backward(self, gradient):
        return gradient


# 2d卷积层
class Conv2D(Layer):
    name = "conv2d"
    """
    params:
    in_channels,
    out_channels:
    kernel_size:卷积核大小,如(3,3)
    stride:1
    padding:0
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D, self).__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 所有卷积核组成的shape
        self.kernel_shape = (self.out_channels, self.in_channels) + self.kernel_size
        # 初始化参数
        W = self.generate_param(-1, 1, self.kernel_shape)
        self.W = LayerParam(Conv2D.name, "omega", W)

        b = self.generate_param(-1, 1, self.out_channels)
        self.b = LayerParam(Conv2D.name, "bias", b)

        self.mem = {}
        self.back_padding = Padding(1, 0)

    # 前向
    def __call__(self, X):
        self.mem["X"] = X
        res = convolution2d(X, self.W.value)
        m = X.shape[0]
        out_c = self.W.value.shape[0]
        for i in range(m):
            for j in range(out_c):
                # 每一个输出通道加上偏置
                res[i][j] -= self.b.value[j]
        return res

        # 反向
    def backward(self, grad):

        # 计算b参数梯度,在(dy的)最后两个维度求和
        m = grad.shape[0]
        self.b.gradient = grad.sum(axis=-1).sum(axis=-1).sum(axis=0)/m

        # 计算W参数梯度,X与dy卷积
        X = self.mem["X"]
        grad = grad.sum(axis=0).reshape(self.W.value.shape[:2]+grad.shape[-2:])
        self.W.gradient = convolution2d(X, grad).sum(axis=0).reshape(self.kernel_shape)

        # 计算X的梯度，dy 0-padding border of size 1, conv with W.T
        pad_grad = self.back_padding(grad)
        out_grad = convolution2d(pad_grad, self.W.value.T)

        return out_grad

    # 返回参数，用于更新参数值
    @property
    def params(self):
        return [self.W, self.b]


# 拉平卷积与线性层连接，特别是反向传播时
class Flatten(Layer):
    name = "flatten"

    def __init__(self, in_shape, out_shape):
        self.__inshape = in_shape
        self.__outshape = out_shape
        super().__init__(self)

    def __call__(self, X):
        return X.reshape((-1,)+self.__outshape)

    def backward(self, gradient):
        return gradient.reshape((-1,)+self.__inshape)


if __name__ == "__main__":
    padding = Padding(1, 0)
    x = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print(padding(x))
