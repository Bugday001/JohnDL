"""
卷积层实现
class Padding
"""
from JohnDL import Model, Layer, LayerParam
import numpy as np
from dlmath import convolution2d, calculate_HW, stride_insert, max_pooling


# 填充
class Padding(Model):
    name = "padding"
    """
    params:
    padding 填充大小(目前默认四边填充，相同范围)
    value 填充值，默认0
    """

    def __init__(self, pad, value=0, mode="constant"):
        super().__init__(self)
        self.__pad = pad
        self.__value = value
        self.__mode = mode

    def __call__(self, X):
        # 直接利用numpy的pad
        pad = self.__pad
        mode = self.__mode
        # 目前只支持四面填充constant
        X_pad = np.zeros(X.shape[:2] + (X.shape[2] + pad * 2, X.shape[3] + pad * 2))
        for b in range(X.shape[0]):
            for c in range(X.shape[1]):
                X_pad[b, c] = np.pad(X[b, c], pad_width=pad, mode=mode, constant_values=self.__value)
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
    stride:步长，默认 1。如果输入元组，第一个值是行方向(竖直移动时)的stride
    padding: 默认 0
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="constant", pad_value=0):
        super(Conv2D, self).__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if isinstance(stride, tuple):
            assert len(stride) == 2, " tuple like (1,1)"
            self.stride = stride
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            raise ValueError("Stride should be a int or tuple!")

        self.padding = padding

        # 所有卷积核组成的shape
        self.kernel_shape = (self.out_channels, self.in_channels) + self.kernel_size
        # 初始化参数
        W = self.generate_param(-1, 1, self.kernel_shape)
        self.W = LayerParam(Conv2D.name, "omega", W)

        b = self.generate_param(-1, 1, self.out_channels)
        self.b = LayerParam(Conv2D.name, "bias", b)

        self.mem = {}
        # 计算X梯度时使用给输入梯度padding
        self.back_padding = Padding(1, 0)
        # 前向传播时由用户输入padding
        self.forward_padding = Padding(padding, pad_value, pad_mode)

    # 前向
    def __call__(self, X):
        if self.padding != 0:
            X = self.forward_padding(X)

        self.mem["X"] = X
        res = convolution2d(X, self.W.value, self.stride)
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
        self.b.gradient = grad.sum(axis=-1).sum(axis=-1).sum(axis=0) / m

        # 计算W参数梯度,X与dy卷积
        X = self.mem["X"]
        grad = grad.sum(axis=0).reshape(self.W.value.shape[:2] + grad.shape[-2:])
        # 若stride不为1则需要填充
        if self.stride != (1, 1):
            inserted_grad = stride_insert(grad, self.stride)
        else:
            inserted_grad = grad
        self.W.gradient = convolution2d(X, grad, (1, 1)).sum(axis=0).reshape(self.kernel_shape)

        # 计算X的梯度，dy 0-padding border of size 1, conv with W.T
        pad_grad = self.back_padding(inserted_grad)
        out_grad = convolution2d(pad_grad, np.rot90(self.W.value, 2), (1, 1))
        # padding的backward直接返回
        return out_grad

    # 返回参数，用于更新参数值
    @property
    def params(self):
        return [self.W, self.b]


# 拉平卷积与线性层连接，特别是反向传播时reshape
class Flatten(Layer):
    name = "flatten"

    def __init__(self, in_shape, out_shape):
        self.__inshape = in_shape
        self.__outshape = out_shape
        super().__init__(self)

    def __call__(self, X):
        return X.reshape((-1,) + self.__outshape)

    def backward(self, gradient):
        return gradient.reshape((-1,) + self.__inshape)


class MaxPooling(Layer):
    name = "maxpooling"

    def __init__(self, pool_size=(2, 2), stride=None, padding=0):
        super().__init__(self)
        self.pool_size = pool_size
        if stride is None:
            self.stride = pool_size
        else:
            self.stride = stride

        self.mem = {}

    def __call__(self, X):
        self.mem["X"] = X
        return max_pooling(X, self.pool_size, self.stride)

    # 反向
    def backward(self, grad):
        X = self.mem["X"]
        b, c, h, w = grad.shape
        stride = self.stride
        kh, kw = self.pool_size
        res = np.zeros_like(X)
        for i in range(b):
            # 关于batch，目前只能串行完成
            for m in range(0, c):
                for n in range(stride[0], X.shape[-2] + 1, stride[0]):
                    for g in range(stride[1], X.shape[-1] + 1, stride[1]):
                        # 计算每一个位置的值
                        row = n - stride[0]
                        col = g - stride[1]
                        # 计算每一个位置的值
                        res[i, m, row:(kh + row), col:(kw + col)] = \
                            np.where(np.abs(X[i, m, row:(kh + row), col:(kw + col)] - np.max(X[i, m, row:(kh + row), col:(kw + col)])) < 1e-7,\
                                     grad[i, m, row // stride[0], col // stride[1]], 0)
        return res


if __name__ == "__main__":
    padding = Padding(1, 0)
    x = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print(padding(x))
