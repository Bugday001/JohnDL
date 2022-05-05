"""
卷积层实现
class Padding
"""
from JohnDL import Model, Layer, LayerParam
import numpy as np
import Functions as F


# 填充
class Padding(Model):
    name = "padding"
    """
    params:
    padding 填充大小(目前默认四边填充，相同范围)
    value 填充值，默认0
    """

    def __init__(self, pad, value=0, mode="constant"):

        self.__pad = pad
        self.__value = value
        self.__mode = mode
        self.mem = {}

    def __call__(self, X):
        self.mem["X_shape"] = X.shape
        # 直接利用numpy的pad
        pad = self.__pad
        mode = self.__mode
        # 目前只支持四面填充constant
        X_pad = np.zeros(X.shape[:2] + (X.shape[2] + pad[0] * 2, X.shape[3] + pad[1] * 2))
        for b in range(X.shape[0]):
            for c in range(X.shape[1]):
                X_pad[b, c] = np.pad(X[b, c], pad_width=((pad[0],)*2, (pad[1],)*2), mode=mode, constant_values=self.__value)
        return X_pad

    def backward(self, gradient):
        pad = self.__pad
        X_shape = self.mem["X_shape"]
        return gradient[..., pad[0]:(gradient.shape[2]-pad[0]), pad[1]:(gradient.shape[3]-pad[1])]


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

        # 所有卷积核组成的shape
        self.kernel_shape = (self.out_channels, self.in_channels) + self.kernel_size
        # 初始化参数
        W = self.generate_param(-1, 1, self.kernel_shape)
        self.W = LayerParam(Conv2D.name, "omega", W)

        b = self.generate_param(-1, 1, self.out_channels)
        self.b = LayerParam(Conv2D.name, "bias", b)

        self.mem = {"backward": 0}

        # 前向传播时由用户输入padding
        if type(padding) == int:
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.forward_padding = Padding(self.padding, pad_value, pad_mode)

    # 前向
    def __call__(self, X):

        if self.padding != (0, 0):
            X = self.forward_padding(X)
        self.mem["X"] = X

        res = F.conv_tensor(X, self.W.value, self.stride)
        out_c = self.W.value.shape[0]
        for j in range(out_c):
            # 每一个输出通道加上偏置
            res[:, j] += self.b.value[j]
        return res

        # 反向

    def backward(self, grad):
        X = self.mem["X"]
        # 若stride不为1则需要填充
        if self.stride != (1, 1):
            inserted_grad = F.stride_insert(grad, self.stride)
        else:
            inserted_grad = grad
        # 计算X梯度时使用给输入梯度padding
        if self.mem["backward"] == 0:
            H_out, W_out = X.shape[-2:]
            H_in, W_in = inserted_grad.shape[-2:]
            padding = F.calculate_pad(H_in, W_in, H_out, W_out, self.kernel_size, (1, 1))
            self.back_padding = Padding(padding, 0)
            self.mem["backward"] = 1

        # 计算b参数梯度,在(dy的)最后两个维度求和
        m = grad.shape[0]
        self.b.gradient = grad.sum(axis=-1).sum(axis=-1).sum(axis=0) / m

        # 计算W参数梯度,X与dy卷积
        X = X.sum(axis=0).reshape((self.in_channels, 1)+X.shape[-2:])
        # 在batch上求和,output shape:(C_, C, oh, ow)
        inserted_grad_sum = inserted_grad.sum(axis=0).reshape((self.kernel_shape[0], 1)+inserted_grad.shape[-2:])
        gradient = F.conv_tensor(X, inserted_grad_sum, (1, 1))
        self.W.gradient = gradient.transpose((1, 0, 2, 3)) / m

        # 计算X的梯度，dy 0-padding border of size 1, conv with W'
        pad_grad = self.back_padding(inserted_grad)
        out_grad = F.conv_tensor(pad_grad, np.rot90(self.W.value, 2, axes=(2, 3)).transpose((1, 0, 2, 3)), (1, 1))
        # padding的backward
        out_grad = self.forward_padding.backward(out_grad)
        return out_grad

    # 返回参数，用于更新参数值
    @property
    def params(self):
        return [self.W, self.b]


# 拉平卷积与线性层连接，特别是反向传播时reshape
class Flatten(Model):
    name = "flatten"

    def __init__(self):

        self.__outshape = None
        self.__inshape = None

    def __call__(self, X):
        self.__inshape = X.shape[1:]
        return X.reshape((-1,) + (X[0].size, ))

    def backward(self, gradient):
        return gradient.reshape((-1,) + self.__inshape)


class MaxPooling(Layer):
    name = "maxpooling"

    def __init__(self, pool_size=(2, 2), stride=None, padding=0):

        self.pool_size = pool_size
        if stride is None:
            self.stride = pool_size
        else:
            self.stride = stride

        self.mem = {}

    def __call__(self, X):
        self.mem["X"] = X
        out = F.max_pooling(X, self.pool_size, self.stride)
        self.index = out.repeat(self.pool_size[1], axis=-1).repeat(self.pool_size[0], axis=-2) == X  # 记录最大值的位置
        return out

    # 反向
    def backward(self, grad):
        output = grad.repeat(self.pool_size[1], axis=-1).repeat(self.pool_size[0], axis=-2) * self.index
        return output


# 平均池化
class AvgPooling(Layer):
    name = "Avgpooling"

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
        return F.avg_pooling(X, self.pool_size, self.stride)

    # 反向
    def backward(self, grad):
        grad = grad / (self.pool_size[0]*self.pool_size[1])
        return grad.repeat(self.pool_size[1], axis=-1).repeat(self.pool_size[0], axis=-2)


if __name__ == "__main__":
    pooling = MaxPooling((2, 2))
    X = np.array([[[[1, -1, 2, 4],
                    [-1, 2, 3, 5],
                    [4, 1, -1, 9],
                    [-1, 2, 3, 5]]]])
    pooling(X)
    a = np.array([[[[1, 2],
                    [3, 4]]]])
    print(pooling.backward(a))
