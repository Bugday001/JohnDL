"""
卷积层实现
class Padding
"""
from JohnDL import Model, Layer, LayerParam
import numpy as np


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
        return np.pad(X, pad_width=self.__pad, mode="constant", constant_values=self.__value)


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
        self.kernel_shape = (self.out_channels, self.in_channels)+self.kernel_size

        W = self.generate_param(self.kernel_shape)
        self.W = LayerParam(Conv2D.name, "omega", W)

        b = self.generate_param(self.out_channels)
        self.b = LayerParam(Conv2D.name, "bias", b)

    # 前向
    def __call__(self, X):
        b, c, h, w = X.shape
        out_c, in_c, kh, kw = self.kernel_shape
        # 不使用分组卷积，c = in_c
        # 卷积核为正方形，kh = kw
        out_h, out_w = self.calculate_HW(h, w)
        res = np.zeros((b, out_c, out_h, out_w))
        # 最终输出形状：b*out_c*(h-kh+1)*(w-kw+1)
        # print(len(res), len(res[0]), len(res[0][0]), len(res[0][0][0]))
        for i in range(b):
            # 关于batch，目前只能串行完成

            for j in range(out_c):
                # 计算每一组的结果

                for m in range(c):
                    for n in range(h - kh + 1):
                        for g in range(w - kw + 1):
                            # 计算每一个位置的值

                            ans = 0
                            for k1 in range(kh):
                                for k2 in range(kw):
                                    ans += X[i][m][n + k1][g + k2] * self.W.value[j][m][k1][k2]
                            res[i][j][n][g] += ans
                # 每一个输出通道加上偏置
                res[i][j] -= self.b.value[j]
        return res

    # 计算输出矩阵的HW
    def calculate_HW(self, H_in, W_in):
        H_out = (H_in+2*self.padding-1*(self.kernel_size[0]-1)-1)/self.stride+1
        W_out = (W_in+2*self.padding-1*(self.kernel_size[1]-1)-1)/self.stride+1
        return int(H_out), int(W_out)


if __name__ == "__main__":
    padding = Padding(1, 0)
    x = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print(padding(x))
