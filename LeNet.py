from Linear_Layers import Linear
from Convolution_Layer import Conv2D, Flatten, AvgPooling
from ActivationFunction import Relu
import JohnDL as Jh


# 搭建全连接神经网络模型
class LeNet5(Jh.Model):
    def __init__(self, class_dim):
        super().__init__()
        self.ConvUnit = Jh.Unit(
            Conv2D(1, 6, kernel_size=(5, 5), stride=1, padding=(2, 2)),  # 填充成32*32
            Relu(),
            AvgPooling((2, 2)),
            Conv2D(6, 16, kernel_size=(5, 5), stride=1),
            Relu(),
            AvgPooling((2, 2)),
            Conv2D(16, 120, kernel_size=(5, 5), stride=1),
            Relu(),
            Flatten(),
        )
        self.LinearUint = Jh.Unit(
            Linear(120, 84),
            Relu(),
            Linear(84, class_dim),
        )

    def forward(self, X):
        if len(X.shape) == 3:
            X = X.reshape((X.shape[0], 1, X.shape[-2], X.shape[-1]))
        X = self.ConvUnit(X)
        res = self.LinearUint(X)
        return res
