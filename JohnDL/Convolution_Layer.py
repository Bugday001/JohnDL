"""
卷积层实现
class Padding
"""
from JohnDL import Model
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


# kernel只沿一个方向移动，如从上到下
class Conv1D(Model):
    name = "conv1d"
    
    def __init__(self):
        super(Conv1D, self).__init__(self)


if __name__ == "__main__":
    padding = Padding(1, 0)
    x = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print(padding(x))
