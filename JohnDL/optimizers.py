"""
优化器
"""
import numpy as np


class Optimizer:

    def __init__(self, model):
        self.model = model

    def step(self):
        for ly in self.model.layers["linear"]:
            for param in ly.params:
                self.update_param(param)

    def update_param(self, param):
        raise Exception("optimizer not implement")


# 固定学习率优化器
class Fixed(Optimizer):
    """
    lr: 学习率
    """

    def __init__(self, model, lr):
        super().__init__(model)
        assert lr is not None, "Please input learning rate"
        self.__lr = lr

    def update_param(self, param):
        param.value += self.__lr * param.gradient
