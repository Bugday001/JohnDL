"""
优化器
"""
import numpy as np


class Optimizer:

    def __init__(self, model):
        self.model = model

    # 获取模型每一层的每一个param并更新
    def step(self):
        for ly in self.model.layers["has_param"]:
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
        param.value -= self.__lr * param.gradient


# 学习率优化器: Adam算法
class Adam(Optimizer):
    """
    mdpr: 动量衰减率 0<mdpr<1
    sdpr: 积累量衰减率 0<sdpr<1
    """

    def __init__(self, model, lr=1e-4, mdpr=0.9, sdpr=0.99):
        super(Adam, self).__init__(model)
        self.__lr = lr
        self.__mdpr = mdpr
        self.__sdpr = sdpr
        self.epsilon = 1e-7

        if mdpr <= 0 or mdpr >= 1:
            raise Exception("invalid mdpr:%f" % mdpr)

        if sdpr <= 0 or sdpr >= 1:
            raise Exception("invalid sdpr:%f" % sdpr)

    def update_param(self, param):
        if not hasattr(param, 'adam_momentum'):
            # 添加动量属性
            param.adam_momentum = np.zeros(param.value.shape)

        if not hasattr(param, 'adam_mdpr_t'):
            # mdpr的t次方
            param.adam_mdpr_t = 1

        if not hasattr(param, 'adam_storeup'):
            # 添加积累量属性
            param.adam_storeup = np.zeros(param.value.shape)

        if not hasattr(param, 'adam_sdpr_t'):
            # 动量sdpr的t次方
            param.adam_sdpr_t = 1

        # 计算动量
        param.adam_momentum = param.adam_momentum * self.__mdpr + param.gradient * (1 - self.__mdpr)
        # 偏差修正
        param.adam_mdpr_t *= self.__mdpr
        momentum = param.adam_momentum / (1 - param.adam_mdpr_t)

        # 计算积累量
        param.adam_storeup = param.adam_storeup * self.__sdpr + (param.gradient ** 2) * (1 - self.__sdpr)
        # 偏差修正
        param.adam_sdpr_t *= self.__sdpr
        storeup = param.adam_storeup / (1 - param.adam_sdpr_t)

        # g'
        grad = self.__lr * momentum / (np.sqrt(storeup) + self.epsilon)
        param.value -= grad
