import numpy as np
import Functions as F


# 交叉熵损失
class CrossEntropy:
    def __init__(self):
        self.mem = {}
        self.epsilon = 1e-7  # 防止求导后分母为 0

    def forward(self, p, y):
        self.mem['p'] = p
        log_p = np.log(p + self.epsilon)
        return np.nan_to_num(np.mean(np.sum(-y * log_p, axis=1)))

    def backward(self, y):
        p = self.mem['p']
        return -y * (1 / (p + self.epsilon))


'''
多类别交叉熵损失函数
'''


class CategoricalCrossEntropy:
    """
    form_logits: 是否把输入数据转换成概率形式. 默认True.
                如果输入数据已经是概率形式可以把这个参数设置成False。
    """

    def __init__(self, form_logits=True):
        self.__form_logists = form_logits
        self.__grad = None

    '''
    输入形状为(m, n)
    '''

    def __call__(self, y_true, y_pred):
        # m = y_true.shape[0]
        n = y_true.shape[-1]
        m = y_true.reshape((-1, n)).shape[0]
        # pdb.set_trace()
        if not self.__form_logists:
            # 计算误差
            loss = (-y_true * np.log(y_pred)).sum(axis=0) / m
            # 计算梯度
            self.__grad = -y_true / (m * y_pred)
            return loss.sum()

        # 转换成概率分布
        y_prob = F.prob_distribution(y_pred)
        # pdb.set_trace()
        # 计算误差
        loss = (-y_true * np.log(y_prob)).sum(axis=0) / m
        # 计算梯度
        self.__grad = (y_prob - y_true) / m

        return loss.sum()

    @property
    def gradient(self):
        return self.__grad


# 均方误差损失函数
class Mse:

    def __init__(self):
        self.__grad = None

    def __call__(self, y_true, y_pred):
        err = y_pred - y_true
        loss = (err ** 2).mean(axis=0) / 2

        n = y_true.shape[0]
        self.__grad = err / n
        # pdb.set_trace()
        return loss.sum()

    @property
    def gradient(self):
        return self.__grad