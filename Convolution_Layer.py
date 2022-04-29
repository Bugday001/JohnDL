"""
卷积层实现
class Padding
"""
from JohnBP import Model


# 填充
class Padding(Model):
    name = "padding"
    """
    params:
    padding 填充大小(目前默认四边填充，相同范围)
    value 填充值，默认0
    """
    def __init__(self, padding, value=0):
        super().__init__(self)
        self.__pad = padding
        self.__value = value
