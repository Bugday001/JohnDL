import pickle
import numpy as np
import time


class Model:
    layers = {"has_param": [], "trace": []}
    __train = True

    # 反向传播，卷积层使用了95%的时间
    def backwards(self, gradient):
        g = gradient
        count = len(Model.layers["trace"])
        for i in range(count - 1, -1, -1):
            # start = time.time()
            ly = Model.layers["trace"][i]
            g = ly.backward(g)
            # print(ly.name, time.time()-start)

    # 添加模型
    def add_model(self, layer):
        if layer is not None and layer.name != "padding":
            Model.layers["trace"].append(layer)
            if layer.name == "linear" or layer.name == "conv2d" or layer.name == "BN":
                Model.layers["has_param"].append(layer)

    # 模型预测
    def predict(self, X):
        self.eval()
        pre_y = self.forward(X)
        return pre_y

    # 预测模式关闭Dropout
    def eval(self):
        Model.__train = False

    # 训练模式
    def train(self):
        Model.__train = True

    @staticmethod
    def is_training(self):
        return Model.__train

    # 用户实现
    def forward(self, X):
        raise Exception("forward not implement")

    # 保存模型
    def save(self, path):
        obj = pickle.dumps(self)
        with open(path, "wb") as f:
            f.write(obj)

    # 加载模型
    @staticmethod
    def load(path):
        obj = None
        with open(path, "rb") as f:
            try:
                obj = pickle.load(f)
            except:
                print("IOError")
        return obj


class Layer(Model):
    name = ""

    # 初始化参数
    def generate_param(self, low, high, shape):
        param = np.random.uniform(low, high, shape) * 0.1
        # param = np.ones(shape)
        return param


# 每层的参数
class LayerParam(object):
    """
    layer_name: 所属层的的名字
    name: 参数名
    value: 参数值
    """

    def __init__(self, layer_name, name, value):
        self.__name = layer_name + "/" + name
        self.value = value

        # 梯度
        self.gradient = None

    @property
    def name(self):
        return self.__name


# 序列
class Unit(Model):
    def __init__(self, *args):
        self.layers = {"unit_trace": []}
        self.isTraced = False
        for module in args:
            if module is not None:
                self.layers["unit_trace"].append(module)

    # 正向传播
    def __call__(self, X):
        for ly in self.layers["unit_trace"]:
            X = ly(X)
        # 若第一次训练，将单元中的层加入trace list，backward时使用
        if not self.isTraced:
            for ly in self.layers["unit_trace"]:
                self.add_model(ly)
            self.isTraced = True
        return X
