import pickle
import numpy as np
import time


class Model:
    layers = {"linear": [], "backtrace": []}
    __train = True

    def __init__(self, layer):
        if layer is not None and layer.name != "padding":
            Model.layers["backtrace"].append(layer)
            if layer.name == "linear" or layer.name == "conv2d":
                Model.layers["linear"].append(layer)

    # 反向传播
    def backwards(self, gradient):
        g = gradient
        count = len(Model.layers["backtrace"])
        for i in range(count - 1, -1, -1):
            # start = time.time()
            ly = Model.layers["backtrace"][i]
            g = ly.backward(g)
            # print(ly.name, time.time()-start)

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

    # 用户来实现
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

    def __init__(self, layer):
        super(Layer, self).__init__(layer)

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
