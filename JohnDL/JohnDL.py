import pickle


class Model:
    __layers = {"linear": [], "backtrace": []}
    __train = True

    def __init__(self, layer):
        if layer is not None:
            Model.__layers["backtrace"].append(layer)
            if layer.name == "linear":
                Model.__layers["linear"].append(layer)

    # 反向传播
    def backward(self, gradient):
        g = gradient
        # pdb.set_trace()
        count = len(Model.__layers["backtrace"])
        for i in range(count - 1, -1, -1):
            ly = Model.__layers["backtrace"][i]
            g = ly.backward(g)

    # 更新参数
    def renew_params(self, lr):
        for ly in Model.__layers["linear"]:
            ly.W += lr * ly.grad_W
            ly.b += lr * ly.grad_b

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
