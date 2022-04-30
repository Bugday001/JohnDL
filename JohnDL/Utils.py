import numpy as np


def one_hot(y, dim):
    y_shape = y.shape
    y = y.reshape((-1, 1))
    m = y.shape[0]

    a = np.ones((m, 1))
    b = np.arange(dim).reshape((1, dim))
    c = a @ b
    res = (y == c).astype(int)
    res = res.reshape(y_shape + (dim,))

    return res


def convolution2d(X, W):
    b, c, h, w = X.shape
    out_c, in_c, kh, kw = W.shape
    # 不使用分组卷积，c = in_c
    # 卷积核为正方形，kh = kw
    out_h, out_w = calculate_HW(h, w, 0, W.shape[-2:], 1)
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
                        res[i, j, n, g] += np.sum(X[i, m, n:(kh + n), g:(kw + g)] * W[j, m])
    return res


# 计算输出矩阵的HW
def calculate_HW(H_in, W_in, padding, kernel_size, stride):
    H_out = (H_in + 2 * padding - 1 * (kernel_size[0] - 1) - 1) / stride + 1
    W_out = (W_in + 2 * padding - 1 * (kernel_size[1] - 1) - 1) / stride + 1
    return int(H_out), int(W_out)

# 评估模型
def evaluate(model, x, y):
    pre_y = model.predict(x)
    accuracy = computeAccuracy(pre_y, y)
    return accuracy


# 计算精确度
def computeAccuracy(prob, labels):
    predicitions = np.argmax(prob, axis=1)
    truth = np.argmax(labels, axis=1)
    return np.mean(predicitions == truth)


if __name__ == "__main__":
    y = np.array([1, 2, 1, 2])
    Y = one_hot(y.reshape((-1, 1)), 3)
    print(y, "\n", Y)
