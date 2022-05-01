import numpy as np

'''
实现一些常用的数学函数
'''


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


'''
转换成分布列
x shape (m, ..., n)
'''


def prob_distribution(x):
    xshape = x.shape
    x = x.reshape((-1, xshape[-1]))

    expval = np.exp(x)
    sum = expval.sum(axis=1).reshape(-1, 1)

    prob_d = expval / sum
    prob_d = prob_d.reshape(xshape)

    return prob_d


'''
类别抽样
categories 类别确信度 shape=(m, ..., n)
count 抽样数量
'''


def categories_sample(categories, count):
    # 转换成分布列
    prob_d = prob_distribution(categories)
    shape = prob_d.shape
    # 转换成分布
    prob_d = prob_d.reshape((-1, shape[-1]))
    m, n = prob_d.shape
    for i in range(1, n):
        prob_d[:, i] += prob_d[:, i - 1]

    prob_d[:, n - 1] = 1.0

    # 随机抽样
    res = np.zeros((m, count))
    for i in range(count):
        p = np.random.uniform(0, 1, (m,)).reshape((m, 1))
        item = (prob_d < p).astype(int)
        item = item.sum(axis=1)
        res[:, i] = item

    res = res.reshape(shape[0:len(shape) - 1] + (count,)).astype(int)
    return res


# 卷积操作
def convolution2d(X, W, stride):
    b, c, h, w = X.shape
    out_c, in_c, kh, kw = W.shape
    # 不使用分组卷积，c = in_c
    # 卷积核为正方形，kh = kw
    out_h, out_w = calculate_HW(h, w, 0, W.shape[-2:], stride)
    res = np.zeros((b, out_c, out_h, out_w))
    # 最终输出形状：b*out_c*(h-kh+1)*(w-kw+1)
    # print(len(res), len(res[0]), len(res[0][0]), len(res[0][0][0]))
    for i in range(b):
        # 关于batch，目前只能串行完成

        for j in range(out_c):
            # 计算每一组的结果

            for m in range(0, c):
                for n in range(0, h - kh + 1, stride[0]):
                    for g in range(0, w - kw + 1, stride[1]):
                        # 计算每一个位置的值
                        res[i, j, n, g] += np.sum(X[i, m, n:(kh + n), g:(kw + g)] * W[j, m])
    return res


# 计算输出矩阵的HW
def calculate_HW(H_in, W_in, padding, kernel_size, stride):
    H_out = (H_in + 2 * padding - 1 * (kernel_size[0] - 1) - 1) / stride[0] + 1
    W_out = (W_in + 2 * padding - 1 * (kernel_size[1] - 1) - 1) / stride[1] + 1
    return int(H_out), int(W_out)


# 用于stride不为一时backward计算,负责内部填充0。外部用padding填充
def stride_insert(src, stride):
    H, W = src.shape

    zero_row = np.zeros((stride[0] - 1, W))
    zero_col = np.zeros((H + (H - 1) * (stride[0] - 1), stride[1] - 1))
    for each in range(0, H - 1):
        sli = np.arange((each + each * (stride[0] - 1) + 1), (1 + each + each * (stride[0] - 1) + stride[0] - 1 - 1))
        src = np.insert(src, sli, axis=0, values=zero_row)
    for each in range(0, W - 1):
        sli = np.arange((each + each * (stride[0] - 1) + 1), (1 + each + each * (stride[0] - 1) + stride[0] - 1 - 1))
        src = np.insert(src, sli, axis=1, values=zero_col)

    return src


# 最大池化函数
def max_pooling(X, pool_size, stride=None):
    b, c, h, w = X.shape
    kh, kw = pool_size
    out_h, out_w = calculate_HW(h, w, 0, pool_size, stride)
    res = np.zeros((b, c, out_h, out_w))
    # 最终输出形状：b*c*out_h*out_w
    for i in range(b):
        # 关于batch，目前只能串行完成
        for m in range(0, c):
            for n in range(stride[0], h + 1, stride[0]):
                for g in range(stride[1], w + 1, stride[1]):
                    # 计算每一个位置的值
                    row = n-stride[0]
                    col = g-stride[1]
                    res[i, m, row//stride[0], col//stride[1]] = \
                        np.max(X[i, m, row:(kh + row), col:(kw + col)])
    return res


if __name__ == "__main__":
    print(calculate_HW(28, 28, 0, (2, 2), (2, 2)))
    # a = np.array([[1, -1, 2],
    #               [-1, 2, 3],
    #               [4, 1, -1]])
    # stride_insert(a, (3, 3))
