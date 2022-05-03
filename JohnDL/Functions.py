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


# 卷积操作，费时的卷积
def convolution2d(X, W, stride):
    b, c, h, w = X.shape
    out_c, in_c, kh, kw = W.shape
    # 不使用分组卷积，c = in_c
    # 卷积核为正方形，kh = kw
    out_h, out_w = calculate_HW(h, w, (0, 0), W.shape[-2:], stride)
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
                        res[i, j, n // stride[0], g // stride[1]] += np.sum(X[i, m, n:(kh + n), g:(kw + g)] * W[j, m])
    return res


# 优化卷积
def conv_tensor(X, W, stride):
    kernel_size = W.shape[-2:]
    A = split_by_strides(X, kernel_size, stride)
    # A: (N,C,OH,OW,KH,KW)
    # Kernel: (KN, C, KH, KW)
    # OUT: (N，oh，ow，kn)，期望(N, kn, oh,ow)

    np_ans = np.tensordot(A, W, [(4, 5, 1), (2, 3, 1)])
    # reshape to (N, kn, oh,ow)
    return np_ans.transpose((0, 3, 1, 2))


# 计算输出矩阵的HW
def calculate_HW(H_in, W_in, padding, kernel_size, stride):
    H_out = (H_in + 2 * padding[0] - 1 * (kernel_size[0] - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - 1 * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return H_out, W_out


# 反向传播时计算padding大小
def calculate_pad(H_in, W_in, H_out, W_out, kernel_size, stride):
    padding = [0, 0]
    padding[0] = ((H_out - 1) * stride[0] + kernel_size[0] - H_in) // 2
    padding[1] = ((W_out - 1) * stride[1] + kernel_size[1] - W_in) // 2
    return padding


# 用于stride不为一时backward计算,负责内部填充0。外部用padding填充
def stride_insert(src, stride):
    b, c, H, W = src.shape
    H_out = H + (H - 1) * (stride[0] - 1)
    W_out = W + (W - 1) * (stride[1] - 1)
    res = np.zeros((b, c, H_out, W_out))
    for each_b in range(b):
        for each_c in range(c):
            res[each_b, each_c, ::stride[0], ::stride[1]] = src[each_b, each_c]
    return res


# 最大池化函数
def max_pooling(X, pool_size, stride=None):
    N, C, H, W = X.shape
    kh, kw = pool_size
    out_h, out_w = calculate_HW(H, W, (0, 0), pool_size, stride)
    res = np.zeros((N, C, out_h, out_w))
    # 最终输出形状：b*c*out_h*out_w
    for i in range(N):
        # 关于batch，目前只能串行完成
        for m in range(0, C):
            for n in range(stride[0], H + 1, stride[0]):
                for g in range(stride[1], W + 1, stride[1]):
                    # 计算每一个位置的值
                    row = n - stride[0]
                    col = g - stride[1]
                    res[i, m, row // stride[0], col // stride[1]] = \
                        np.max(X[i, m, row:(kh + row), col:(kw + col)])
    return res


# 将数据划分点乘代替卷积
def split_by_strides(X, kernel_size, s):
    """
    X:源数据
    kernel_size:tuple,like (3, 3)
    s: stride, tuple,like (1, 2). 1 for row ,2 for col
    """
    N, C, H, W = X.shape
    kh, kw = kernel_size
    out_h, out_w = calculate_HW(H, W, (0, 0), kernel_size, s)
    strides = (*X.strides[:-2], X.strides[-2] * s[0], X.strides[-1] * s[1], *X.strides[-2:])
    A = np.lib.stride_tricks.as_strided(X, shape=(N, C, out_h, out_w, kh, kw), strides=strides)
    return A


if __name__ == "__main__":
    # H_in, W_in = (5, 5)
    # kernel_size = (3, 3)
    # stride = (2, 2)
    # H_out, W_out = calculate_HW(H_in, W_in, (0, 0), kernel_size, stride)
    # print(H_out, W_out)
    # H_out = H_out + (H_out - 1) * (stride[0] - 1)
    # W_out = W_out + (W_out - 1) * (stride[1] - 1)
    # print(H_out, W_out)
    # padding = calculate_pad(H_out, W_out, H_in, W_in, kernel_size, (1, 1))
    # print(padding)
    # print(calculate_HW(H_out, W_out, padding, kernel_size, (1, 1)))
    #
    a = np.array([[[[1, -1, 2],
                    [-1, 2, 3],
                    [4, 1, -1]]]])
    print(split_by_strides(a, (2, 2), (1, 1)))
