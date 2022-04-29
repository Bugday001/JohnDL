# John's DeepLearning wheel

## 2022.04.29
今天继续学习卷积。由于卷积内容较多，重开一个新的Markdown记录。

修复`model.eval()`的Bug。支持预测后重新设置为训练模式。

实现基础的`padding`利用numpy的pad。

实现了固定学习率优化器，将更新操作移除了Model类。

又学了一天卷积，虽然Conv1d看起来简单，但一般用在NLP问题上。(似乎tf和pytorch实现有所不同(方向)，~~看的我迷糊~~)
暂时决定直接实现在图像上常用的2d，可能可以利用numpy的convolve函数。

已利用for循环基本实现卷积，但对于偏置Bias的处理，发现Pytorch似乎有些不同，但Functional源码无法直接看到。
目前默认偏置直接减去。

### 待解决
卷积的反向传播，和优化卷积计算(现在的循环太多了，估计极慢)

## 2022.04.28
创建Layer类，作为所有层和激活函数的父类。

暂时想到解决办法：
- 创建一个list依次append各个Layer进去，包括激活函数
- 新增两个量，指向“父节点”和”子节点“

已实现自动BP和更新参数(利用方法一，创建了一个dict，保存所有线性层和线性层加激活函数)。
Model类作为Layer和激活函数的父类。
```txt
# 后向传播
model.backward(criterion.gradient)
# 调整参数
model.renew_params(lr)
```

尝试优化DataLoader，目前数据结构还有一定问题，每次训练前还要reshape浪费资源。

已实现，之前想到了迭代器，但是忘记了yield。参考网上的博客，用了yield，十分方便。

尝试添加Dropout。参考Pytorch已实现Dropout，但发现pytorch中都是函数+类的搭配。
原因未知。在此还没有此需求，暂不模仿。

### 待解决
GUP加速，卷积层。
加速似乎要用到CuPy，一个类numpy的包？将其计划延后，先解决卷积的问题。

### 卷积看起来有点复杂
关于Conv1d和conv2d：2d先横着扫再竖着扫，1d只能竖着扫？

参考连接：
- [卷积层的反向传播](https://blog.csdn.net/weixin_37721058/article/details/102327691)
- [Python实现卷积神经网络](https://blog.csdn.net/weixin_37251044/article/details/81349287)
- [Conv1D和Conv2D的区别](https://zhuanlan.zhihu.com/p/156825903)
- [卷积操作的初始化方法](https://blog.csdn.net/weixin_44503976/article/details/117284487)

## 2022.04.27
实现实现逻辑与，两层隐藏，全连接，ReLU，Sigmoid。

实现保存模型，及调用模型预测。

尝试手写数字识别问题，似乎网络中值过大，
Softmax，表现不好。

Softmax已解决，原来在交叉熵`CategoricalCrossEntropy`里就已经化为概率了。
如果再加上Softmax效果反而不好。参考Pytorch，发现似乎也有类似的提示。
还是怪自己学习交叉熵和激活函数时没有融汇贯通，浮于表面。

***Too young, too simple, sometimes naive.***

手写数字Acc可以达到0.5左右(一层隐藏层,500个epochs, batch=64)，尝试学习优化器。
使用三层全连接可以近乎达到0.9(20个epochs, batch=64)。

尝试自动调用BP，用户只需要写forward

已将loss移除models。

### 待解决
将Layer统一一个父类，以注册，自动修正参数。

将network class中的backward函数去除，自动进行。利用loss.backward()?

## 2022.04.26
创建工程，构建全连接。
激活函数ReLu, Softmax(还存在问题)

实现了简单Dataloader