# John's DeepLearning wheel

## 2022.05.09
感觉这个项目差不多就这样了。学习的目的已经达到了。

直接定义各种运算的梯度求法。自定义数据类型。

## 2022.05.08
实现LeNet5。同时添加了一个简单的可视化接口。

## 2022.05.05
&emsp;&emsp;尝试调整一下整个架构，加入一些新的属性和用法。但由于函数和用法是模仿Pytorch，
而我自己的思想似乎还停留在静态图(个人认为)，导致在添加一些新的属性和用法时出现了来自底层的冲突。

- 调整网络结构，同时学习`BatchNormalization`。
- 已实现`BatchNormalization`
    ——[BN](https://zhuanlan.zhihu.com/p/504159192)

### 待完善
Uint和普通Layer交替使用问题

## 2022.05.04

- 发现问题，损失函数的loss使NAN导致参数无法更新。
- 解决了一些Bug，怀疑加入卷积后其实并非没用学习，acc没用上涨，只是由于计算太慢，
    和超参数,网络的设置导致上升较为缓慢。
- 尝试加入其他优化器，看是否能够抑制此现象。
- 通过计时发现是Maxpooling的backward花费了大量时间(一大半)
- 使用AvgPooling后速度有巨大提升。MaxPooling中使用了不少循环，准备用numpy的
    函数替换。
- MaxPooling已优化。[Pool](https://zhuanlan.zhihu.com/p/70713747)
    
## 2022.05.03
- Stride和padding已经可以使用，但输入要求较为严格。除了要输入规定的数据类型，
    还要调整Stride,padding,kernel_size大小使卷积可以覆盖到每一个元素。

- 看到几个卷积计算优化的方法，打算试试np.as_strided。
- 优化卷积和反向传播梯度处理的程序后，速度有明显提升。

## 2022.05.01
继续完善卷积层。
- 接口功能实现,padding, stride。(基础已实现，但有一些Bug，和需要完善的接口)
- MaxPooling以及加入，经过了测试。但是训练的更慢了。
- 有很多善后的事。Stride, padding ,Maxpooling，以及它们的加速。看了几篇关于用numpy什么的
    加速卷积的博客。难顶。
    
## 2022.04.30
假期开始了。

- 利用numpy，优化卷积计算
- 统一由于添加卷积变动的父类接口
- 学习卷积的反向传播
- 测试反向传播中修复了之前父类子类函数名引发的问题，同时似乎要有一个拉平层
    负责卷积和线性层连接的反向传播(reshape)。
- 优化了进度条显示。测试卷积时内存占用巨大，反而CPU占用不高了。
- 减小了batch_size(32)，还是占用内存。训练极慢。
- 卷积测试：acc到还行看起来。训练了2个epoch， 测试集acc在90%以上。
    (但是居然花费了近20分钟)。模型大小200MB+，占用了C盘大量虚拟内存。
    急需优化。模型如下(交叉熵损失)
    ```txt
      self.conv2d = Conv2D(1, 2, kernel_size=(3, 3))
      self.flatten = Flatten((2, 26, 26), (2 * 26 * 26,))
      self.relu1 = Relu()
      self.linear2 = Linear(26 * 26 * 2, 128)
      self.dropout = Dropout(0.1)
      self.relu2 = Relu()
      self.linear3 = Linear(128, class_dim)
    ```

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