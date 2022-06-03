# John's DeepLearning wheel

**A light-weight deep learning "framework"**

## JohnDL
一个用作学习的深度学习"框架",姑且这样称呼他。

目前已有支持：
- 全连接，卷积(Conv2D)
- Maxpooling, Avgpooling, BatchNormalization, Dropout
- Softmax, tanh, Relu, Sigmoid
- 交叉损失熵，均方误差损失

示例可参见`Native_Network.py`和`LeNet.py`

**Note**

- 目前定义网络必须使用JohnDL.Unit()
- 文件含有LeNet5示例。使用train.py训练，test.py测试
- Native_Network.py中的与逻辑(Andtest function)测试需要用户自己设定网络，并初始化设置。

## 文件结构
- doc：文档及学习记录
- JohnDL：核心代码
- 其余为测试用程序
