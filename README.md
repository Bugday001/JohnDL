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

### Note

- 目前定义网络必须使用JohnDL.Unit()
    - 按照Uint中定义顺序初始化网格
    - 依据第一次正向传播的顺序设定反向传播的传播顺序
- 文件含有LeNet5示例。使用train.py训练，test.py测试(在test.py中修改model即可测试不同网络)
- Native_Network.py中的与逻辑(Andtest function)测试需要用户自己设定网络，并初始化设置。
- 使用时可能想要将JohnDL文件夹设为source root。

## Requirements
```requirements
matplotlib==3.2.1
numpy==1.18.3
tensorflow==2.9.1
tensorflow_gpu==2.2.0rc3
tqdm==4.64.0
```
**tensorflow只用来导入Minst数据集。若本地存有，可不使用。**

## 文件结构
- doc：文档及学习记录
- JohnDL：核心代码
- Models：模型目录
- 其余为实例及测试程序
