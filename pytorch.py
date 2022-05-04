import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd


class Conv_net(nn.Module):
    def __init__(self):
        super(Conv_net, self).__init__()
        self.conv = nn.Conv2d(2, 4, (3, 3), bias=False)  # 为了方便，不设置偏置

    def forward(self, x):
        x = self.conv(x)
        return x  # 为了方便，这里不做flatten以及全连接操作，直接对feature map各元素求和作为输出


criterion=nn.MSELoss()
convnet=Conv_net()

init_weight=torch.ones((4, 2, 3, 3), dtype=torch.float, requires_grad=True)

# 用上述的自定义初始权值替换默认的随机初始值
convnet.conv.weight.data = init_weight

input_=torch.arange(2*6*6*2, dtype=torch.float, requires_grad=True).view(2, 2, 6, 6)
target = torch.tensor(250.) # 随意设定的值
output = convnet(input_) # 网络输出

# 计算损失
loss=criterion(output,target)

# 打印看看结果
print("MSE Loss:", loss.item())


# 将之前buffer中的梯度清零
convnet.zero_grad()

# 反向传播计算梯度
# grad = autograd.grad(outputs=output, inputs=convnet.conv.weight, grad_outputs=torch.ones_like(output))
# print("weight", grad)
grad = autograd.grad(outputs=output, inputs=input_, grad_outputs=torch.ones_like(output))
print("input", grad)
# 打印看看卷积核各个参数的梯度情况
print('卷积核参数梯度为:', convnet.conv.weight.grad)

