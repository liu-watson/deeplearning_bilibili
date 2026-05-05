"""
案例：
    绘制激活函数SoftMax的函数图像 和 导数图像

Sigmoid激活函数介绍：
    激活函数的目的：
        给模型增加非线性功能，让模型（神经元）既可以做分类，还可以做回归问题。
    激活函数的分类：
        Sigmoid:
        ReLU:
        Tanh:
        Softmax:

    Sigmoid激活函数：
        主要用于 二分类的输出层，且适用于 浅层神经网络（不超过5层）
        数据在[-6, 6]之间有效果，在[-3, 3]之间效果明显，会将数据值映射到：[0, 1]
        求导范围在[0, 0.25]

    Tanh:
        主要应用于 隐藏层，且适用于 浅层神经网络（不超过5层）
        数据在[-3, 3]之间有效果，在[-1, 1]之间效果明显，会将数据值映射到：[-1, 1]
        求导后范围在[0, 1]，较之于Sigmoid，收敛速度快。

    ReLU:
        计算公式为：max(0, x)，计算量相对较小，训练成本低，多应用于 隐藏层，且适合 深层神经网络。
        求导后，值要么是0，要么是1，较之于Tanh，收敛速度更快。
        默认情况下ReLU只考虑正样本，可以使用Leaky ReLU,PReLU 来考虑 正负样本。

    SoftMax:
        将多分类的结果以概率的形式展示，且概率和相加为1，最终选取概率值最大的分类 作为最终结果

"""
# 导包
import torch
import matplotlib.pyplot as plt

# 中文显示设置（macOS专用）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "SimHei", "Arial Unicode MS"] # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False                                      # 用来正常显示负号

# 1.定义张量，记录分类数据
scores = torch.tensor([0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75])

# 2. dim = 0，按行计算
probablities = torch.softmax(scores, dim=0)
print(probablities)