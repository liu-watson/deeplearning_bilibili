"""
案例：
    演示池化层相关操作

池化层解释(Pooling):
    目的：
        降维
    思路：
        最大池化
        平均池化
    特点：
        池化不会改变数据的 通道数
"""

# 导包
import torch
import torch.nn as nn

# 1.定义函数，演示单通道池化
def dm01():
    # 1.创建 1通道 3*3的二维矩阵
    inputs = torch.tensor([[
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ]])

    # print(f"inputs:{inputs}, shape:{inputs.shape}") # [1, 3, 3]

    # 2.创建最大池化层
    # 参1:池化核（池化窗口）大小，参2:步长，参3:填充
    pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    outputs = pool1(inputs)
    print(f"outputs:{outputs}, shape:{outputs.shape}") # [1, 2, 2]

    # 3.创建平均池化层
    pool2 = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    outputs = pool2(inputs)
    print(f"outputs:{outputs}, shape:{outputs.shape}") # [1, 2, 2]


# 2.定义函数，演示多通道池化
def dm02():
    # 1.创建 3通道 3*3的二维矩阵
    inputs = torch.tensor([
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ],
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ],
        [
            [11, 22, 33],
            [44, 55, 66],
            [77, 88, 99]
        ]
    ])

    # print(f"inputs:{inputs}, shape:{inputs.shape}") # [3, 3, 3]

    # 2.创建最大池化层
    # 参1:池化核（池化窗口）大小，参2:步长，参3:填充
    pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    outputs = pool1(inputs)
    print(f"outputs:{outputs}, shape:{outputs.shape}") # [3, 2, 2]

    # 3.创建平均池化层
    pool2 = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    outputs = pool2(inputs)
    print(f"outputs:{outputs}, shape:{outputs.shape}") # [3, 2, 2]

# 3.测试
if __name__ == '__main__':
    # dm01()
    dm02()