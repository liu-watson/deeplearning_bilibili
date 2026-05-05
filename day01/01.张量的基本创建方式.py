"""
    案例：演示张量的基本创建方式
"""

# 导包
import torch
import numpy as np

def dm01():
    # 场景1:标量 张量
    t1 = torch.tensor(10)
    print(f"t1:{t1}, type:{type(t1)}")
    print("*" * 30)

    # 场景2:二维列表->张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.tensor(data)
    print(f"t2:{t2}, type:{type(t2)}")
    print("*" * 30)

    # 场景3:numpy np数组-> 张量
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.tensor(data, dtype=torch.float)
    print(f"t3:{t3}, type:{type(t3)}")
    print("*" * 30)

    # 场景4:尝试直接创建 指定维度（例如：2行3列的）张量
    # t4 = torch.tensor(2,3);
    # print(f"t4:{t4}, type:{type(t4)}")


def dm02():
    # 场景1:标量 张量
    t1 = torch.Tensor(10)
    print(f"t1:{t1}, type:{type(t1)}")
    print("*" * 30)

    # 场景2:二维列表->张量
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.Tensor(data)
    print(f"t2:{t2}, type:{type(t2)}")
    print("*" * 30)

    # 场景3:numpy np数组-> 张量
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.Tensor(data)
    print(f"t3:{t3}, type:{type(t3)}")
    print("*" * 30)

    # 场景4:尝试直接创建 指定维度（例如：2行3列的）张量
    t4 = torch.Tensor(2, 3)
    print(f"t4:{t4}, type:{type(t4)}")


def dm03():
    # 场景1:标量 张量
    t1 = torch.IntTensor(10)
    print(f"t1:{t1}, type:{type(t1)}")
    print("*" * 30)

    # 场景2:二维列表->张量
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.IntTensor(data)
    print(f"t2:{t2}, type:{type(t2)}")
    print("*" * 30)

    # 场景3:numpy np数组-> 张量
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.IntTensor(data)
    print(f"t3:{t3}, type:{type(t3)}")
    print("*" * 30)

    # 场景4:如果类型不匹配，会尝试自动转换类型
    data = np.random.randint(0, 10, size=(2, 3))
    t4 = torch.FloatTensor(data)
    print(f"t4:{t4}, type:{type(t4)}")
    print("*" * 30)

if __name__ == '__main__':
    dm03()