import random
import numpy as np
import torch


def set_random_seed(seed):
    """ 设置随机种子 """
    # 设置 Python 内置随机种子
    random.seed(seed)

    # 设置 NumPy 随机种子
    np.random.seed(seed)

    # 设置 PyTorch 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确保 PyTorch 计算可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
