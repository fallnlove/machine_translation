import os
import random

import numpy as np
import torch


def set_random_seed(seed: int):
    """
    Set random seed for model training.

    Input:
        seed (int): seed.
    """

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    np.random.seed(seed)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
