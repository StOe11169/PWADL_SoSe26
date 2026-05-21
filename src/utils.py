import os
import torch
import numpy as np
import random


def setup_env(seed):

    # set python, numpy, torch random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # when running on the CuDNN backend
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # set precision
    torch.set_float32_matmul_precision('high')    