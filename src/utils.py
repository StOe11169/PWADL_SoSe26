import os
import torch
import numpy as np
import random


def setup_env(seed):
    """
    Setzt Zufallsseeds für reproduzierbarere Experimente.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.set_float32_matmul_precision("high")        