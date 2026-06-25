import os
import torch
import numpy as np
import random
import subprocess
import webbrowser
import time
from torch.utils.tensorboard import SummaryWriter


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


#Create Summary Writer for specific Study/Trial
def get_writer(study_dir, trial_number):
    log_dir = os.path.join(study_dir, f"tensorboard_trial_{trial_number}")
    return SummaryWriter(log_dir=log_dir)


def start_tensorboard(log_dir="logs", port=6006):
    """
    Starts TensorBoard in the background and opens it in the browser.
    """
    # Start TensorBoard process
    process = subprocess.Popen(
        ["tensorboard", "--logdir=",log_dir, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, shell=True
    )

    # Give it a moment to start
    time.sleep(2)

    # Open browser
    webbrowser.open(f"http://localhost:{port}")

    return process