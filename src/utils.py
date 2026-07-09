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

# Start TensorBoard process and open in browser

def start_tensorboard(study_dir, port=6006):

   log_dir = os.path.join(os.getcwd(), study_dir)
   cmd = ["tensorboard", "--logdir", log_dir, "--port", str(port)]
   print(f"Executing: {' '.join(cmd)}")

   #Call tensorboard via cmd
   process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
   # Give it a moment to start
   time.sleep(5)

    # Open browser
   webbrowser.open(f"http://localhost:{port}")

   return process

