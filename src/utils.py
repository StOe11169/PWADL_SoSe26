import os
import torch
import numpy as np
import random
import subprocess
import torch.optim as optim
import webbrowser
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

def setup_env(seed):

    #set python, numpy, torch random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    #when running on the CuDNN backend prefer deterministic kernels for reproducability
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # set precision
    torch.set_float32_matmul_precision('high') #allows high-throughput matrix multiplication


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
   process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
   # Give it a moment to start
   time.sleep(5)
   if process.poll() is not None:
        raise RuntimeError(f"TensorBoard failed to start on port {port}. Check whether another process already uses that port.")

    # Open browser
   webbrowser.open(f"http://localhost:{port}")

   return process

def  plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    #plots raw counts instead of normalized percentages
    cm = confusion_matrix(y_true, y_pred) 
    fig = plt.figure()
    sns.heatmap(cm, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    return fig

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (torch.cuda.get_device_name(0) if device.type == "cuda"else "CPU")
    print(f"Device: {device} ({device_name})")
    return device

def build_optimizer(model, cfg):
    # Get trainable parameters and hand to optimizer
    tp = [p for p in model.parameters() if p.requires_grad]
    
    #Get optimizer from cfg
    opt_name = cfg["optimizer"]
    
    if opt_name == "adamw":
        return optim.AdamW(tp, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    
    elif opt_name == "sgd":
        return optim.SGD(tp, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    
    else:
        raise ValueError(f"Unkown optimizer: {opt_name}")

def build_scheduler(optimizer, cfg):
    #Get LR Scheduler from cfg
    sched_name = cfg["scheduler"]

    if sched_name == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg["gamma"])

    elif sched_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"])

    elif sched_name == "none":
        return None
    
    else:
        raise ValueError(f"Unkown scheduler: {sched_name}")