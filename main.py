import os
import numpy as np
import glob
from datetime import datetime
import argparse, time
import torch
from torch.utils.data import DataLoader
import optuna
import json
import shutil
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from src.utils import setup_env, start_tensorboard
from src.data import YawDDDataset, get_all_data_paths
from src.training import trainer
from src.model import YawDDclassifier
from src.evaluation import evaluate
from src.config import build_config
from src.experiment import run_experiment


"""
def objective(trial,train_df_outer,args, study_dir):
    try:

        #Load config & print to console
        cfg = build_config(trial, args)
        print(f"Trial {trial.number}")
        for k, v in cfg.items():
            print(f"{k}: {v}")
        
        # get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Split for inner nested cv loop
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=trial.number)
        train_idx, val_idx = next(gss.split(train_df_outer, y=train_df_outer["yawning"], groups=train_df_outer["id"]))
        train_df = train_df_outer.iloc[train_idx].reset_index(drop=True)
        val_df   = train_df_outer.iloc[val_idx].reset_index(drop=True)

        # data preparation
        trainset = YawDDDataset(train_df, num_frames=cfg["num_frames"])
        valset = YawDDDataset(val_df, num_frames=cfg["num_frames"])
        #testset = YawDDDataset(test_df, num_frames=cfg["num_frames"])

        # dataloaders
        trainloader = DataLoader(trainset, batch_size=cfg["batch_size"], num_workers=0, shuffle=True, drop_last=True)
        valloader = DataLoader(valset, batch_size=cfg["batch_size"], num_workers=0, shuffle=False)
        #testloader = DataLoader(testset, batch_size=cfg["batch_size"], num_workers=0, shuffle=False)

        # model
        model = YawDDclassifier(cfg["dropout"]).to(device)
  
        # start training
        f1_val, epoch = trainer(trainloader=trainloader, valloader=valloader, model=model, device=device, trial_number= trial.number, study_dir = study_dir, cfg=cfg, trial = trial)
        
        
        #Save Trial summary
        trial_summary = {"trial_number": trial.number, "f1_val": f1_val, "best_epoch": epoch, "params": cfg }
        with open(os.path.join(study_dir, f"trial_{trial.number}_summary.json"), "w") as f:
            json.dump(trial_summary, f, indent=4)

        return f1_val
    
    
    except Exception as e:
        #Remove incomplete files
        trial_files = glob.glob(os.path.join(study_dir, f"*trial_{trial.number}*"))
        for f in trial_files:
            try:
                os.remove(f)
            except:
                pass

        print(f"[Trial {trial.number}] Failed: {e}")
        raise e
"""

if __name__ == "__main__":
    # get start time
    start_timestamp = time.time()

    # set seed and precision
    setup_env(seed=0)    

    # get client args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='YawDD')
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=1)
    args = parser.parse_args()

    #Load dataset
    df = get_all_data_paths("data")

    #Create unique folder for study
    study_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = os.path.join("logs","vision", f"study_{study_name}") #replace vision with audio or multimodal later
    os.makedirs(study_dir, exist_ok=True)

    tb_process = start_tensorboard(study_dir)

    run_experiment(df, args, study_dir)
    
    time_passed = time.time() - start_timestamp
    print(f'\nTraining finished in {time_passed//3600}h {(time_passed%3600)//60}min {time_passed%60:.0f}s\n')

    tb_process.terminate()
    