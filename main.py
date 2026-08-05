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
from src.utils import setup_env, get_writer, start_tensorboard
from src.data import YawDDDataset, get_all_data_paths, create_group_splits
from src.training import trainer
from src.model import YawDDclassifier
from src.evaluation import evaluate
from src.config import build_config

#Create unique folder for study
study_name = datetime.now().strftime("%Y%m%d_%H%M%S")
study_dir = os.path.join("logs", f"study_{study_name}")

os.makedirs(study_dir, exist_ok=True)

#Load dataset
df = get_all_data_paths("data")

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

    #Outer Loop
    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    outer_results = []
    
    # Create & run study, maximizing validation F1, lamba as extra study_dir arg isnt passed directly by optuna
    #study = optuna.create_study(direction="maximize")
    #Start Tensorboard
    tb_process = start_tensorboard(study_dir)

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(df, y=df["yawning"], groups=df["id"])):
        print(f"\n================ OUTER FOLD {fold} ================")

        train_df_outer = df.iloc[train_idx].reset_index(drop=True)
        test_df_outer = df.iloc[test_idx].reset_index(drop=True)

        #Fold specific directory for logging
        fold_dir = os.path.join(study_dir, f"outer_fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        #Inner Loop
        study = optuna.create_study(direction="maximize", 
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=2, #dont prune immediatly
                                                                        n_warmup_steps=2, #wait one epoch
                                                                        interval_steps=1))
        study.optimize(lambda trial: objective(trial, train_df_outer, args, fold_dir), n_trials=args.n_trials, show_progress_bar=True)

        #Skip if trial pruned to early
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) == 0:
            print("No completed trials in this fold. Skipping...")
            continue  # skip this outer fold entirely

        print(f"Bestial F1 (inner): {study.best_value:4f}")
        best_trial = study.best_trial
        best_model_path = os.path.join(fold_dir, f"best_model_trial_{best_trial.number}.pth")

        checkpoint = torch.load(best_model_path, map_location="cpu")
        best_cfg = checkpoint["cfg"]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Training on outer Train set
        model = YawDDclassifier(best_cfg["dropout"]).to(device)
        trainset = YawDDDataset(train_df_outer, num_frames=best_cfg["num_frames"])
        testset  = YawDDDataset(test_df_outer,  num_frames=best_cfg["num_frames"])

        trainloader = DataLoader(trainset, batch_size=best_cfg["batch_size"], shuffle=True, drop_last=True)
        testloader  = DataLoader(testset,  batch_size=best_cfg["batch_size"], shuffle=False)

        trainer(trainloader=trainloader, valloader=testloader, model=model, device=device, trial_number="final", study_dir=fold_dir, cfg=best_cfg)

        #Evaluate outer
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        test_metrics = evaluate(testloader, model, device)

        print(f"Fold {fold} F1: {test_metrics['f1']:.4f}")

        outer_results.append(test_metrics["f1"])

    #print final results
    print("\n================ FINAL RESULTS ================")
    print(f"Mean F1: {np.mean(outer_results):.4f}")
    print(f"Std F1:  {np.std(outer_results):.4f}")

    time_passed = time.time() - start_timestamp
    print(f'\nTraining finished in {time_passed//3600}h {(time_passed%3600)//60}min {time_passed%60:.0f}s\n')

    tb_process.terminate()
    