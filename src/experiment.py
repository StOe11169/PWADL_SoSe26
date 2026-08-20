import os
import json
import numpy as np
import torch
import optuna
import glob

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

from src.data import YawDDDataset
from src.data_audio import AudioYawDDDataset
from src.utils_audio import filter_audio_dataframe

from src.models.visual.model import YawDDclassifier
from src.models.audio.yamnet import YamNetLikeAudioClassifier

from src.training import trainer
from src.evaluation import evaluate
from src.config import build_config
from src.utils import get_device

def get_input_key(mode):
    #return key used by trainer/evaluate
    if mode == "visual":
        return "frames"

    if mode == "audio":
        return "audio"

    else:
        raise ValueError(f"Unkown mode: {mode}")

def build_dataset(df, cfg, mode):
    #build dataset depending on mode
    if mode == "visual":
        return YawDDDataset(df, num_frames=cfg["num_frames"])

    if mode == "audio":
        return AudioYawDDDataset(df, cfg)

    else:
            raise ValueError(f"Unkown mode: {mode}")

def build_model(cfg, mode, device):
    #Build model depending on mode
    
    if mode == "visual":
        return YawDDclassifier(cfg["dropout"]).to(device)

    if mode == "audio":
        return YamNetLikeAudioClassifier(
            dropout=cfg["dropout"],
            sample_rate=cfg["audio_sample_rate"],
        ).to(device)

    else:
        raise ValueError(f"Unknown mode: {mode}")

def build_loaders(trainset, valset, cfg):
   #build dataloaders depending on mode
    trainloader = DataLoader(trainset,batch_size=cfg["batch_size"],num_workers=cfg["num_workers"],shuffle=True,drop_last=True,)

    valloader = DataLoader(valset,batch_size=cfg["batch_size"],num_workers=cfg["num_workers"],shuffle=False,)

    return trainloader, valloader

def prepare_dataframe_for_mode(df, args):
    #Applies mode-specific df filtering before cv
    if args.mode == "audio":
        df = filter_audio_dataframe(
            df,
            exclude_path_parts=args.audio_exclude_path_parts,
        )

        if len(df) == 0:
            raise ValueError(
                "Audio mode has zero usable samples after filtering. "
                "Fix audio conversion or provide dummy audio files."
            )

    return df.reset_index(drop=True)

def objective(trial,train_df_outer,args, study_dir):
    try:
        #Load config & print to console
        cfg = build_config(trial, args)
        mode = args.mode
        input_key = get_input_key(mode)
        print(f"Trial {trial.number}")
        print(f"Mode: {mode}")
        for k, v in cfg.items():
            print(f"{k}: {v}")
        
        device = get_device()

        #Split for inner nested cv loop
        gss = GroupShuffleSplit(n_splits=3, test_size=0.15, random_state=trial.number)
        train_idx, val_idx = next(gss.split(train_df_outer, y=train_df_outer["yawning"], groups=train_df_outer["id"]))
        train_df = train_df_outer.iloc[train_idx].reset_index(drop=True)
        val_df   = train_df_outer.iloc[val_idx].reset_index(drop=True)

        # data preparation
        trainset = build_dataset(train_df, cfg, mode)
        valset = build_dataset(val_df, cfg, mode)

        # dataloaders
        trainloader,valloader = build_loaders(trainset, valset, cfg)
         

        model = build_model(cfg, mode, device)
  
        # start training
        f1_val, epoch = trainer(trainloader=trainloader, valloader=valloader, model=model, device=device, trial_number= trial.number, study_dir = study_dir, cfg=cfg, trial = trial, input_key=input_key)
        
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

def run_experiment(df, args, study_dir):
    mode = args.mode
    input_key = get_input_key(mode)
        
    #Outer Loop
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    outer_results = []
    
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
        device = get_device()
    
        #Training on outer Train set
        model = build_model(best_cfg, mode, device)
        trainset = build_dataset(train_df_outer, best_cfg, mode)
        testset  = build_dataset(test_df_outer, best_cfg, mode)
    
        trainloader = DataLoader(trainset, batch_size=best_cfg["batch_size"], shuffle=True, drop_last=True)
        testloader  = DataLoader(testset,  batch_size=best_cfg["batch_size"], shuffle=False)
    
        trainer(trainloader=trainloader, valloader=testloader, model=model, device=device, trial_number="final", study_dir=fold_dir, cfg=best_cfg, input_key=input_key)

        # Evaluate best final model, not inner-CV model
        final_model_path = os.path.join(fold_dir, "best_model_trial_final.pth")
        final_checkpoint = torch.load(final_model_path, map_location="cpu")

        #Evaluate outer
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        test_metrics = evaluate(testloader, model, device, input_key=input_key)
    
        print(f"Fold {fold} F1: {test_metrics['f1']:.4f}")
        outer_results.append(test_metrics["f1"])
    
    #print final results
    print("\n================ FINAL RESULTS ================")
    print(f"Mean F1: {np.mean(outer_results):.4f}")
    print(f"Std F1:  {np.std(outer_results):.4f}")
    