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
from src.models.audio.yamnet import YamNetAudioClassifier

from src.training import trainer
from src.evaluation import evaluate, predict_logits
from src.config import build_config
from src.utils import get_device, get_writer
from src.fusion import fuse_logits, get_fusion_metrics, get_contribution_summary, save_fusion_results

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
        return YamNetAudioClassifier(
            dropout=cfg["dropout"],
            sample_rate=cfg["audio_sample_rate"],
        ).to(device)

    else:
        raise ValueError(f"Unknown mode: {mode}")

def build_loaders(trainset, valset, cfg):
   #build train and val-dataloaders depending on mode
    trainloader = DataLoader(trainset,batch_size=cfg["batch_size"],num_workers=cfg["num_workers"],shuffle=True,drop_last=True,)

    valloader = DataLoader(valset,batch_size=cfg["batch_size"],num_workers=cfg["num_workers"],shuffle=False,)

    return trainloader, valloader

def prepare_dataframe_for_mode(df, args):
    #Applies mode-specific df filtering before cv
    if args.mode in ("audio", "multimodal"):
        df = filter_audio_dataframe(df,exclude_path_parts=args.audio_exclude_path_parts)
        if len(df) == 0:
            raise ValueError("Audio mode has zero usable samples after filtering. Fix audio conversion or provide dummy audio files.")
    return df.reset_index(drop=True)

def evaluate_multimodal(test_df, visual_model, audio_model, visual_cfg, audio_cfg, device, fold_dir, visual_weight=0.5):
    #runs both pipelines on the same video and fuses their logits output

    #build each mode with its dataset
    visual_testset = build_dataset(test_df, visual_cfg, "visual")
    audio_testset = build_dataset(test_df, audio_cfg, "audio")

    #build dataloaders, no shuffling so predictions logs stay deterministic
    audio_loader = DataLoader(audio_testset, batch_size=audio_cfg["batch_size"], shuffle=False)
    visual_loader = DataLoader(visual_testset, batch_size=visual_cfg["batch_size"], shuffle=False)

    #existing models stay independent
    audio_predictions = predict_logits(audio_loader, audio_model, device, input_key="audio")
    visual_predictions = predict_logits(visual_loader, visual_model, device, input_key="frames")

    #combine logits after each pipeline finishes
    fused = fuse_logits(visual_predictions, audio_predictions, visual_weight=visual_weight)

    #get metrics, summary and save them
    fusion_metrics = get_fusion_metrics(fused)
    contribution_summary = get_contribution_summary(fused, visual_weight=visual_weight)
    save_fusion_results(fused, fusion_metrics, contribution_summary, fold_dir)

    return fusion_metrics, contribution_summary

def objective(trial,train_df_outer,inner_splits,args, study_dir, mode):
    try:
        #Load config & print to console
        cfg = build_config(trial, args)
        input_key = get_input_key(mode)
        print(f"Trial {trial.number}")
        print(f"Mode: {mode}")
        for k, v in cfg.items():
            print(f"{k}: {v}")
        
        device = get_device()
        inner_f1_scores = []
        inner_best_epochs = []

        for inner_fold, (train_idx, val_idx) in enumerate(inner_splits):
            print(f"--- Inner fold {inner_fold} ---")

            train_df = train_df_outer.iloc[train_idx].reset_index(drop=True)
            val_df = train_df_outer.iloc[val_idx].reset_index(drop=True)

            #build datasets for this fold
            trainset = build_dataset(train_df, cfg, mode)
            valset = build_dataset(val_df, cfg, mode)

            trainloader, valloader = build_loaders(trainset, valset, cfg)

            #Fresh model for every fold
            model = build_model(cfg, mode, device)
  
            # start training
            f1_val, epoch = trainer(trainloader=trainloader, valloader=valloader, model=model, device=device, 
                                    trial_number= f"{trial.number}_inner_{inner_fold}", study_dir = study_dir, 
                                    cfg=cfg, trial = trial, input_key=input_key, pruning_step_offset = inner_fold * cfg["epochs"])

            inner_f1_scores.append(f1_val)
            inner_best_epochs.append(epoch)
        mean_f1 = float(np.mean(inner_f1_scores))
        
        #Save Trial summary
        trial_summary = {"trial_number": trial.number,"mean_f1_val": mean_f1, "fold_f1": inner_f1_scores, "best_epoch": inner_best_epochs, "params": cfg }
        with open(os.path.join(study_dir, f"trial_{trial.number}_summary.json"), "w") as f:
            json.dump(trial_summary, f, indent=4)

        return mean_f1
    
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

def run_inner_study(train_df_outer, inner_splits, args, study_dir, mode):
    #runs hyperparam optimization for one modality

    #keep visual/audio checkpoints and logs separate
    os.makedirs(study_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2, interval_steps=1))

    #each trial uses same, predefined inner cv folds so none "get lucky" with the splits
    study.optimize(lambda trial: objective(trial, train_df_outer, inner_splits, args, study_dir, mode), n_trials=args.n_trials, show_progress_bar=True)

    #ensure at least one trial finishes
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]

    if len(completed_trials) == 0:
        print(f"No completed {mode} trials")
        return None

    print(f"Best {mode} inner F1: {study.best_value:4f}")

    #read cfg of best trial
    best_trial = study.best_trial
    summary_path = os.path.join(study_dir,f"trial_{best_trial.number}_summary.json")

    with open(summary_path, "r") as f:
        best_summary = json.load(f)

    return best_summary["params"]

def train_final_model(final_train_df, final_val_df, cfg, mode, device, model_dir):
    #train one final unimodal model using its best config
    #The outer test set is deliberately not passed here

    input_key = get_input_key(mode)
    #build fresh model for final training
    model = build_model(cfg, mode, device)

    #build mode specific datasets
    trainset = build_dataset(final_train_df, cfg, mode)
    valset = build_dataset(final_val_df, cfg, mode)

    trainloader, valloader = build_loaders(trainset, valset, cfg)

    #select best epoch only using final validation data
    trainer(trainloader=trainloader, valloader=valloader, model=model, device=device, trial_number="final", study_dir=model_dir, cfg=cfg, input_key=input_key)

    #reload the best final checkpoint
    checkpoint_path = os.path.join(model_dir, "best_model_trial_final.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

def run_multimodal_experiment(df, args, study_dir):
    #run visual and audio pipelien independently, then fuse their logits
    device = get_device()

    #use same outer folds for both modes
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    outer_results = []

    #clean tensorboard for fusion results
    fusion_writer = get_writer(study_dir, "fusion")

    #default to equal weightng for logits
    visual_weight = getattr(args, "visual_weight", 0.5)

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(df, y=df["yawning"], groups=df["id"])):
        print(f"MULTIMODAL OUTER FOLD {fold}")

        #modes use exactly the same videos
        train_df_outer = df.iloc[train_idx].reset_index(drop=True)
        test_df_outer = df.iloc[test_idx].reset_index(drop=True)

        #keep logs/checkpoints eparated by mode
        fold_dir = os.path.join(study_dir, f"outer_fold_{fold}")
        visual_dir = os.path.join(fold_dir,"visual")
        audio_dir = os.path.join(fold_dir, "audio")
        fusion_dir = os.path.join(fold_dir, "fusion")

        os.makedirs(visual_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(fusion_dir, exist_ok=True)

        #create inner folds once -> visual and audio see same ids 
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

        inner_splits = list(inner_cv.split(train_df_outer, y=train_df_outer["yawning"], groups=train_df_outer["id"]))

        #visual hyperparams
        print("\n----- VISUAL OPTIMIZATION -----")
        visual_cfg = run_inner_study(train_df_outer, inner_splits, args, visual_dir, mode="visual")

        #audio hyperparams
        print("\n----- AUDIO OPTIMIZATION -----")

        audio_cfg = run_inner_study(train_df_outer, inner_splits, args, audio_dir, mode="audio")

        #skip fold if either modality failed completely
        if visual_cfg is None or audio_cfg is None:
            print("Missing completed modality study. Skipping fold")
            continue

        #make ONE final train/val split
        #both modalities must use the same videos
        final_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        final_train_idx, final_val_idx = next(final_cv.split(train_df_outer, y=train_df_outer["yawning"], groups=train_df_outer["id"]))

        final_train_df = train_df_outer.iloc[final_train_idx].reset_index(drop=True)

        final_val_df = train_df_outer.iloc[final_val_idx].reset_index(drop=True)

        # -------------------------Final visual model-------------------------
        print("\n----- FINAL VISUAL MODEL -----")
        visual_model = train_final_model( final_train_df, final_val_df, visual_cfg, mode="visual", device=device, model_dir=visual_dir)

        #-------------------------Final audio model-------------------------
        print("\n----- FINAL AUDIO MODEL -----")
        audio_model = train_final_model(final_train_df, final_val_df, audio_cfg, mode="audio", device=device, model_dir=audio_dir)


        # -------------------------Late fusion-------------------------
        print("\n----- LATE FUSION -----")
        fusion_metrics, contributions = evaluate_multimodal(test_df=test_df_outer, visual_model=visual_model, audio_model=audio_model, visual_cfg=visual_cfg, audio_cfg=audio_cfg, device=device, fold_dir=fusion_dir, visual_weight=visual_weight)

        print(f"Fold {fold} fused F1:{fusion_metrics['f1']:.4f}")
        print(f"Mean contribution share: visual={contributions["mean_visual_abs_share"]:.3f} audio={contributions["mean_audio_abs_share"]:.3f}")

        #sending only aggregate results to tensorboard
        fusion_writer.add_scalar("F1/fused", fusion_metrics["f1"], fold)

        fusion_writer.add_scalar("Contribution/visual_abs_share", contributions["mean_visual_abs_share"], fold)

        fusion_writer.add_scalar("Contribution/audio_abs_share", contributions["mean_audio_abs_share"], fold)
        outer_results.append(fusion_metrics["f1"])

    fusion_writer.close()
    #final multimodal CV result
    print("MULTIMODAL FINAL RESULTS ")
    print(f"Mean F1: {np.mean(outer_results):.4f}")
    print(f"Std F1:  {np.std(outer_results):.4f}")


def run_experiment(df, args, study_dir):
    """Dispatch the requested experiment mode.
    Visual/audio use one model and one input key.
    Multimodal coordinates both pipelines"""

    #filter data before creating cv folds -> identical samples for both modes
    df = prepare_dataframe_for_mode(df, args)

    mode = args.mode
    if mode == "multimodal":
        return run_multimodal_experiment(df, args, study_dir)

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

        #same inner folds vor every trial so one trial does not get a lucky split (->running multiple outer folds prevents having "one unlucky split")
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splits = list(inner_cv.split(train_df_outer, y=train_df_outer["yawning"], groups=train_df_outer["id"]))
    
        #Inner Loop
        study = optuna.create_study(direction="maximize", 
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, #dont prune immediatly
                                                                            n_warmup_steps=2, #wait one epoch
                                                                            interval_steps=1))
        study.optimize(lambda trial: objective(trial, train_df_outer,inner_splits, args, fold_dir, mode), n_trials=args.n_trials, show_progress_bar=True)
    
        #Skip if trial pruned to early
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) == 0:
            print("No completed trials in this fold. Skipping...")
            continue  # skip this outer fold entirely
    
        print(f"Bestial F1 (inner): {study.best_value:4f}")
        best_trial = study.best_trial
        best_summary_path = os.path.join(fold_dir,f"trial_{best_trial.number}_summary.json",)

        with open(best_summary_path, "r") as f:
            best_summary = json.load(f)

        best_cfg = best_summary["params"]
        device = get_device()
    
        #Final training after hyperparam selection
        model = build_model(best_cfg, mode, device)
       
        #Final val split only from outer training data
        final_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        final_train_idx, final_val_idx, = next(final_sgkf.split(train_df_outer, y=train_df_outer["yawning"], groups=train_df_outer["id"]))

        final_train_df = train_df_outer.iloc[final_train_idx].reset_index(drop=True)
        final_val_df = train_df_outer.iloc[final_val_idx].reset_index(drop=True)

        #Build train/val datasets
        final_trainset = build_dataset(final_train_df, best_cfg, mode)
        final_valset = build_dataset(final_val_df, best_cfg, mode)

        final_trainloader, final_valloader = build_loaders(final_trainset, final_valset, best_cfg)

        #Train without touching outer test data
        trainer(trainloader=final_trainloader, valloader=final_valloader, model=model, device=device, trial_number="final", study_dir=fold_dir, cfg=best_cfg, input_key=input_key)

        #Load final model selected on final_val_df
        final_model_path = os.path.join(fold_dir, "best_model_trial_final.pth")
        final_checkpoint = torch.load(final_model_path, map_location="cpu")

        #Evaluate outer
        model.load_state_dict(final_checkpoint["model_state_dict"])

        #Create datasets for final outer test
        testset = build_dataset(test_df_outer, best_cfg, mode)
        testloader = DataLoader(testset, batch_size=best_cfg["batch_size"], num_workers=best_cfg["num_workers"], shuffle=False)

        model.eval()
        test_metrics = evaluate(testloader, model, device, input_key=input_key)
    
        print(f"Fold {fold} F1: {test_metrics['f1']:.4f}")
        outer_results.append(test_metrics["f1"])
    
    #print final results
    print("\n================ FINAL RESULTS ================")
    print(f"Mean F1: {np.mean(outer_results):.4f}")
    print(f"Std F1:  {np.std(outer_results):.4f}")
    