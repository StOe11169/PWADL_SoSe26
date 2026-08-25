from optuna.trial import Trial
from argparse import Namespace

def build_config(trial: Trial, args: Namespace):
    cfg = {} 
    #values are share by al optuna trials
    #Static
    cfg["data"] = args.data
    cfg["epochs"] = args.epochs
    cfg["num_frames"] = args.num_frames

    #Audio
    cfg["audio_sample_rate"] = 16000
    cfg["num_audio_clips"] = 4
    cfg["audio_clip_seconds"] = 1.0
    cfg["audio_num_samples"] = int( cfg["audio_clip_seconds"] * cfg["audio_sample_rate"] ) #number of waveform samples from one clip
    cfg["audio_mono"] = True
    cfg["audio_normalize"] = True
    cfg["audio_missing_policy"] = "skip"
    cfg["audio_exclude_path_parts"] = getattr(args, "audio_exclude_path_parts", ["Mirror"]) #filepath without audio

    #Dataloader
    cfg["batch_size"] = trial.suggest_categorical("batch_size", [4,8]) #small batches to limit memory usage
    cfg["num_workers"] = 0

    #Model
    cfg["dropout"] = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)

    #Loss, weight calculated for each fold separatly
    cfg["class_weighting"] = "train_negative_to_positive_ratio"

    #Optimizer
    #hyperparams sampled independently for each trial
    cfg["optimizer"] = trial.suggest_categorical("optimizer", ["adamw", "sgd"])
    cfg["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    if cfg["optimizer"] == "sgd":
        cfg["momentum"] = trial.suggest_float("momentum", 0.0, 0.95)

    #LR Scheduler
    cfg["scheduler"] = trial.suggest_categorical("scheduler", ["none", "exponential", "step"])

    if cfg["scheduler"] == "exponential":
        cfg["gamma"] = trial.suggest_float("gamma", 0.85, 0.99)
    
    elif cfg["scheduler"] == "step":
        cfg["step_size"] = trial.suggest_int("step_size", 2, 10)
        cfg["gamma"] = trial.suggest_float("gamma", 0.1, 0.9)

    return cfg

