
from optuna.trial import Trial
from argparse import Namespace

def build_config(trial: Trial, args: Namespace):
    cfg = {} 

    #Static
    cfg["epochs"] = args.epochs
    cfg["num_frames"] = args.num_frames

    #Tunable
    cfg["batch_size"] = trial.suggest_categorical("batch_size", [4, 8])
    cfg["dropout"] = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)

    #Optimizer
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