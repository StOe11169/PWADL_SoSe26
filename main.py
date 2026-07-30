import os
import glob
from datetime import datetime
import argparse, time
import torch
from torch.utils.data import DataLoader
import optuna
import json
import shutil
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

#Load dataset and create splits
df = get_all_data_paths("data")
train_df, val_df, test_df = create_group_splits(df, output_dir= study_dir, test_size=0.15, val_size=0.15,seed=42, )



def objective(trial, study_dir, args):
    try:

        #Load config & print to console
        cfg = build_config(trial, args)
        print(f"Trial {trial.number}")
        for k, v in cfg.items():
            print(f"{k}: {v}")

        
        # get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # data preparation
        trainset = YawDDDataset(train_df, num_frames=cfg["num_frames"])
        valset = YawDDDataset(val_df, num_frames=cfg["num_frames"])
        testset = YawDDDataset(test_df, num_frames=cfg["num_frames"])
        

        # dataloaders
        trainloader = DataLoader(trainset, batch_size=cfg["batch_size"], num_workers=0, shuffle=True, drop_last=True)
        valloader = DataLoader(valset, batch_size=cfg["batch_size"], num_workers=0, shuffle=False)
        testloader = DataLoader(testset, batch_size=cfg["batch_size"], num_workers=0, shuffle=False)

        # model
        model = YawDDclassifier(cfg["dropout"]).to(device)
  

        # start training
        f1_val, epoch = trainer(trainloader=trainloader, valloader=valloader, model=model, device=device, trial_number= trial.number, study_dir = study_dir, cfg=cfg)
        
        # Decide if trial should be pruned
        trial.report(f1_val, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        #Save Trial summary
        trial_summary = {"trial_number": trial.number, "f1_val": f1_val, "best_epoch": epoch, "params": cfg }
        with open(os.path.join(study_dir, f"trial_{trial.number}_summary.json"), "w") as f:
            json.dump(trial_summary, f, indent=4)
        
        # Load best model before testing
        best_model_path = os.path.join(study_dir, f"best_model_trial_{trial.number}.pth")
        #Check if a best model exists
        if not os.path.exists(best_model_path):
            print(f"[Trial {trial.number}] No model saved (likely pruned or failed)")
            raise RuntimeError(f"No model saved for trial {trial.number}")

        checkpoint = torch.load(best_model_path, map_location=device)

        best_cfg = checkpoint["cfg"]

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # test
        test_metrics = evaluate(testloader, model, device)
        print(f"=================================================================\nTest Acc: {test_metrics['accuracy']:.3f}") 
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
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=2)
    args = parser.parse_args()

    
    # Create & run study, maximizing validation F1, lamba as extra study_dir arg isnt passed directly by optuna
    study = optuna.create_study(direction="maximize")
    #Start Tensorboard
    tb_process = start_tensorboard(study_dir)
    study.optimize(lambda trial: objective(trial, study_dir, args), n_trials=args.n_trials, show_progress_bar=True)
    
  
    print("=================================================================")
    print(f"Best trial (val_f1): {study.best_value:.4f}")

    best_trial = study.best_trial

    # ✅ Load full cfg from saved model
    best_model_path = os.path.join(study_dir, f"best_model_trial_{best_trial.number}.pth")

    if not os.path.exists(best_model_path):
        raise RuntimeError(f"Best model file not found for trial {best_trial.number}")

    checkpoint = torch.load(best_model_path, map_location="cpu")
    best_cfg = checkpoint["cfg"]

    # ✅ Save full best trial summary
    best_summary = {
    "trial_number": best_trial.number,
    "f1_val": best_trial.value,
    "optuna_params": best_trial.params,
    "cfg": best_cfg
    }

    with open(os.path.join(study_dir, "best_trial.json"), "w") as f:
        json.dump(best_summary, f, indent=4)

    # ✅ Copy best model to a fixed name (very useful later)
    best_model_dst = os.path.join(study_dir, "best_model.pth")
    shutil.copy(best_model_path, best_model_dst)

    # ✅ Clean print
    print("Best params (Optuna):")
    for k, v in best_trial.params.items():
     print(f"  {k}: {v}")
    

    # Print out best trial
    #print(f'=================================================================\nBest trial (val_f1): {study.best_value:.4f}')
    #print(f'  Params:')
    #print(study.best_params.items())

    # info on training time
    time_passed = time.time()-start_timestamp
    print(f'\nTraining finished in {time_passed//3600}h {(time_passed%3600)//60}min {time_passed%60:.0f}s\n')
    
    tb_process.terminate()
    