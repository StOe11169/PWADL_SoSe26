import os
import argparse, time
import gc
import torch
from torch.utils.data import DataLoader
import optuna
from torch.utils.tensorboard import SummaryWriter

from src.utils import setup_env
from src.data import CustomDataset, prepare_and_split_data
from src.training import trainer, YawDDclassifier
from src.evaluation import evaluate


# Optional TODOs: 
# * Hand more hyperparameters as arguments / add to optuna search space
# * comparison with PWADL 2025: freeze/unfreeze backbone, two separate optimizers, lr scheduler

def objective(trial):

    # training hyperparameters to tune
    args.batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    args.freeze_backbone = trial.suggest_categorical("freeze_backbone", [0, 1])
    args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    args.dropout = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)
    print(f'=================================================================')
    print(f' batch_size: {args.batch_size}, freeze_backbone: {args.freeze_backbone}, lr: {args.lr:0.5f}, dropout: {args.dropout:0.1f}')

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data preparation
    trainset = CustomDataset('train', num_frames=args.num_frames)
    valset = CustomDataset('val', num_frames=args.num_frames)
    testset = CustomDataset('test', num_frames=args.num_frames)

    # dataloaders
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=3, shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=3, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=3, shuffle=False)

    # model
    model = YawDDclassifier(args.dropout).to(device)
    
    trial_params = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "freeze_backbone": args.freeze_backbone
    }

    log_dir = f"runs/trial_{trial.number}"
    writer = SummaryWriter(log_dir=log_dir)

    # start training
    f1_val, epoch = trainer(trainloader=trainloader,
            valloader=valloader,
            model=model,
            epochs=args.epochs,
            lr=args.lr,
            freeze_backbone = args.freeze_backbone,
            device=device,
            save_dir=f"models/trial_{trial.number}",
            trial_params=trial_params,
            tb_writer=writer
            )
    
    # Decide if trial should be pruned
    trial.report(f1_val, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    # test
    test_metrics = evaluate(testloader, model, device)
    print(f"=================================================================\nTest Acc: {test_metrics['accuracy']:.3f}")

    hparams_dict = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "freeze_backbone": args.freeze_backbone
    }
    
    metrics_dict = {
        "hparam/best_val_f1": f1_val,
        "hparam/best_val_epoch": epoch,
        "hparam/test_acc": test_metrics['accuracy']
    }
    
    # Wirte parameter result matching in the log
    writer.add_hparams(hparams_dict, metrics_dict, run_name=".")
    writer.close()
    
    # Clear GPU VRAM
    del model
    if 'optimizer' in locals():
        del optimizer
    
    gc.collect()
    torch.cuda.empty_cache()

    return f1_val


if __name__ == "__main__":
    # get start time
    start_timestamp = time.time()

    print("Cuda status: ")
    print(torch.cuda.is_available())

    # set seed and precision
    setup_env(seed=0)    

    # get args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='YawDD')
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=2)
    parser.add_argument("--prepare_data", action="store_true", help="Run data preparation and split raw files before training")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of the dataset to use (0.0 to 1.0)")
    args = parser.parse_args()

    # Prepare data if requested
    if args.prepare_data:
        print("=================================================================")
        print("Running automated data preparation and split...")
        prepare_and_split_data(data_fraction=args.data_fraction)
        print("=================================================================")

    # Create & run study, maximizing validation F1
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Print out best trial
    print(f'=================================================================\nBest trial (val_f1): {study.best_value:.4f}')
    print(f'  Params:')
    print(study.best_params.items())

    best_trial = study.best_trial
    best_path = f"models/trial_{best_trial.number}/best_model.pth"

    checkpoint = torch.load(best_path)

    best_model = YawDDclassifier(best_trial.params.get("dropout", 0.5))
    best_model.load_state_dict(checkpoint["model_state_dict"])

    final_path = "models/best_yawdd_model.pth"
    torch.save(best_model.state_dict(), final_path)

    print(f"=================================================================\n-> Best model saved to: {final_path}")

    # info on training time
    time_passed = time.time()-start_timestamp
    print(f'\nTraining finished in {time_passed//3600}h {(time_passed%3600)//60}min {time_passed%60:.0f}s\n')