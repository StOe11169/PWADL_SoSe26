from csv import writer
import argparse, time
import torch
from torch.utils.data import DataLoader
import optuna

from src.utils import setup_env
from src.data import YawDDDataset
from src.training import trainer, YawDDclassifier
from src.evaluation import evaluate

from torch.utils.tensorboard import SummaryWriter


# Optional TODOs: 
# * Hand more hyperparameters as arguments / add to optuna search space
# * comparison with PWADL 2025: freeze/unfreeze backbone, two separate optimizers, lr scheduler
# * Tensorboard
# * Logging of results / save (best) model

def objective(trial):

    # training hyperparameters to tune
    #args.batch_size = 4
    #args.freeze_backbone = 0
    #args.lr = 0.000102997397467
    #args.dropout = 0.2
    args.batch_size = trial.suggest_categorical("batch_size", [4, 8, 12, 16])
    args.freeze_backbone = trial.suggest_categorical("freeze_backbone", [0, 1])
    args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    args.dropout = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)
    print(f'=================================================================')
    print(f' batch_size: {args.batch_size}, freeze_backbone: {args.freeze_backbone}, lr: {args.lr:0.5f}, dropout: {args.dropout:0.1f}')

    writer = SummaryWriter(
        log_dir=f"runs/trial_{trial.number}"
    )

    

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data preparation
    trainset = YawDDDataset('train', num_frames=args.num_frames)
    valset = YawDDDataset('val', num_frames=args.num_frames)
    testset = YawDDDataset('test', num_frames=args.num_frames)

    # dataloaders
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    # model
    model = YawDDclassifier(args.dropout).to(device)
    
    # start training
    f1_val, epoch = trainer(trainloader=trainloader,
            valloader=valloader,
            model=model,
            epochs=args.epochs,
            lr=args.lr,
            freeze_backbone = args.freeze_backbone,
            device=device,
            writer=writer,
            trial_number=trial.number
            )
    
    # Decide if trial should be pruned
    trial.report(f1_val, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    # test
    test_metrics = evaluate(testloader, model, device)
    for metric, value in test_metrics.items():
        writer.add_scalar(
            f"{metric}/Test",
            value,
            0
        )
    print(f"=================================================================\nTest Acc: {test_metrics['accuracy']:.3f}") 

    writer.add_hparams(
        {
        "batch_size": args.batch_size,
        "freeze_backbone": args.freeze_backbone,
        "lr": args.lr,
        "dropout": args.dropout,
        },
        {
        "f1_val": f1_val,
        "test_acc": test_metrics["accuracy"],
        }
    )

    writer.close()

    return f1_val


if __name__ == "__main__":
    # get start time
    start_timestamp = time.time()

    # set seed and precision
    setup_env(seed=0)    

    # get args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='YawDD')
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--n_trials", type=int, default=2)
    args = parser.parse_args()
    
    # Create & run study, maximizing validation F1
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Print out best trial
    print(f'=================================================================\nBest trial (val_f1): {study.best_value:.4f}')
    print(f'  Params:')
    print(study.best_params.items())

    # info on training time
    time_passed = time.time()-start_timestamp
    print(f'\nTraining finished in {time_passed//3600}h {(time_passed%3600)//60}min {time_passed%60:.0f}s\n')