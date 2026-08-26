import os
import sys
import argparse, time
import gc
import torch
from torch.utils.data import DataLoader
import optuna
from torch.utils.tensorboard import SummaryWriter

from src.utils import setup_env
from src.data import CustomDataset, prepare_and_split_data, check_data_leakage
from src.training import trainer, YawDDclassifier
from src.evaluation import evaluate

def objective(trial, trainset, valset, testset, args):
    """
    Optuna objective function for hyperparameter tuning.

    Args:
        trial (optuna.Trial): A specific hyperparameter trial.
        trainset (Dataset): The training dataset.
        valset (Dataset): The validation dataset.
        testset (Dataset): The independent test dataset.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        float: The best validation F1 score achieved in this trial.
    """
    trial_args = argparse.Namespace(**vars(args))

    # Define training hyperparameters to tune
    trial_args.batch_size = trial.suggest_categorical("batch_size", [4, 8])
    trial_args.freeze_backbone = bool(trial.suggest_categorical("freeze_backbone", [1, 0]))
    trial_args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    trial_args.dropout = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)
    print(f'=================================================================')
    print(f' batch_size: {trial_args.batch_size}, freeze_backbone: {trial_args.freeze_backbone}, lr: {trial_args.lr:0.5f}, dropout: {trial_args.dropout:0.1f}')

    # Establish computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize dataloaders
    trainloader = DataLoader(trainset, batch_size=trial_args.batch_size, num_workers=4, shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=trial_args.batch_size, num_workers=4, shuffle=False)
    testloader = DataLoader(testset, batch_size=trial_args.batch_size, num_workers=4, shuffle=False)

    # Initialize model
    model = YawDDclassifier(trial_args.dropout).to(device)

    trial_params = {
        "batch_size": trial_args.batch_size,
        "lr": trial_args.lr,
        "dropout": trial_args.dropout,
        "freeze_backbone": int(trial_args.freeze_backbone)
    }

    log_dir = f"runs/trial_{trial.number}"
    writer = SummaryWriter(log_dir=log_dir)

    try:
        # Launch training loop
        f1_val, epoch = trainer(
                trainloader=trainloader,
                valloader=valloader,
                model=model,
                epochs=trial_args.epochs,
                lr=trial_args.lr,
                freeze_backbone = trial_args.freeze_backbone,
                device=device,
                save_dir=f"models/trial_{trial.number}",
                trial_params=trial_params,
                tb_writer=writer,
                patience=trial_args.patience,
                trial=trial
                )
        
        # Evaluate on test set
        test_metrics = evaluate(testloader, model, device)
        print(f"\nTest Acc: {test_metrics['accuracy']:.3f}")
        print(f"Test F1: {test_metrics['f1']:.3f}")

        hparams_dict = {
            "batch_size": trial_args.batch_size,
            "lr": trial_args.lr,
            "dropout": trial_args.dropout,
            "freeze_backbone": int(trial_args.freeze_backbone)
        }
        
        metrics_dict = {
            "hparam/best_val_f1": f1_val,
            "hparam/best_val_epoch": epoch,
            "hparam/test_acc": test_metrics['accuracy'],
            "hparam/test_f1": test_metrics['f1'],
        }
        
        # Write hyperparameter metrics to TensorBoard log
        writer.add_hparams(hparams_dict, metrics_dict, run_name=".")
        return f1_val
    
    finally:
        # Close TensorBoard writer and free GPU memory even if pruned/exception occurs
        try:
            writer.close()
        except Exception:
            pass
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Record start time
    start_timestamp = time.time()

    print("Cuda status: ")
    print(torch.cuda.is_available())

    # Establish determinism (set seeds and precision) 
    setup_env(seed=0)    

    # Parse command-line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=2)
    parser.add_argument('--patience', type=int, default=7, help='Number of epochs to wait for improvement before early stopping.')
    parser.add_argument("--prepare_data", action="store_true", help="Run data preparation and split raw files before training")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of the dataset to use (0.0 to 1.0)")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Execution mode: 'train' for optimization, 'test' for evaluating a saved model.")
    parser.add_argument("--model_path", type=str, default="models/best_yawdd_model.pth", help="Path to the saved model state dict for testing.")
    args = parser.parse_args()

    # Prepare data if requested
    if args.prepare_data:
        print("=================================================================")
        print("Running automated data preparation and split...")
        prepare_and_split_data(data_fraction=args.data_fraction)
        print("=================================================================")

    # Verify data integrity
    check_data_leakage()

    # data preparation
    trainset = CustomDataset('train', num_frames=args.num_frames)
    valset = CustomDataset('val', num_frames=args.num_frames)
    testset = CustomDataset('test', num_frames=args.num_frames)

    if args.mode == "test":
        print("\n" + "="*65)
        print("RUNNING IN TEST MODE")
        print(f"Loading model weights from: {args.model_path}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model (Dropout value doesn't matter for eval mode)
        model = YawDDclassifier(dropout=0.5).to(device)
        
        # Load weights
        if not os.path.exists(args.model_path):
            print(f"[!] Error: Model file '{args.model_path}' not found.")
            sys.exit(1)
            
        model.load_state_dict(torch.load(args.model_path))
        
        # Create Test DataLoader (using batch size 8 as default for testing)
        testloader = DataLoader(testset, batch_size=8, num_workers=4, shuffle=False)
        
        # Evaluate
        print("Evaluating on test set...")
        test_metrics = evaluate(testloader, model, device)
        
        print(f"\n--- Final Test Results ---")
        print(f"Accuracy:  {test_metrics['accuracy']*100:.2f} %")
        print(f"F1-Score:  {test_metrics['f1']*100:.2f} %")
        print(f"Precision: {test_metrics['precision']*100:.2f} %")
        print(f"Recall:    {test_metrics['recall']*100:.2f} %")
        print("="*65 + "\n")
        
        # Terminate script early since we only want to test
        sys.exit(0)


    print("RUNNING IN TRAIN MODE")
    # Initialize and execute Optuna study, maximizing validation F1 score
    study = optuna.create_study(
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(
        lambda trial: objective(trial, trainset, valset, testset, args), 
        n_trials=args.n_trials, 
        show_progress_bar=True
    )

    # Print out best trial
    print(f'=================================================================\nBest trial (val_f1): {study.best_value:.4f}')
    print(f'  Params:')
    print(study.best_params.items())

    # Retrieve and save the optimal model weights
    best_trial = study.best_trial
    best_path = f"models/trial_{best_trial.number}/best_model.pth"

    checkpoint = torch.load(best_path)
    best_model = YawDDclassifier(best_trial.params.get("dropout", 0.5))
    best_model.load_state_dict(checkpoint["model_state_dict"])

    final_path = "models/best_yawdd_model.pth"
    torch.save(best_model.state_dict(), final_path)

    print(f"=================================================================\n-> Best model saved to: {final_path}")

    # Display total elapsed execution time
    time_passed = time.time()-start_timestamp
    print(f'\nTraining finished in {time_passed//3600}h {(time_passed%3600)//60}min {time_passed%60:.0f}s\n')