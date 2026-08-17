import argparse, time
import torch
from torch.utils.data import DataLoader #Wandelt Dataset in Batches um
import optuna #Bibliothek für Hyperparameter-Optimierung

from src.utils import setup_env
from src.data import YawDDDataset
from src.training import trainer, YawDDclassifier #importiert Modell und Trainingsfunktion
from src.evaluation import evaluate
import winsound #hier für Signalton

from torch.utils.data import ConcatDataset, Subset # um Datensätze zusammenzufügen
from sklearn.model_selection import KFold #für KFold Cross Validation benötigt
from torch.utils.tensorboard import SummaryWriter



# Trial: Versuch mit bestimmten Hyperparametern (Optuna), wird mehrfach aufgerufen
def objective(trial):

    # Parameter und Einstellungen, die Optuna für das Training wählen kann
    # training hyperparameters to tune
    # Batch Size, Optuna wählt entweder 4 oder 8
    args.batch_size = trial.suggest_categorical("batch_size", [4, 8])
    # Friert zufällig ein 0 = trainieren, 1 = einfrieren
    args.freeze_backbone = trial.suggest_categorical("freeze_backbone", [0, 1])
    # Lernrate variieren
    args.lr = trial.suggest_float("lr", 1e-5, 2e-4, log=True)
    # Dropout variieren
    args.dropout = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)
    args.threshold = trial.suggest_float("threshold", 0.25, 0.35)
    # Gibt aktuelle Hyperparameter aus
    print(f'=================================================================')
    print(f' batch_size: {args.batch_size}, freeze_backbone: {args.freeze_backbone}, lr: {args.lr:0.5f}, dropout: {args.dropout:0.1f}')
    print(f"... threshold: {args.threshold:.2f}")

    # Falls verfügbar Training auf NVIDIA GPU, sonst CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Trainings- und Validierungs-Datasets erstellen
    trainset = YawDDDataset('train', num_frames=args.num_frames, train = True)
    valset = YawDDDataset('val', num_frames=args.num_frames, train = False)

    
    
    # Kombiniertes Dataset für KFold erstellen
    full_dataset = ConcatDataset([trainset, valset])

    # Testset unabhängig anlegen
    testset = YawDDDataset('test', num_frames=args.num_frames, train = False)

    

    #Training mit KFold:
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    fold_f1s = []

    # TensorBoard-Logging für diesen Optuna-Trial
    writer = SummaryWriter(f"runs/optuna_trial_{trial.number}")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):

        print(f"\n===== FOLD {fold} =====")

        train_dataset = ConcatDataset([
            YawDDDataset('train', num_frames=args.num_frames, train=True),
            YawDDDataset('val',   num_frames=args.num_frames, train=True)
        ])

        val_dataset = ConcatDataset([
            YawDDDataset('train', num_frames=args.num_frames, train=False),
            YawDDDataset('val',   num_frames=args.num_frames, train=False)
        ])

        train_subset = Subset(train_dataset, train_idx)
        val_subset   = Subset(val_dataset, val_idx)

        # Dataloader
        trainloader = DataLoader(train_subset, batch_size=args.batch_size, num_workers=0, shuffle=True)
        valloader   = DataLoader(val_subset, batch_size=args.batch_size, num_workers=0, shuffle=False)


        # Modell pro Fold neu initialisieren
        model = YawDDclassifier(args.dropout).to(device)

        f1_val, _ = trainer(
            trainloader=trainloader,
            valloader=valloader,
            model=model,
            epochs=args.epochs,
            lr=args.lr,
            freeze_backbone=args.freeze_backbone,
            device=device,
            threshold=args.threshold,
            patience=10,
            log_dir=f"runs/optuna_trial_{trial.number}"
        )

        fold_f1s.append(f1_val)
    
    mean_f1 = sum(fold_f1s) / len(fold_f1s)

    print(f"\nMean F1 over folds: {mean_f1:.3f}")

    #Hyperparameter-Logging für TensorBoard
    writer.add_hparams(
        hparam_dict={
            "batch_size": args.batch_size,
            "freeze_backbone": args.freeze_backbone,
            "lr": args.lr,
            "dropout": args.dropout,
            "threshold": args.threshold,
        },
        metric_dict={
            "val_f1": mean_f1,  # **Jetzt korrekt definiert!**
        },
        run_name=f"trial_{trial.number}"
    )


    writer.close()
    return mean_f1




if __name__ == "__main__":
    start_timestamp = time.time()

    setup_env(seed=0)    

    # Parameter für Training festlegen
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='YawDD')
    parser.add_argument("--num_frames", type=int, default=32) # Anzahl Frames pro Sample
    parser.add_argument("--epochs", type=int, default=20) # Anzahl Trainingsdurchläufe
    parser.add_argument("--n_trials", type=int, default=2) # Anzahl Optuna-Versuche
    args = parser.parse_args()

    # Optuna Study erstellen und ausführen. Für jeden Trial neue Hyperparameter, komplettes Training
    study = optuna.create_study(direction="maximize") # Ziel: Maximiere den F1-Score
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True) # Führt mehrere Trials durch
    

    # Bestes F1 und Hyperparameter ausgeben
    print(f'=================================================================\nBest trial (val_f1): {study.best_value:.4f}')
    print(f'  Params:')
    print(study.best_params.items())





    # ===== FINALES TRAINING AUF ALLEN TRAININGS- UND VALIDIERUNGSDATEN =====

    trainset = YawDDDataset('train', num_frames=args.num_frames, train = True)
    valset   = YawDDDataset('val', num_frames=args.num_frames, train = True)

    full_dataset = ConcatDataset([trainset, valset])

    batch_size = study.best_params['batch_size']

    trainloader = DataLoader(full_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    # Modell mit besten Parametern neu initialisieren
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = YawDDclassifier(study.best_params['dropout']).to(device)

    log_dir = f"runs/final_train_{time.strftime('%Y%m%d_%H%M%S')}"
    best_f1, best_epoch = trainer(
        trainloader=trainloader,
        valloader=trainloader,  # egal, finales Training
        model=best_model,
        epochs=args.epochs,
        lr=1e-4,                # finales Training mit fixer Lernrate
        freeze_backbone=study.best_params['freeze_backbone'],
        device=device,
        threshold=study.best_params['threshold'],
        patience=10,
        log_dir=log_dir
    )

    

    #===== MODELL SPEICHERN MIT HYPERPARAMETERN =====
    # Erstelle neues Modell mit den besten Parametern
    best_model = YawDDclassifier(study.best_params['dropout']).to(device)

    # Speichere NUR die Gewichte (kein Dictionary)
    torch.save(best_model.state_dict(), "best_model.pt")



    # Speichere Modell + Hyperparameter in EINER Datei
    '''torch.save({
        "model_state": best_model.state_dict(),  #Nur die Gewichte speichern
        "dropout": study.best_params['dropout'],
        "threshold": study.best_params['threshold'],
        "freeze_backbone": study.best_params['freeze_backbone']
    }, "best_model.pt")'''

    print("\nBeste Hyperparameter wurden mit dem Modell gespeichert!")




    # ===== ABSCHLIEßENDER TEST AUF TESTDATEN =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Modell mit besten Parametern erstellen
    best_model = YawDDclassifier(study.best_params['dropout']).to(device)

    # Gewichte laden
    best_model.load_state_dict(torch.load("best_model.pt"))

    # Testset laden
    testset = YawDDDataset('test', num_frames=args.num_frames)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    # Evaluation
    test_metrics = evaluate(testloader, best_model, device, study.best_params['threshold'])

    print(f'\n================ FINAL TEST ================')
    print(f"Test Acc: {test_metrics['accuracy']:.3f}")
    print(f"Test F1:  {test_metrics['f1']:.3f}")


    #TensorBoard-Logging für finales Training
    writer = SummaryWriter(log_dir)
    writer.add_hparams(
        hparam_dict={
            "batch_size": study.best_params['batch_size'],
            "freeze_backbone": study.best_params['freeze_backbone'],
            "lr": 1e-4,
            "dropout": study.best_params['dropout'],
            "threshold": study.best_params['threshold'],
        },
        metric_dict={
            "val_f1": best_f1,
        },
        run_name="best_model"
    )
    writer.close()


    # info on training time
    time_passed = time.time()-start_timestamp
    print(f'\nTraining finished in {time_passed//3600}h {(time_passed%3600)//60}min {time_passed%60:.0f}s\n')


    #Signaltöne nach dem Test ausgeben
    for _ in range(3):
        winsound.Beep(1000, 500)

