import argparse, time
import torch
from torch.utils.data import DataLoader #Wandelt Dataset in Batches um
import optuna #Bibliothek für Hyperparameter-Optimierung

from src.utils import setup_env
from src.data import YawDDDataset
from src.training import trainer, YawDDclassifier #importiert Modell und Trainingsfunktion
from src.evaluation import evaluate


# Optional TODOs: 
# * Hand more hyperparameters as arguments / add to optuna search space
# * comparison with PWADL 2025: freeze/unfreeze backbone, two separate optimizers, lr scheduler
# * Tensorboard
# * Logging of results / save (best) model


# Trial: Versuch mit bestimmten Hyperparametern (Optuna), wird mehrfach aufgerufen
def objective(trial):

    # training hyperparameters to tune
    # Batch Size, Optuna wählt entweder 4 oder 8
    args.batch_size = trial.suggest_categorical("batch_size", [4, 8])
    # Friert zufällig ein 0 = trainieren, 1 = einfrieren
    args.freeze_backbone = trial.suggest_categorical("freeze_backbone", [0, 1])
    # Lernrate zwischen 0.00001 und 0.001, logarithmisch verteilt.
    args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    # Dropout zwischen 0.2 - 0.6, in 0.1 Schritten 
    args.dropout = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)
    # Gibt aktuelle Hyperparameter aus
    print(f'=================================================================')
    print(f' batch_size: {args.batch_size}, freeze_backbone: {args.freeze_backbone}, lr: {args.lr:0.5f}, dropout: {args.dropout:0.1f}')

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data preparation
    # Drei Splits, Wie viele Frames pro Video
    trainset = YawDDDataset('train', num_frames=args.num_frames)
    valset = YawDDDataset('val', num_frames=args.num_frames)
    testset = YawDDDataset('test', num_frames=args.num_frames)

    # dataloaders
    # Shuffle mischt Daten durch, drop_last entfernt unvollständige Batches, num_workers = 0 Daten werden im Hauptprozess geladen,
    # keine Parallelisierung. Sonst würde neben dem Training schon der nächste Batch vorbereitet werden, um GPU auszulasten
    # Könnte hier etwa auf CPU-Kerne/2 hochgesetzt werden AUSPROBIEREN
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=6, shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    # model
    # Initialisiert Modell, schieb es auf Device
    model = YawDDclassifier(args.dropout).to(device)
    
    # start training
    # Übergibt Daten, Modell, Hyperparameter, gibt beste F1 und die entsprechende Epoche zurück
    f1_val, epoch = trainer(trainloader=trainloader,
            valloader=valloader,
            model=model,
            epochs=args.epochs,
            lr=args.lr,
            freeze_backbone = args.freeze_backbone,
            device=device
            )
    
    # Decide if trial should be pruned
    trial.report(f1_val, epoch) # Meldet Zwischenergebnis an Optuna
    if trial.should_prune(): # schlechte Trials werden früh abgebrochen, um Zeit zu sparen
        raise optuna.TrialPruned()
    
    # test
    # Bewertet Modell auf Testdaten
    test_metrics = evaluate(testloader, model, device)
    print(f"=================================================================\nTest Acc: {test_metrics['accuracy']:.3f}") 
    return f1_val #Optuna optimiert diesen Wert


if __name__ == "__main__":
    # get start time
    start_timestamp = time.time()

    # set seed and precision
    # setzt Seed für reproduzierbare Ergebnisse
    setup_env(seed=0)    

    # get args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='YawDD')
    parser.add_argument("--num_frames", type=int, default=16) # Anzahl Frames pro Sample
    parser.add_argument("--epochs", type=int, default=5) # Anzahl Trainingsdurchläufe
    parser.add_argument("--n_trials", type=int, default=2) # Anzahl Optuna-Versuche
    args = parser.parse_args() # Liest Parameter aus CLI

    # Create & run study, maximizing validation F1
    study = optuna.create_study(direction="maximize") # Ziel: Maximiere den F1-Score
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True) # Führt mehrere Trials durch
    # Für jeden Trial neue Hyperparameter, komplettes Training

    # Print out best trial
    # Bestes F1 und Hyperparameter ausgeben
    print(f'=================================================================\nBest trial (val_f1): {study.best_value:.4f}')
    print(f'  Params:')
    print(study.best_params.items())

    # info on training time
    time_passed = time.time()-start_timestamp
    print(f'\nTraining finished in {time_passed//3600}h {(time_passed%3600)//60}min {time_passed%60:.0f}s\n')

    """
    Ideen:
    - Testset nicht für jeden Trial benutzen --> Testset nur nach Optuna evaluieren
    - Freeze Backbone Booleans statt ints setzen (Form: True, False statt 0,1)
    - Nicht nur besten Score, sondern auch bestes Modell speichern
    - Learning Rate Scheduler hinzufügen
    - Backbone teilweise unfreezen
    """