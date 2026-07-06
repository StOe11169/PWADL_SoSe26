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



# Optional TODOs: 
# * Hand more hyperparameters as arguments / add to optuna search space
# * comparison with PWADL 2025: freeze/unfreeze backbone, two separate optimizers, lr scheduler
# * Tensorboard
# * Logging of results / save (best) model


# Trial: Versuch mit bestimmten Hyperparametern (Optuna), wird mehrfach aufgerufen
def objective(trial):

    # Parameter und Einstellungen, die Optuna für das Training wählen kann
    # training hyperparameters to tune
    # Batch Size, Optuna wählt entweder 4 oder 8: Zum Test verkleinert
    args.batch_size = trial.suggest_categorical("batch_size", [8])  #[4, 8])
    # Friert zufällig ein 0 = trainieren, 1 = einfrieren
    args.freeze_backbone = trial.suggest_categorical("freeze_backbone", [0, 1]) #[0, 1])
    # Lernrate zwischen 0.00001 und 0.001, logarithmisch verteilt.
    
    args.lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
    #args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True) #1e-4 #im Test ist 1e-3 zu groß
    # Dropout zwischen 0.2 - 0.6, in 0.1 Schritten 
    args.dropout = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)
    args.threshold = trial.suggest_float("threshold", 0.25, 0.35)
    # Gibt aktuelle Hyperparameter aus
    print(f'=================================================================')
    print(f' batch_size: {args.batch_size}, freeze_backbone: {args.freeze_backbone}, lr: {args.lr:0.5f}, dropout: {args.dropout:0.1f}')
    print(f"... threshold: {args.threshold:.2f}")

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data preparation
    # Drei Splits, Wie viele Frames pro Video
    trainset = YawDDDataset('train', num_frames=args.num_frames, train = True)
    valset = YawDDDataset('val', num_frames=args.num_frames, train = False)
    #Komplettes Dataset für KFold
    
    
    full_dataset = YawDDDataset('train', num_frames=args.num_frames, train=True)
    #full_dataset = ConcatDataset([trainset, valset])

    testset = YawDDDataset('test', num_frames=args.num_frames, train = False)

    
    #Klassischer Split
    """
    # dataloaders
    # Shuffle mischt Daten durch, drop_last entfernt unvollständige Batches, num_workers = 0 Daten werden im Hauptprozess geladen,
    # keine Parallelisierung. Sonst würde neben dem Training schon der nächste Batch vorbereitet werden, um GPU auszulasten
    # Könnte hier etwa auf CPU-Kerne/2 hochgesetzt werden AUSPROBIEREN
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=10, shuffle=True, drop_last=False)   #=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=10, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=10, shuffle=False)

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
            device=device,
            threshold=args.threshold
            )
    
    # Decide if trial should be pruned
    trial.report(f1_val, epoch) # Meldet Zwischenergebnis an Optuna
    if trial.should_prune(): # schlechte Trials werden früh abgebrochen, um Zeit zu sparen
        raise optuna.TrialPruned()
    
    
    #Komplett Entfernen, Data Leakage
    # test
    # Bewertet Modell auf Testdaten
    #test_metrics = evaluate(testloader, model, device)
    #print(f"=================================================================\nTest Acc: {test_metrics['accuracy']:.3f}") 
    return f1_val #Optuna optimiert diesen Wert
    """

    #Training mit KFold:
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):

        print(f"\n===== FOLD {fold} =====")

        # Subsets erstellen
        #train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        #val_subset   = torch.utils.data.Subset(full_dataset, val_idx)
        train_subset = Subset(full_dataset, train_idx)
        val_subset   = Subset(full_dataset, val_idx)
        train_subset.dataset.train = True
        val_subset.dataset.train = False


        # Dataloader
        trainloader = DataLoader(train_subset, batch_size=args.batch_size, num_workers=0, shuffle=True)
        valloader   = DataLoader(val_subset, batch_size=args.batch_size, num_workers=0, shuffle=False)


        # Modell NEU pro Fold!
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
            patience=10
        )

        fold_f1s.append(f1_val)
    
    mean_f1 = sum(fold_f1s) / len(fold_f1s)

    print(f"\nMean F1 over folds: {mean_f1:.3f}")

    return mean_f1




if __name__ == "__main__":
    # get start time
    start_timestamp = time.time()

    # set seed and precision
    # setzt Seed für reproduzierbare Ergebnisse
    setup_env(seed=0)    

    # get args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='YawDD')
    parser.add_argument("--num_frames", type=int, default=2) # Anzahl Frames pro Sample
    parser.add_argument("--epochs", type=int, default=2) # Anzahl Trainingsdurchläufe
    parser.add_argument("--n_trials", type=int, default=1) # Anzahl Optuna-Versuche
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





        # ===== FINAL TRAINING AUF GANZEM DATASET =====

    trainset = YawDDDataset('train', num_frames=args.num_frames)
    valset   = YawDDDataset('val', num_frames=args.num_frames)

    full_dataset = ConcatDataset([trainset, valset])


    batch_size = study.best_params['batch_size']

    trainloader = DataLoader(full_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    # Modell neu mit besten Parametern
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = YawDDclassifier(study.best_params['dropout']).to(device)

    trainer(
        trainloader=trainloader,
        valloader=trainloader,  # egal, finales Training
        model=best_model,
        epochs=args.epochs,
        lr=1e-4,
        freeze_backbone=study.best_params['freeze_backbone'],
        device=device,
        threshold=study.best_params['threshold'],
        patience=10
    )








    # ===== BESTES MODELL LADEN =====
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
    #test_metrics = evaluate(testloader, best_model, device)

    print(f'\n================ FINAL TEST ================')
    print(f"Test Acc: {test_metrics['accuracy']:.3f}")
    print(f"Test F1:  {test_metrics['f1']:.3f}")





    # info on training time
    time_passed = time.time()-start_timestamp
    print(f'\nTraining finished in {time_passed//3600}h {(time_passed%3600)//60}min {time_passed%60:.0f}s\n')


    #Signaltöne
    for _ in range(3):
        winsound.Beep(1000, 500)


    """
    Ideen:
    - Backbone teilweise unfreezen -> Aktuell Optuna Args und im Training Gradient = True gesetzt
    -----------------------------------------------------------
    Erledigt:
    - Learning Rate Scheduler hinzufügen
    - Backbone unfreeze
    - Testset nur am Ende nutzen
    - Bestes Modell speichern
    - Signalton nach Trainingsende eingefügt
    - Von Lern- zu Generalisierungs- zu overfitting- zu Dataleakage Problem
    -KFold Cross Validation testweise implimentiert
    -RandomFlip, Rotation, ColorJitter und getrennte Transforms integriert
    ----------------------------------------------------------------
    Erkenntnisse:
    - Treshold ideal bei ~0,33
    - Auf 16GB RAM Systemen mit GPU Trainingsgeschwindigkeit durch RAM beschränkt
    - Windows hat Probleme mit Multi-Worker-Systemen
    """