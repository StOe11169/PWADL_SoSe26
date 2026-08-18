# Betriebssystemmodul für Umgebungsvariablen, Pfade und Ordnerverwaltung
import os
# TensorFlow-/TensorBoard-Info- und Warnmeldungen in der Konsole reduzieren
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# argparse ermöglicht das Setzen von Parametern über die Kommandozeile
import argparse
# time wird zur Messung der Gesamtlaufzeit verwendet
import time
# NumPy wird für numerische Operationen und Mittelwertberechnungen genutzt
import numpy as np
# PyTorch wird für Modelltraining, Checkpoints und Geräteverwaltung verwendet
import torch
# DataLoader erstellt Batches aus Dataset-Objekten
from torch.utils.data import DataLoader, Subset
# SummaryWriter ermöglicht Logging von Hyperparametern und Metriken in TensorBoard
from torch.utils.tensorboard import SummaryWriter
# Optuna wird für die Hyperparameteroptimierung eingesetzt
import optuna
# Hilfsfunktion zum Setzen von Seeds und reproduzierbaren Einstellungen
from src.utils import setup_env
# Dataset-Klasse und Funktionen zur Split-Erzeugung bzw. Metadatenabfrage
from src.data import YawDDDataset, get_metadata_for_split, create_split_csv
# Trainingsfunktion und Modellklasse
from src.training import trainer, YawDDclassifier
# Evaluierungsfunktion für Validierung und Test
from src.evaluation import evaluate


# Wird für einen Signalton nach Trainingsende genutzt (Funktioniert nur unter Windows)
try:
    import winsound

    # Flag setzen, falls winsound erfolgreich importiert wurde
    HAS_WINSOUND = True

# ImportError abfangen, damit der Code auch auf anderen Betriebssystemen lauffähig bleibt
except ImportError:

    # Signalton deaktivieren, wenn winsound nicht verfügbar ist
    HAS_WINSOUND = False


def build_full_metadata():
    """
    Lädt die Metadaten des kombinierten trainval-Splits.

    Der trainval-Split dient als Grundlage für die Cross-Validation.
    Der unabhängige Testsplit bleibt dabei unangetastet.

    Returns:
        tuple: DataFrame, Label-Array und Gruppen-Array.
    """

    # Metadaten für train und val gemeinsam laden
    df_full = get_metadata_for_split("trainval")

    # Binäre Labels als Integer-Array extrahieren
    labels = df_full["yawning"].astype(int).to_numpy()

    # IDs als Gruppenvariable für gruppierte Cross-Validation extrahieren
    groups = df_full["id"].astype(str).to_numpy()

    # Metadaten, Labels und Gruppen zurückgeben
    return df_full, labels, groups


def get_cv_splits(labels, groups, n_splits=5):
    """
    Erstellt Cross-Validation-Splits für den trainval-Datensatz.

    Bevorzugt wird StratifiedGroupKFold, damit die Klassenverteilung möglichst
    erhalten bleibt und gleiche IDs nicht gleichzeitig in Training und Validierung liegen.

    Args:
        labels (np.ndarray): Binäre Labels.
        groups (np.ndarray): Gruppen-IDs zur Vermeidung von ID-Overlap.
        n_splits (int): Anzahl der Cross-Validation-Folds.

    Returns:
        list: Liste von Train-/Validierungsindexpaaren.
    """

    # Dummy-Featurematrix erzeugen, da scikit-learn ein X-Argument erwartet
    X_dummy = np.zeros(len(labels))

    # Eindeutige Gruppen bestimmen
    unique_groups = np.unique(groups)

    # Sicherstellen, dass nicht mehr Folds als Gruppen angefordert werden
    if n_splits > len(unique_groups):
        raise ValueError(
            f"n_splits={n_splits} ist größer als die Anzahl eindeutiger IDs "
            f"im trainval-Split ({len(unique_groups)}). "
            f"Bitte n_splits_cv reduzieren."
        )

    # Zuerst versuchen, eine stratifizierte gruppierte Cross-Validation zu verwenden
    try:

        # StratifiedGroupKFold trennt Gruppen und berücksichtigt möglichst die Klassenverteilung
        from sklearn.model_selection import StratifiedGroupKFold

        # Hinweis zur verwendeten Split-Methode ausgeben
        print("Verwende StratifiedGroupKFold für Cross-Validation.")

        # Cross-Validation-Objekt initialisieren
        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=0
        )

        # Splits als Liste erzeugen
        splits = list(cv.split(X_dummy, labels, groups))

        # Erzeugte Splits zurückgeben
        return splits

    # Falls StratifiedGroupKFold nicht verfügbar ist oder fehlschlägt, Fallback verwenden
    except Exception as e:

        # Fehlermeldung und Fallback-Hinweis ausgeben
        print(f"StratifiedGroupKFold nicht verfügbar oder fehlgeschlagen: {e}")
        print("Fallback auf GroupKFold.")

    # GroupKFold als robuste Alternative importieren
    from sklearn.model_selection import GroupKFold

    # GroupKFold stellt sicher, dass Gruppen nicht zwischen Train und Val überlappen
    cv = GroupKFold(n_splits=n_splits)

    # Splits als Liste erzeugen
    splits = list(cv.split(X_dummy, labels, groups))

    # Erzeugte Splits zurückgeben
    return splits


def objective(trial):
    """
    Zielfunktion für Optuna.

    Ein Trial entspricht einem Hyperparametersatz.
    Für jeden Trial wird eine Cross-Validation auf trainval durchgeführt.

    Args:
        trial: Optuna-Trial-Objekt.

    Returns:
        float: Mittlerer Validation-F1 über alle Folds.
    """

    # Batchgröße als kategorialen Hyperparameter wählen
    args.batch_size = trial.suggest_categorical("batch_size", [4, 8])

    # Festlegen, ob der Backbone teilweise eingefroren wird
    args.freeze_backbone = trial.suggest_categorical("freeze_backbone", [0, 1])

    # Lernrate logarithmisch im angegebenen Bereich wählen
    args.lr = trial.suggest_float("lr", 1e-5, 2e-4, log=True)

    # Dropout-Rate für den Klassifikationskopf wählen
    args.dropout = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)

    # Klassifikationsschwellwert für die binäre Entscheidung wählen
    args.threshold = trial.suggest_float("threshold", 0.25, 0.35)

    # Übersicht über den aktuellen Trial ausgeben
    print("\n=================================================================")
    print(f"Trial {trial.number}")
    print(f"batch_size      : {args.batch_size}")
    print(f"freeze_backbone : {args.freeze_backbone}")
    print(f"lr              : {args.lr:.6f}")
    print(f"dropout         : {args.dropout:.1f}")
    print(f"threshold       : {args.threshold:.3f}")
    print("=================================================================")

    # GPU verwenden, falls verfügbar, sonst CPU nutzen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset mit Trainingsaugmentationen für die Trainingsfolds erstellen
    full_dataset_aug = YawDDDataset(
        "trainval",
        num_frames=args.num_frames,
        train=True
    )

    # Dataset ohne Augmentation für Validierung und Trainingsmetriken erstellen
    full_dataset_eval = YawDDDataset(
        "trainval",
        num_frames=args.num_frames,
        train=False
    )

    # Labels und Gruppen-IDs für die Cross-Validation laden
    _, labels, groups = build_full_metadata()

    # Cross-Validation-Splits erzeugen
    splits = get_cv_splits(
        labels=labels,
        groups=groups,
        n_splits=args.n_splits_cv
    )

    # Liste zur Speicherung der F1-Scores aller Folds initialisieren
    fold_f1s = []

    # Über alle Cross-Validation-Folds iterieren
    for fold, (train_idx, val_idx) in enumerate(splits):

        # Aktuellen Fold in der Konsole ausgeben
        print(f"\n================ FOLD {fold + 1}/{len(splits)} ================")

        # Augmentierte Trainingsdaten für den aktuellen Fold auswählen
        train_subset_aug = Subset(full_dataset_aug, train_idx)

        # Nicht augmentierte Trainingsdaten für Trainingsmetriken auswählen
        train_subset_eval = Subset(full_dataset_eval, train_idx)

        # Nicht augmentierte Validierungsdaten für den aktuellen Fold auswählen
        val_subset = Subset(full_dataset_eval, val_idx)

        # DataLoader für augmentierte Trainingsdaten erstellen
        trainloader = DataLoader(
            train_subset_aug,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        # DataLoader für Trainingsmetriken ohne Augmentation erstellen
        train_eval_loader = DataLoader(
            train_subset_eval,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        # DataLoader für Validierungsdaten erstellen
        valloader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        # Modell für den aktuellen Fold neu initialisieren
        model = YawDDclassifier(args.dropout).to(device)

        # TensorBoard-Logverzeichnis für den aktuellen Fold definieren
        fold_log_dir = f"runs/optuna_trial_{trial.number}/fold_{fold}"

        # Checkpoint-Pfad für das beste Modell des aktuellen Folds definieren
        fold_save_path = f"checkpoints/trial_{trial.number}_fold_{fold}.pt"

        # Modell für den aktuellen Fold trainieren
        f1_val, best_epoch = trainer(
            trainloader=trainloader,
            valloader=valloader,
            train_eval_loader=train_eval_loader,
            model=model,
            epochs=args.epochs,
            lr=args.lr,
            freeze_backbone=args.freeze_backbone,
            device=device,
            threshold=args.threshold,
            patience=args.patience,
            log_dir=fold_log_dir,
            save_path=fold_save_path,
            early_stopping=True
        )

        # Bestes Fold-Ergebnis in der Konsole ausgeben
        print(
            f"Fold {fold + 1}: "
            f"Best Val F1 = {f1_val:.3f} "
            f"bei Epoche {best_epoch + 1}"
        )

        # F1-Score des aktuellen Folds speichern
        fold_f1s.append(f1_val)

        # Aktuellen Mittelwert der bisher abgeschlossenen Folds berechnen
        interim_mean = float(np.mean(fold_f1s))

        # Zwischenwert an Optuna melden
        trial.report(interim_mean, fold)

        # Trial frühzeitig abbrechen, falls Optuna diesen als ungünstig bewertet
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Mittleren F1-Score über alle Folds berechnen
    mean_f1 = float(np.mean(fold_f1s))

    # Standardabweichung der F1-Scores über alle Folds berechnen
    std_f1 = float(np.std(fold_f1s))

    # Zusammenfassung der Cross-Validation ausgeben
    print(f"\nMean F1 over folds: {mean_f1:.3f} ± {std_f1:.3f}")

    # TensorBoard-Writer für die Trial-Zusammenfassung erstellen
    writer = SummaryWriter(f"runs/optuna_trial_{trial.number}/summary")

    # Hyperparameter und aggregierte CV-Metriken in TensorBoard speichern
    writer.add_hparams(
        hparam_dict={
            "batch_size": args.batch_size,
            "freeze_backbone": args.freeze_backbone,
            "lr": args.lr,
            "dropout": args.dropout,
            "threshold": args.threshold,
        },
        metric_dict={
            "cv_mean_val_f1": mean_f1,
            "cv_std_val_f1": std_f1,
        },
        run_name=f"trial_{trial.number}"
    )

    # TensorBoard-Writer schließen
    writer.close()

    # Mittleren F1-Score als Optimierungsziel an Optuna zurückgeben
    return mean_f1


# Hauptprogramm nur ausführen, wenn diese Datei direkt gestartet wird
if __name__ == "__main__":

    # Startzeitpunkt zur späteren Laufzeitmessung speichern
    start_timestamp = time.time()

    # Seeds und deterministische Einstellungen setzen
    setup_env(seed=0)

    # Kommandozeilenparser initialisieren
    parser = argparse.ArgumentParser()

    # Optionaler Datensatzname für Dokumentation oder spätere Erweiterungen
    parser.add_argument("--data", type=str, default="YawDD")

    # Anzahl der Frames pro Video festlegen
    parser.add_argument("--num_frames", type=int, default=32)

    # Maximale Anzahl an Trainingsepochen pro Fold bzw. finalem Training
    parser.add_argument("--epochs", type=int, default=20)

    # Anzahl der Optuna-Trials festlegen
    parser.add_argument("--n_trials", type=int, default=2)

    # Anzahl der Cross-Validation-Folds festlegen
    parser.add_argument("--n_splits_cv", type=int, default=5)

    # Early-Stopping-Patience festlegen
    parser.add_argument("--patience", type=int, default=10)

    # Kommandozeilenargumente einlesen
    args = parser.parse_args()

    # Ordner für Modell-Checkpoints erstellen
    os.makedirs("checkpoints", exist_ok=True)

    # Split-Datei erzeugen oder bestehende Datei verwenden
    create_split_csv(
        video_dir="data/videos",
        split_csv="data/splits.csv",
        test_size=0.20,
        val_size=0.20,
        seed=0,
        force=False
    )

    #==================== Optuna-Hyperparamtersuche ===============================================================

    # Optuna-Study erstellen, bei der der F1-Score maximiert wird
    study = optuna.create_study(direction="maximize")

    # Hyperparameteroptimierung starten
    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    # Bestes Ergebnis der Hyperparametersuche ausgeben
    print("\n=================================================================")
    print(f"Best trial CV-F1: {study.best_value:.4f}")
    print("Beste Hyperparameter:")

    # Beste Hyperparameter einzeln ausgeben
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Zielgerät für finales Training und Test festlegen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Beste Batchgröße aus der Optuna-Study übernehmen
    best_batch_size = study.best_params["batch_size"]

    # Besten Dropout-Wert übernehmen
    best_dropout = study.best_params["dropout"]

    # Beste Backbone-Freeze-Einstellung übernehmen
    best_freeze_backbone = study.best_params["freeze_backbone"]

    # Beste Lernrate übernehmen
    best_lr = study.best_params["lr"]

    # Besten Klassifikationsschwellwert übernehmen
    best_threshold = study.best_params["threshold"]

    #=========================== Finales Training ===============================

    # Abschnittsüberschrift für finales Training ausgeben
    print("\n================ FINALES TRAINING AUF TRAINVAL ================")

    # Finales Trainingsdataset mit Augmentationen erstellen
    final_train_dataset_aug = YawDDDataset(
        "trainval",
        num_frames=args.num_frames,
        train=True
    )

    # Finales Trainingsdataset ohne Augmentation für Trainingsmetriken erstellen
    final_train_dataset_eval = YawDDDataset(
        "trainval",
        num_frames=args.num_frames,
        train=False
    )

    # DataLoader für finales Training erstellen
    final_trainloader = DataLoader(
        final_train_dataset_aug,
        batch_size=best_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # DataLoader für finale Trainingsmetriken ohne Augmentation erstellen
    final_train_eval_loader = DataLoader(
        final_train_dataset_eval,
        batch_size=best_batch_size,
        shuffle=False,
        num_workers=0
    )

    # Finales Modell mit bestem Dropout-Wert initialisieren
    final_model = YawDDclassifier(best_dropout).to(device)

    # Pfad für das finale state_dict definieren
    final_state_path = "checkpoints/final_state_dict.pt"

    # TensorBoard-Logverzeichnis für das finale Training definieren
    final_log_dir = f"runs/final_train_{time.strftime('%Y%m%d_%H%M%S')}"

    # Finales Modell auf trainval ohne Early Stopping trainieren
    final_train_f1, final_epoch = trainer(
        trainloader=final_trainloader,
        valloader=None,
        train_eval_loader=final_train_eval_loader,
        model=final_model,
        epochs=args.epochs,
        lr=best_lr,
        freeze_backbone=best_freeze_backbone,
        device=device,
        threshold=best_threshold,
        patience=args.patience,
        log_dir=final_log_dir,
        save_path=final_state_path,
        early_stopping=False
    )

    # Gespeicherte finale Modellgewichte laden
    final_model.load_state_dict(torch.load(final_state_path, map_location=device))

    # Einheitlichen Pfad für den finalen Checkpoint definieren
    final_checkpoint_path = "best_model_final.pt"

    # Finales Modell inklusive relevanter Hyperparameter speichern
    torch.save(
        {
            "model_state": final_model.state_dict(),
            "dropout": best_dropout,
            "threshold": best_threshold,
            "freeze_backbone": best_freeze_backbone,
            "batch_size": best_batch_size,
            "lr": best_lr,
            "num_frames": args.num_frames,
            "cv_best_f1": study.best_value,
            "final_train_f1": final_train_f1,
        },
        final_checkpoint_path
    )

    # Speicherort des finalen Modells ausgeben
    print(f"\nFinales Modell gespeichert unter: {final_checkpoint_path}")

   #==============Finaler Test===========================================================

    # Abschnittsüberschrift für finalen Test ausgeben
    print("\n================ FINALER TEST AUF TESTDATEN ================")

    # Testdataset ohne Augmentation erstellen
    testset = YawDDDataset(
        "test",
        num_frames=args.num_frames,
        train=False
    )

    # DataLoader für Testdaten erstellen
    testloader = DataLoader(
        testset,
        batch_size=best_batch_size,
        shuffle=False,
        num_workers=0
    )

    # Finales Modell auf dem unabhängigen Testsplit evaluieren
    test_metrics = evaluate(
        testloader,
        final_model,
        device,
        best_threshold,
        writer=None,
        epoch=None,
        prefix="test",
        verbose=True
    )

    # Zusammenfassung der finalen Testmetriken ausgeben
    print("\n================ FINAL TEST ================")

    # Test-Accuracy ausgeben
    print(f"Test Accuracy : {test_metrics['accuracy']:.3f}")

    # Test-Precision ausgeben
    print(f"Test Precision: {test_metrics['precision']:.3f}")

    # Test-Recall ausgeben
    print(f"Test Recall   : {test_metrics['recall']:.3f}")

    # Test-F1 ausgeben
    print(f"Test F1       : {test_metrics['f1']:.3f}")

    # ROC-AUC nur formatiert ausgeben, wenn der Wert definiert ist
    if test_metrics["roc_auc"] == test_metrics["roc_auc"]:
        print(f"Test ROC-AUC  : {test_metrics['roc_auc']:.3f}")
    else:
        print("Test ROC-AUC  : nan")

    # PR-AUC nur formatiert ausgeben, wenn der Wert definiert ist
    if test_metrics["pr_auc"] == test_metrics["pr_auc"]:
        print(f"Test PR-AUC   : {test_metrics['pr_auc']:.3f}")
    else:
        print("Test PR-AUC   : nan")

    # Überschrift für Confusion Matrix ausgeben
    print("Confusion Matrix:")

    # Confusion Matrix ausgeben
    print(test_metrics["confusion_matrix"])

    # TensorBoard-Writer für finale Hyperparameter und Testmetriken erstellen
    writer = SummaryWriter(final_log_dir)

    # Finale Hyperparameter und Ergebniswerte in TensorBoard speichern
    writer.add_hparams(
        hparam_dict={
            "batch_size": best_batch_size,
            "freeze_backbone": best_freeze_backbone,
            "lr": best_lr,
            "dropout": best_dropout,
            "threshold": best_threshold,
            "num_frames": args.num_frames,
        },
        metric_dict={
            "cv_best_f1": study.best_value,
            "final_train_f1": final_train_f1,
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
        },
        run_name="best_model"
    )

    # TensorBoard-Writer schließen
    writer.close()

    # Gesamtlaufzeit berechnen
    time_passed = time.time() - start_timestamp

    # Gesamtlaufzeit formatiert ausgeben
    print(
        f"\nTraining finished in "
        f"{time_passed // 3600:.0f}h "
        f"{(time_passed % 3600) // 60:.0f}min "
        f"{time_passed % 60:.0f}s\n"
    )

    # Unter Windows nach Abschluss einen Signalton ausgeben
    if HAS_WINSOUND:

        # Drei kurze Signaltöne abspielen
        for _ in range(3):
            winsound.Beep(1000, 500)
