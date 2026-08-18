import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #TensorFlow soll INFO- und WARNING-Meldungen nicht anzeigen.
import argparse
import time
import os
import numpy as np
import torch

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import optuna

from src.utils import setup_env
from src.data import YawDDDataset, get_metadata_for_split, create_split_csv
from src.training import trainer, YawDDclassifier
from src.evaluation import evaluate

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


def build_full_metadata():
    """
    Metadaten für train+val.
    Das ist die Datenbasis für die Cross-Validation.

    Der Testsplit bleibt komplett unangetastet.
    """

    df_full = get_metadata_for_split("trainval")

    labels = df_full["yawning"].astype(int).to_numpy()
    groups = df_full["id"].astype(str).to_numpy()

    return df_full, labels, groups


def get_cv_splits(labels, groups, n_splits=5):
    """
    Erstellt Cross-Validation-Splits auf trainval.

    Bevorzugt:
        StratifiedGroupKFold

    Vorteil:
        - möglichst ähnliche Klassenverteilung
        - gleiche ID nicht in Train und Validation desselben Folds

    Fallback:
        GroupKFold

    Damit vermeiden wir, dass dieselbe Person/ID innerhalb der CV
    gleichzeitig in Trainings- und Validierungsdaten liegt.
    """

    X_dummy = np.zeros(len(labels))

    unique_groups = np.unique(groups)

    if n_splits > len(unique_groups):
        raise ValueError(
            f"n_splits={n_splits} ist größer als die Anzahl eindeutiger IDs "
            f"im trainval-Split ({len(unique_groups)}). "
            f"Bitte n_splits_cv reduzieren."
        )

    # Versuch 1: StratifiedGroupKFold
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        print("Verwende StratifiedGroupKFold für Cross-Validation.")

        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=0
        )

        splits = list(cv.split(X_dummy, labels, groups))

        return splits

    except Exception as e:
        print(f"StratifiedGroupKFold nicht verfügbar oder fehlgeschlagen: {e}")
        print("Fallback auf GroupKFold.")

    # Fallback: GroupKFold
    from sklearn.model_selection import GroupKFold

    cv = GroupKFold(n_splits=n_splits)

    splits = list(cv.split(X_dummy, labels, groups))

    return splits


def objective(trial):
    """
    Optuna-Zielfunktion.
    Ein Trial entspricht einem Hyperparametersatz.
    Für jeden Trial wird Cross-Validation auf trainval durchgeführt.
    """

    args.batch_size = trial.suggest_categorical("batch_size", [4, 8])
    args.freeze_backbone = trial.suggest_categorical("freeze_backbone", [0, 1])
    args.lr = trial.suggest_float("lr", 1e-5, 2e-4, log=True)
    args.dropout = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)
    args.threshold = trial.suggest_float("threshold", 0.25, 0.35)

    print("\n=================================================================")
    print(f"Trial {trial.number}")
    print(f"batch_size      : {args.batch_size}")
    print(f"freeze_backbone : {args.freeze_backbone}")
    print(f"lr              : {args.lr:.6f}")
    print(f"dropout         : {args.dropout:.1f}")
    print(f"threshold       : {args.threshold:.3f}")
    print("=================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ein Dataset mit Augmentation für Training
    full_dataset_aug = YawDDDataset(
        "trainval",
        num_frames=args.num_frames,
        train=True
    )

    # Dasselbe Dataset ohne Augmentation für Evaluation
    full_dataset_eval = YawDDDataset(
        "trainval",
        num_frames=args.num_frames,
        train=False
    )

    _, labels, groups = build_full_metadata()

    splits = get_cv_splits(
        labels=labels,
        groups=groups,
        n_splits=args.n_splits_cv
    )

    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n================ FOLD {fold + 1}/{len(splits)} ================")

        train_subset_aug = Subset(full_dataset_aug, train_idx)
        train_subset_eval = Subset(full_dataset_eval, train_idx)
        val_subset = Subset(full_dataset_eval, val_idx)

        trainloader = DataLoader(
            train_subset_aug,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        train_eval_loader = DataLoader(
            train_subset_eval,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        valloader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        model = YawDDclassifier(args.dropout).to(device)

        fold_log_dir = f"runs/optuna_trial_{trial.number}/fold_{fold}"
        fold_save_path = f"checkpoints/trial_{trial.number}_fold_{fold}.pt"

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

        print(
            f"Fold {fold + 1}: "
            f"Best Val F1 = {f1_val:.3f} "
            f"bei Epoche {best_epoch + 1}"
        )

        fold_f1s.append(f1_val)

        interim_mean = float(np.mean(fold_f1s))
        trial.report(interim_mean, fold)

        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_f1 = float(np.mean(fold_f1s))
    std_f1 = float(np.std(fold_f1s))

    print(f"\nMean F1 over folds: {mean_f1:.3f} ± {std_f1:.3f}")

    writer = SummaryWriter(f"runs/optuna_trial_{trial.number}/summary")

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

    writer.close()

    return mean_f1


if __name__ == "__main__":
    start_timestamp = time.time()

    setup_env(seed=0)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="YawDD")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n_trials", type=int, default=2)
    parser.add_argument("--n_splits_cv", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    # Split-Datei aus data/videos erzeugen oder vorhandene verwenden.
    # force=False ist wichtig, damit der Testsplit stabil bleibt.
    create_split_csv(
        video_dir="data/videos",
        split_csv="data/splits.csv",
        test_size=0.20,
        val_size=0.20,
        seed=0,
        force=False
    )

    # ============================================================
    # Optuna-Hyperparametersuche
    # ============================================================

    study = optuna.create_study(direction="maximize")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    print("\n=================================================================")
    print(f"Best trial CV-F1: {study.best_value:.4f}")
    print("Beste Hyperparameter:")

    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_batch_size = study.best_params["batch_size"]
    best_dropout = study.best_params["dropout"]
    best_freeze_backbone = study.best_params["freeze_backbone"]
    best_lr = study.best_params["lr"]
    best_threshold = study.best_params["threshold"]

    # ============================================================
    # Finales Training auf trainval
    # ============================================================

    print("\n================ FINALES TRAINING AUF TRAINVAL ================")

    final_train_dataset_aug = YawDDDataset(
        "trainval",
        num_frames=args.num_frames,
        train=True
    )

    final_train_dataset_eval = YawDDDataset(
        "trainval",
        num_frames=args.num_frames,
        train=False
    )

    final_trainloader = DataLoader(
        final_train_dataset_aug,
        batch_size=best_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    final_train_eval_loader = DataLoader(
        final_train_dataset_eval,
        batch_size=best_batch_size,
        shuffle=False,
        num_workers=0
    )

    final_model = YawDDclassifier(best_dropout).to(device)

    final_state_path = "checkpoints/final_state_dict.pt"
    final_log_dir = f"runs/final_train_{time.strftime('%Y%m%d_%H%M%S')}"

    # Wichtig:
    # Kein Early Stopping auf Trainingsdaten.
    # Wir trainieren die gewählte Epochenzahl vollständig.
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

    final_model.load_state_dict(torch.load(final_state_path, map_location=device))

    # Einheitliches Checkpoint-Format speichern
    final_checkpoint_path = "best_model_final.pt"

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

    print(f"\nFinales Modell gespeichert unter: {final_checkpoint_path}")

    # ============================================================
    # Finaler Test auf Testsplit
    # ============================================================

    print("\n================ FINALER TEST AUF TESTDATEN ================")

    testset = YawDDDataset(
        "test",
        num_frames=args.num_frames,
        train=False
    )

    testloader = DataLoader(
        testset,
        batch_size=best_batch_size,
        shuffle=False,
        num_workers=0
    )

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

    print("\n================ FINAL TEST ================")
    print(f"Test Accuracy : {test_metrics['accuracy']:.3f}")
    print(f"Test Precision: {test_metrics['precision']:.3f}")
    print(f"Test Recall   : {test_metrics['recall']:.3f}")
    print(f"Test F1       : {test_metrics['f1']:.3f}")

    if test_metrics["roc_auc"] == test_metrics["roc_auc"]:
        print(f"Test ROC-AUC  : {test_metrics['roc_auc']:.3f}")
    else:
        print("Test ROC-AUC  : nan")

    if test_metrics["pr_auc"] == test_metrics["pr_auc"]:
        print(f"Test PR-AUC   : {test_metrics['pr_auc']:.3f}")
    else:
        print("Test PR-AUC   : nan")

    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])

    # TensorBoard-HParams für finales Modell
    writer = SummaryWriter(final_log_dir)

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

    writer.close()

    time_passed = time.time() - start_timestamp

    print(
        f"\nTraining finished in "
        f"{time_passed // 3600:.0f}h "
        f"{(time_passed % 3600) // 60:.0f}min "
        f"{time_passed % 60:.0f}s\n"
    )

    if HAS_WINSOUND:
        for _ in range(3):
            winsound.Beep(1000, 500)
