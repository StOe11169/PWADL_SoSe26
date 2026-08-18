import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # TensorFlow soll INFO- und WARNING-Meldungen nicht anzeigen.
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

from src.training import YawDDclassifier
from src.data import YawDDDataset
from src.evaluation import evaluate


def load_model(checkpoint_path, device):
    """
    Lädt Modell und Hyperparameter.

    Unterstützt:
    1. neues Format:
        {
            "model_state": ...,
            "dropout": ...,
            "threshold": ...,
            "num_frames": ...
        }

    2. altes Format:
        reines state_dict
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        dropout = checkpoint.get("dropout", 0.3)
        threshold = checkpoint.get("threshold", 0.3)
        num_frames = checkpoint.get("num_frames", 32)
        batch_size = checkpoint.get("batch_size", 8)
        state_dict = checkpoint["model_state"]

    else:
        print("WARNUNG: Altes Checkpoint-Format erkannt. Verwende Default-Hyperparameter.")
        dropout = 0.3
        threshold = 0.3
        num_frames = 32
        batch_size = 8
        state_dict = checkpoint

    model = YawDDclassifier(dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, dropout, threshold, num_frames, batch_size


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = "best_model_final.pt"

    model, dropout, threshold, num_frames, batch_size = load_model(
        checkpoint_path,
        device
    )

    print("\nGeladene Modellparameter:")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Dropout    : {dropout}")
    print(f"Threshold  : {threshold}")
    print(f"Num Frames : {num_frames}")
    print(f"Batch Size : {batch_size}")

    testset = YawDDDataset(
        "test",
        num_frames=num_frames,
        train=False
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    log_dir = f"runs/test_results_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    test_metrics = evaluate(
        testloader,
        model,
        device,
        threshold,
        writer=writer,
        epoch=0,
        prefix="test",
        verbose=True
    )

    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write("===== TESTERGEBNISSE =====\n")
        f.write(f"Accuracy : {test_metrics['accuracy']:.3f}\n")
        f.write(f"Precision: {test_metrics['precision']:.3f}\n")
        f.write(f"Recall   : {test_metrics['recall']:.3f}\n")
        f.write(f"F1 Score : {test_metrics['f1']:.3f}\n")
        f.write(f"ROC-AUC  : {test_metrics['roc_auc']:.3f}\n")
        f.write(f"PR-AUC   : {test_metrics['pr_auc']:.3f}\n")
        f.write(f"Threshold: {threshold:.3f}\n")
        f.write(f"Dropout  : {dropout}\n")
        f.write(f"NumFrames: {num_frames}\n")
        f.write(f"BatchSize: {batch_size}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(test_metrics["confusion_matrix"]))

    writer.add_hparams(
        hparam_dict={
            "batch_size": batch_size,
            "num_frames": num_frames,
            "threshold": threshold,
            "dropout": dropout,
        },
        metric_dict={
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
        },
        run_name="test_run"
    )

    writer.close()

    print("\n===== TESTERGEBNISSE =====")
    print(f"Accuracy : {test_metrics['accuracy']:.3f}")
    print(f"Precision: {test_metrics['precision']:.3f}")
    print(f"Recall   : {test_metrics['recall']:.3f}")
    print(f"F1 Score : {test_metrics['f1']:.3f}")

    if test_metrics["roc_auc"] == test_metrics["roc_auc"]:
        print(f"ROC-AUC  : {test_metrics['roc_auc']:.3f}")
    else:
        print("ROC-AUC  : nan")

    if test_metrics["pr_auc"] == test_metrics["pr_auc"]:
        print(f"PR-AUC   : {test_metrics['pr_auc']:.3f}")
    else:
        print("PR-AUC   : nan")

    print(f"Threshold: {threshold:.3f}")
    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()