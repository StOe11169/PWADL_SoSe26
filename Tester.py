# Betriebssystemmodul zum Setzen von Umgebungsvariablen
import os
# TensorFlow-/TensorBoard-Info- und Warnmeldungen in der Konsole reduzieren
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# PyTorch wird zum Laden des Modells und zur Geräteverwaltung verwendet
import torch
# DataLoader erstellt Batches aus dem Testdataset
from torch.utils.data import DataLoader
# SummaryWriter ermöglicht das Logging der Testergebnisse in TensorBoard
from torch.utils.tensorboard import SummaryWriter
# time wird zur Erzeugung eindeutiger Logverzeichnisnamen verwendet
import time
# Modellklasse für die Gähn-Erkennung
from src.training import YawDDclassifier
# Dataset-Klasse zum Laden des Testsplit
from src.data import YawDDDataset
# Evaluierungsfunktion zur Berechnung der Testmetriken
from src.evaluation import evaluate


def load_model(checkpoint_path, device):
    """
    Lädt ein gespeichertes Modell inklusive relevanter Hyperparameter.

    Unterstützt:
        1. neues Checkpoint-Format mit Modellgewichten und Hyperparametern
        2. altes Format mit reinem state_dict (alte Version, hatte auch andere Datenstruktur)

    Args:
        checkpoint_path (str): Pfad zum gespeicherten Checkpoint.
        device: Zielgerät für das Modell.

    Returns:
        tuple: Modell, Dropout, Threshold, Anzahl Frames und Batchgröße.
    """

    # Checkpoint vom Datenträger laden und direkt auf das Zielgerät mappen
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Prüfen, ob der Checkpoint dem neuen Dictionary-Format entspricht
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:

        # Dropout-Wert aus dem Checkpoint lesen oder Default verwenden
        dropout = checkpoint.get("dropout", 0.3)

        # Klassifikationsschwellwert aus dem Checkpoint lesen oder Default verwenden
        threshold = checkpoint.get("threshold", 0.3)

        # Anzahl der Frames aus dem Checkpoint lesen oder Default verwenden
        num_frames = checkpoint.get("num_frames", 32)

        # Batchgröße aus dem Checkpoint lesen oder Default verwenden
        batch_size = checkpoint.get("batch_size", 8)

        # Modellgewichte aus dem Checkpoint extrahieren
        state_dict = checkpoint["model_state"]

    # Fallback für ältere Checkpoints, die nur Modellgewichte enthalten
    else:

        # Warnhinweis ausgeben, da Hyperparameter nicht im Checkpoint enthalten sind
        print("WARNUNG: Altes Checkpoint-Format erkannt. Verwende Default-Hyperparameter.")

        # Default-Dropout verwenden
        dropout = 0.3

        # Default-Threshold verwenden
        threshold = 0.3

        # Default-Anzahl an Frames verwenden
        num_frames = 32

        # Default-Batchgröße verwenden
        batch_size = 8

        # Checkpoint direkt als state_dict interpretieren
        state_dict = checkpoint

    # Modell mit dem gespeicherten bzw. gewählten Dropout-Wert initialisieren
    model = YawDDclassifier(dropout).to(device)

    # Geladene Modellgewichte in das Modell einfügen
    model.load_state_dict(state_dict)

    # Modell in den Evaluationsmodus setzen
    model.eval()

    # Modell und relevante Hyperparameter zurückgeben
    return model, dropout, threshold, num_frames, batch_size


def main():
    """
    Lädt das finale Modell und evaluiert es auf dem unabhängigen Testsplit.
    """

    # GPU verwenden, falls verfügbar, sonst CPU nutzen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pfad zum final gespeicherten Modellcheckpoint festlegen
    checkpoint_path = "best_model_final.pt"

    # Modell und gespeicherte Hyperparameter laden
    model, dropout, threshold, num_frames, batch_size = load_model(
        checkpoint_path,
        device
    )

    # Geladene Modellparameter zur Kontrolle ausgeben
    print("\nGeladene Modellparameter:")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Dropout    : {dropout}")
    print(f"Threshold  : {threshold}")
    print(f"Num Frames : {num_frames}")
    print(f"Batch Size : {batch_size}")

    # Testdataset ohne Augmentation initialisieren
    testset = YawDDDataset(
        "test",
        num_frames=num_frames,
        train=False
    )

    # DataLoader für die Testdaten erstellen
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Eindeutiges TensorBoard-Logverzeichnis für diesen Testlauf erzeugen
    log_dir = f"runs/test_results_{time.strftime('%Y%m%d_%H%M%S')}"

    # TensorBoard-Writer für die Testergebnisse initialisieren
    writer = SummaryWriter(log_dir)

    # Modell auf dem Testsplit evaluieren
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

    # Testergebnisse zusätzlich in einer Textdatei speichern
    with open("test_results.txt", "w", encoding="utf-8") as f:

        # Überschrift in die Ergebnisdatei schreiben
        f.write("===== TESTERGEBNISSE =====\n")

        # Accuracy speichern
        f.write(f"Accuracy : {test_metrics['accuracy']:.3f}\n")

        # Precision speichern
        f.write(f"Precision: {test_metrics['precision']:.3f}\n")

        # Recall speichern
        f.write(f"Recall   : {test_metrics['recall']:.3f}\n")

        # F1-Score speichern
        f.write(f"F1 Score : {test_metrics['f1']:.3f}\n")

        # ROC-AUC speichern
        f.write(f"ROC-AUC  : {test_metrics['roc_auc']:.3f}\n")

        # PR-AUC speichern
        f.write(f"PR-AUC   : {test_metrics['pr_auc']:.3f}\n")

        # Verwendeten Threshold speichern
        f.write(f"Threshold: {threshold:.3f}\n")

        # Verwendeten Dropout-Wert speichern
        f.write(f"Dropout  : {dropout}\n")

        # Verwendete Frameanzahl speichern
        f.write(f"NumFrames: {num_frames}\n")

        # Verwendete Batchgröße speichern
        f.write(f"BatchSize: {batch_size}\n")

        # Überschrift für die Confusion Matrix speichern
        f.write("\nConfusion Matrix:\n")

        # Confusion Matrix speichern
        f.write(str(test_metrics["confusion_matrix"]))

    # Hyperparameter und Testmetriken in TensorBoard speichern
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

    # TensorBoard-Writer sauber schließen
    writer.close()

    # Testergebnisse in der Konsole ausgeben
    print("\n===== TESTERGEBNISSE =====")

    # Accuracy ausgeben
    print(f"Accuracy : {test_metrics['accuracy']:.3f}")

    # Precision ausgeben
    print(f"Precision: {test_metrics['precision']:.3f}")

    # Recall ausgeben
    print(f"Recall   : {test_metrics['recall']:.3f}")

    # F1-Score ausgeben
    print(f"F1 Score : {test_metrics['f1']:.3f}")

    # ROC-AUC nur formatiert ausgeben, wenn der Wert definiert ist
    if test_metrics["roc_auc"] == test_metrics["roc_auc"]:
        print(f"ROC-AUC  : {test_metrics['roc_auc']:.3f}")
    else:
        print("ROC-AUC  : nan")

    # PR-AUC nur formatiert ausgeben, wenn der Wert definiert ist
    if test_metrics["pr_auc"] == test_metrics["pr_auc"]:
        print(f"PR-AUC   : {test_metrics['pr_auc']:.3f}")
    else:
        print("PR-AUC   : nan")

    # Verwendeten Threshold ausgeben
    print(f"Threshold: {threshold:.3f}")

    # Überschrift für die Confusion Matrix ausgeben
    print("Confusion Matrix:")

    # Confusion Matrix ausgeben
    print(test_metrics["confusion_matrix"])


# Hauptfunktion nur ausführen, wenn die Datei direkt gestartet wird
if __name__ == "__main__":

    # Testablauf starten
    main()