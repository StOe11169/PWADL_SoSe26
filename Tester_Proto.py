# Bestes Modell soll wieder aufgerufen werden, um es auf unterschiedlichen Daten zu testen.
# Testet hier noch das alte Modell, was Threshold und Dropout noch manuell als Input braucht

import torch
from torch.utils.data import DataLoader
from src.training import YawDDclassifier
from src.data import YawDDDataset
from src.evaluation import evaluate
from torch.utils.tensorboard import SummaryWriter  # Für TensorBoard-Logging
import time

def load_model(checkpoint_path, device):
    """
    Lädt Modell + Hyperparameter aus der gespeicherten Datei.
    Unterstützt sowohl neue Format (mit Hyperparametern) als auch alte Format (nur state_dict).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extrahiere Hyperparameter oder verwende Default-Werte
    dropout = checkpoint.get("dropout", 0.3)
    threshold = checkpoint.get("threshold", 0.3)

    # Modell initialisieren und Gewichte laden
    model = YawDDclassifier(dropout).to(device)
    model.load_state_dict(checkpoint["model_state"])  #Nur die Gewichte laden
    model.eval()

    return model, dropout, threshold

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modell + Hyperparameter laden
    model, dropout, threshold = load_model("best_model.pt", device)

    # ===== DATEN LADEN =====
    batch_size = 8
    num_frames = 4  #WIE IN TRAINING SETZEN
    testset = YawDDDataset('test', num_frames=num_frames, train=False)
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # ===== EVALUATION MIT TENSORBOARD =====
    # Logge in separates TensorBoard-Verzeichnis für den Tester
    log_dir = f"runs/test_results_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    # Evaluation mit TensorBoard-Logging
    test_metrics = evaluate(
        testloader,
        model,
        device,
        threshold,
        writer=writer,  # TensorBoard-Logging aktivieren
        epoch=0         # Wird im evaluate() für TensorBoard verwendet
    )

    # Speichere Test-Metriken für spätere Analyse
    with open("test_results.txt", "w") as f:
        f.write(f"Test Accuracy: {test_metrics['accuracy']:.3f}\n")
        f.write(f"Test F1 Score: {test_metrics['f1']:.3f}\n")
        f.write(f"Verwendeter Threshold: {threshold:.3f}\n")

    # ===== TENSORBOARD HYPERPARAMETER-LOGGING =====
    writer.add_hparams(
        hparam_dict={
            "batch_size": batch_size,
            "num_frames": num_frames,
            "threshold": threshold,
            "dropout": dropout,
        },
        metric_dict={
            "Test_Accuracy": test_metrics['accuracy'],
            "Test_F1": test_metrics['f1'],
        },
        run_name="test_run"
    )

    # Schließe TensorBoard Writer
    writer.close()

    # ===== ERGEBNISSE AUSGEBEN =====
    print("\n===== TESTERGEBNISSE =====")
    print(f"Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"F1 Score: {test_metrics['f1']:.3f}")
    print(f"Precision: {test_metrics['precision']:.3f}")
    print(f"Recall: {test_metrics['recall']:.3f}")
    print(f"Verwendeter Threshold: {threshold:.3f}")

if __name__ == "__main__":
    main()