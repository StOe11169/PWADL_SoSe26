# Bestes Modell soll wieder aufgerufen werden, um es auf unterschiedlichen Daten zu testen.
# Testet hier noch das alte Modell, was Threshold und Dropout noch manuell als Input braucht

import torch
from torch.utils.data import DataLoader
from src.training import YawDDclassifier
from src.data import YawDDDataset
from src.evaluation import evaluate

def load_model(checkpoint_path, device):
    #Lädt Modell + Hyperparameter aus gespeicherter Datei
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Fallback für alte Dateien (ohne Hyperparameter)
    dropout = checkpoint.get("dropout", 0.3)
    threshold = checkpoint.get("threshold", 0.3)

    model = YawDDclassifier(dropout).to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()

    return model, dropout, threshold


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modell + Hyperparameter laden
    model, dropout, threshold = load_model("best_model.pt", device)


    # ===== DATEN LADEN =====
    batch_size = 8
    num_frames = 32
    testset = YawDDDataset('test', num_frames=num_frames, train=False)
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # ===== EVALUATION =====
    metrics = evaluate(testloader, model, device, threshold)

    print("\n===== TESTERGEBNISSE =====")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    print(f"Verwendeter Threshold: {threshold:.3f}")  # Zur Kontrolle

if __name__ == "__main__":
    main()