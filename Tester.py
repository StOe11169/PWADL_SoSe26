# Bestes Modell soll wieder aufgerufen werden, um es auf unterschiedlichen Daten zu testen.

import torch
from torch.utils.data import DataLoader
from src.training import YawDDclassifier
from src.data import YawDDDataset
from src.evaluation import evaluate

def load_model(checkpoint_path, device):
    # Checkpoint laden
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    """
    # Parameter aus Checkpoint holen, nach neuem Training machbar
    dropout = checkpoint["dropout"]
    threshold = checkpoint["threshold"]
    """

    # Modell korrekt rekonstruieren
    model = YawDDclassifier(dropout).to(device)
    model.load_state_dict(checkpoint["model_state"])

    model.eval()  # GANZ wichtig für Inferenz

    return model, threshold


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Modell laden
    model, threshold = load_model("best_model.pt", device)

    # Parameter festlegen
    batch_size = 8
    num_frames = 32
    #Nach nächstem Training überflüssig
    dropout = 0.3          # <- MUSS zu deinem best_params passen
    threshold = 0.3

    # ===== DATEN LADEN =====
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