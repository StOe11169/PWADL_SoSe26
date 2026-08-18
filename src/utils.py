# Betriebssystemmodul zum Setzen von Umgebungsvariablen
import os
# PyTorch wird für Seeds und numerische Einstellungen verwendet
import torch
# NumPy wird für reproduzierbare numerische Operationen initialisiert
import numpy as np
# random steuert den Python-internen Zufallsgenerator
import random


def setup_env(seed):
    """
    Setzt Zufallsseeds und relevante PyTorch-Einstellungen für reproduzierbarere Experimente.

    Args:
        seed (int): Seed-Wert für Python, NumPy und PyTorch.
    """

    # Python-internen Zufallsgenerator initialisieren
    random.seed(seed)

    # NumPy-Zufallsgenerator initialisieren
    np.random.seed(seed)

    # PyTorch-Zufallsgenerator für CPU-Operationen initialisieren
    torch.manual_seed(seed)

    # Hash-Seed setzen, um die Reproduzierbarkeit bestimmter Python-Operationen zu verbessern
    os.environ["PYTHONHASHSEED"] = str(seed)

    # CUDA-spezifische Einstellungen nur setzen, wenn eine GPU verfügbar ist
    if torch.cuda.is_available():

        # PyTorch-Zufallsgenerator für die aktuelle CUDA-GPU initialisieren
        torch.cuda.manual_seed(seed)

        # PyTorch-Zufallsgeneratoren für alle CUDA-GPUs initialisieren
        torch.cuda.manual_seed_all(seed)

        # Deterministische CuDNN-Operationen bevorzugen
        torch.backends.cudnn.deterministic = True

        # CuDNN-Benchmark deaktivieren, um nichtdeterministische Algorithmuswahl zu vermeiden
        torch.backends.cudnn.benchmark = False

    # Höhere Genauigkeit für Float32-Matrixmultiplikationen erlauben
    torch.set_float32_matmul_precision("high")     