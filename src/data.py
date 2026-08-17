import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchcodec.decoders import VideoDecoder # Bibliothek zum Decodieren von Video-Dateien


def get_image_paths(split):
    """
    Liest alle Videodateien aus einem Datensatz-Ordner (train/val/test) und extrahiert Pfade sowie Labels.

    Args:
        split (str): Name des Datensatz-Teils ('train', 'val' oder 'test')

    Returns:
        pd.DataFrame: DataFrame mit Spalten ['id', 'info_labels', 'activity', 'filepath', 'yawning']
                     - 'filepath': Vollständiger Pfad zur Videodatei
                     - 'yawning': Binäres Label (1.0 = Gähnen, 0.0 = Kein Gähnen)
    """
    file_paths = [] # Liste zum Speichern der Videopfade
    file_names = [] # Liste zum Speichern der Dateinamen (ohne Endung)

    # Angegebenen Ordner rekursiv nach Videodateien durchsuchen
    folder_path = os.path.join("data", split)
    for dirpath, _, filenames in os.walk(folder_path):
            for fname in filenames:
                # Nur Video-Dateien verarbeiten (MP4, AVI, MOV)
                if fname.lower().endswith((".mp4", ".avi", ".mov")):
                    file_paths.append(dirpath+'/'+fname) # Vollständiger Pfad zur Datei
                    file_names.append(os.path.splitext(fname)[0]) # Dateiname ohne Endung extrahieren

    # Dateinamen im Format "ID-info_labels-activity" parsen
    df = pd.DataFrame([fn.split('-') for fn in file_names], columns=['id', 'info_labels', 'activity'])

    # Vollständige Pfade und Labels zum DataFrame hinzufügen
    df['filepath'] = file_paths
    # Label erstellen: 1.0 wenn 'yawning' im activity-Feld vorkommt (case-insensitive), sonst 0.0
    df['yawning'] = [1.0 if 'yawning' in g.lower() else 0.0 for g in df['activity']]

    return df


def load_images_from_path(file_path, num_frames):
    """
    Lädt eine bestimmte Anzahl von Frames aus einer Videodatei.

    Args:
        file_path (str): Pfad zur Videodatei
        num_frames (int): Anzahl der zu extrahierenden Frames

    Returns:
        torch.Tensor: Tensor mit Form (num_frames, C, H, W) - normalisierte Bilddaten
    """
    # Video-Decoder initialisieren
    decoder = VideoDecoder(file_path)

    # Gleichmäßig verteilte Frame-Indizes berechnen
    indices = torch.linspace(0, decoder.metadata.num_frames - 1, num_frames).long()

    # Frames an den berechneten Positionen extrahieren
    # .data entfernt Metadaten und gibt nur den Tensor zurück
    return decoder.get_frames_at(indices=list(indices)).data


class YawDDDataset(Dataset):
    """
    PyTorch-Dataset-Klasse für den YawDD-Datensatz (Yawning Detection Dataset).
    Verantwortlich für das Laden, Transformieren und Zurückgeben von Videosequenzen.

    Args:
        split (str): Datensatz-Teil ('train', 'val' oder 'test')
        num_frames (int): Anzahl der Frames pro Videosequenz
        train (bool): True für Trainingsdaten (mit Augmentierung), False für Validierung/Test (ohne Augmentierung)
    """
    def __init__(self, split, num_frames, train=True):
        # Pfade und Labels aus dem Datensatz laden
        df_image_paths = get_image_paths(split)

        self.image_paths = df_image_paths['filepath'].tolist() # Liste aller Videopfade
        self.labels = df_image_paths['yawning'].tolist() # Liste aller Labels (1.0/0.0)
        self.train = train # Flag, ob Trainings- oder Validierungs-/Test-Daten geladen werden

        
        # ===== Daten-Transformationen definieren =====
        # Transformations-Pipeline für Trainingsdaten (mit Augmentierung zur Verbesserung der Generalisierung)
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(), # Konvertiere Tensor zu PIL-Image für Transformationen
            transforms.Resize((256, 341)), # Größe anpassen (Breite:Höhe = 4:3)

            # Datenaugmentierung zur Verbesserung der Robustheit des Modells
            transforms.RandomHorizontalFlip(p=0.5), # 50% Wahrscheinlichkeit für horizontale Spiegelung
            transforms.RandomRotation(10), # Zufällige Rotation bis ±10 Grad
            transforms.ColorJitter( # Zufällige Farbveränderungen zur Robustheit gegenüber Beleuchtung
                brightness=0.2, # Helligkeit um ±20% anpassen
                contrast=0.2, # Kontrast um ±20% anpassen
                saturation=0.2, # Sättigung um ±20% anpassen
                hue=0.05 # Farbton leicht variieren
            ),

            transforms.CenterCrop(224), # Zentrale Region ausschneiden (224x224)
            transforms.ToTensor(), # Konvertiere PIL-Image zurück zu PyTorch-Tensor (C,H,W)
            # Normalisierung mit ImageNet-Standardwerten für bessere Konvergenz
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], # Mittelwerte der RGB-Kanäle (ImageNet)
                std=[0.229, 0.224, 0.225] # Standardabweichungen der RGB-Kanäle (ImageNet)
            )
        ])
        
        
        
        # Transformations-Pipeline für Validierungs- und Testdaten (ohne Augmentierung, nur Vorverarbeitung)
        self.val_transform = transforms.Compose([
             transforms.ToPILImage(), # Konvertiere Tensor zu PIL-Image
             transforms.Resize((256, 341)), # Größe anpassen (wie in train_transform)
             transforms.CenterCrop(224), # Zentrale Region ausschneiden (wie in train_transform)
             transforms.ToTensor(), # Konvertiere zu Tensor (C,H,W)
             # Normalisierung mit denselben Werten wie in train_transform für Konsistenz
             transforms.Normalize(
                  mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]
             )
        ])

        self.num_frames = num_frames # Anzahl der Frames pro Videosequenz speichern

    def __len__(self):
        """
        Gibt die Anzahl der Videosequenzen im Datensatz zurück.
        Wird von PyTorch für die Iteration über den Datensatz benötigt.
        """
        return len(self.labels)

    def __getitem__(self, idx): 
        """
        Lädt und transformiert eine Videosequenz an Position idx.
        Wird von PyTorch bei jedem Zugriff auf den Datensatz aufgerufen.

        Args:
            idx (int): Index der zu ladenden Videosequenz

        Returns:
            tuple: (torch.Tensor, torch.Tensor) - Tensor mit Form (num_frames, C, H, W) und zugehöriges Label
        """
        # Videosequenz mit der angegebenen Anzahl von Frames laden
        # image_sequence hat Form (num_frames, H, W, C)
        image_sequence = load_images_from_path(self.image_paths[idx], num_frames=self.num_frames)
        # Daten je nach Modus (Trainings- oder Validierungs-/Test-Daten) transformieren
        if self.train:
            # Trainingsdaten: Zufällige Augmentierungen anwenden
            images = [self.train_transform(frame) for frame in image_sequence]
        else:
            # Validierungs-/Testdaten: Nur Vorverarbeitung (keine Augmentierung)
            images = [self.val_transform(frame) for frame in image_sequence]
        

        # Label als Tensor zurückgeben (binär: 0 oder 1)
        label = self.labels[idx]

        return torch.stack(images), torch.tensor(label)