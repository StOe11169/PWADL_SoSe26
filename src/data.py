# Betriebssystemfunktionen für Pfade, Ordnerprüfung und Dateinamen
import os
# Zufallsfunktionen für reproduzierbare bzw. konsistente Datenaugmentationen
import random
# Pandas wird für tabellarische Metadaten und die Split-Datei verwendet
import pandas as pd
# NumPy wird für numerische Hilfsarrays bei den Split-Operationen verwendet
import numpy as np
# PyTorch wird für Tensoren und gleichmäßige Frame-Indizes verwendet
import torch
# Dataset ist die Basisklasse für eigene PyTorch-Datensätze
from torch.utils.data import Dataset, Subset, ConcatDataset
# Funktionale torchvision-Transformationen für PIL-Bilder und Tensoren
import torchvision.transforms.functional as TF
# VideoDecoder liest Videodateien und extrahiert einzelne Frames
from torchcodec.decoders import VideoDecoder

# Standardordner, in dem alle Videodateien gespeichert werden
DEFAULT_VIDEO_DIR = os.path.join("data", "videos")
# Standardpfad für die automatisch erzeugte Split-Datei
DEFAULT_SPLIT_CSV = os.path.join("data", "splits.csv")


def scan_video_folder(video_dir=DEFAULT_VIDEO_DIR):
    """
    Scannt den Videoordner rekursiv nach Videodateien und erstellt eine Metadatentabelle.

    Erwartetes Dateinamenformat:
        ID-info_labels-activity.mp4/avi/mov

    Beispiele:
        001-Male-yawning.mp4
        001-Male-normal.mp4
        150-eigenesVideo-yawning.mp4

    Args:
        video_dir (str): Ordner, in dem die Videodateien liegen.

    Returns:
        pd.DataFrame: Tabelle mit Dateipfaden, Labels und Metadaten.
    """

    # Prüfen, ob der angegebene Videoordner existiert
    if not os.path.exists(video_dir):
        raise FileNotFoundError(
            f"Videoordner nicht gefunden: {video_dir}\n"
            f"Bitte Videos unter data/videos/ ablegen."
        )

    # Liste für vollständige Dateipfade initialisieren
    file_paths = []
    # Liste für Dateinamen ohne Dateiendung initialisieren
    file_names = []

    # Rekursiv durch den Videoordner und mögliche Unterordner laufen
    for dirpath, dirnames, filenames in os.walk(video_dir):
        # Unterordner sortieren, damit die Reihenfolge reproduzierbar bleibt
        dirnames.sort()

        # Dateinamen sortieren, damit die spätere Dataset-Reihenfolge stabil ist
        for fname in sorted(filenames):
            # Nur unterstützte Videodateien berücksichtigen
            if fname.lower().endswith((".mp4", ".avi", ".mov")):
                # Vollständigen und normalisierten Dateipfad erstellen
                full_path = os.path.normpath(os.path.join(dirpath, fname))
                # Dateiendung entfernen, damit der Name geparst werden kann
                name_without_ext = os.path.splitext(fname)[0]

                # Dateipfad speichern
                file_paths.append(full_path)
                # Dateiname ohne Endung speichern
                file_names.append(name_without_ext)

    # Fehler auslösen, wenn keine Videodateien gefunden wurden
    if len(file_paths) == 0:
        raise FileNotFoundError(f"Keine Videos in {video_dir} gefunden.")

    # Liste für die späteren Tabellenzeilen initialisieren
    rows = []

    # Dateinamen und Pfade gemeinsam verarbeiten
    for fn, fp in zip(file_names, file_paths):

        # Dateiname in ID, Zusatzinformationen und Aktivität zerlegen
        parts = fn.split("-", maxsplit=2)

        # Sicherstellen, dass das erwartete Dateinamenformat eingehalten wurde
        if len(parts) != 3:
            raise ValueError(
                f"Dateiname hat nicht das erwartete Format "
                f"'ID-info_labels-activity': {fn}"
            )

        # Die drei Bestandteile des Dateinamens entpacken
        sample_id, info_labels, activity = parts

        # Binäres Label aus der Aktivität ableiten
        yawning = 1.0 if "yawning" in activity.lower() else 0.0

        # Metadaten des Videos als Dictionary speichern
        rows.append(
            {
                "id": str(sample_id),
                "info_labels": info_labels,
                "activity": activity,
                "filepath": fp,
                "yawning": yawning,
                "filename": os.path.basename(fp),
            }
        )

    # Metadaten in einen DataFrame umwandeln
    df = pd.DataFrame(rows)

    # DataFrame sortieren, damit die Reihenfolge unabhängig vom Dateisystem ist
    df = df.sort_values(["id", "filename"]).reset_index(drop=True)

    # Fertige Metadatentabelle zurückgeben
    return df


def _safe_group_split(df, test_size, seed, split_name):
    """
    Erstellt einen gruppierten Split mit möglichst ähnlicher Klassenverteilung.

    Ziel:
        - gleiche IDs dürfen nicht in beiden Teilmengen vorkommen
        - der Anteil der positiven Klasse soll möglichst ähnlich bleiben

    Args:
        df (pd.DataFrame): Metadatentabelle.
        test_size (float): Anteil der zweiten Teilmenge.
        seed (int): Zufallsseed für reproduzierbare Splits.
        split_name (str): Name des Splits für Konsolenausgaben.

    Returns:
        tuple: Indizes für Trainings- und Test-/Validierungsanteil.
    """

    # Import innerhalb der Funktion, da diese Methode nur beim Splitten benötigt wird
    from sklearn.model_selection import GroupShuffleSplit

    # Labels als Integer-Array extrahieren
    y = df["yawning"].astype(int).to_numpy()
    # IDs als Gruppenvariable extrahieren
    groups = df["id"].astype(str).to_numpy()
    # Dummy-Featurematrix erstellen, da GroupShuffleSplit ein X-Argument erwartet
    X = np.zeros(len(df))

    # Positiven Klassenanteil im gesamten betrachteten Datensatz berechnen
    overall_pos_rate = y.mean()

    # Platzhalter für den besten Trainingsindex initialisieren
    best_train_idx = None
    # Platzhalter für den besten Test-/Validierungsindex initialisieren
    best_test_idx = None
    # Startwert für den besten Balance-Score setzen
    best_score = float("inf")

    # Anzahl der zufälligen Split-Kandidaten festlegen
    n_candidates = 200

    # Mehrere gruppierte Zufallssplits erzeugen und den besten auswählen
    for i in range(n_candidates):
        # GroupShuffleSplit erzeugt Splits ohne ID-Überlappung
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed + i
        )

        # Einen Split-Kandidaten erzeugen
        train_idx, test_idx = next(splitter.split(X, y, groups))

        # Labels der ersten Teilmenge extrahieren
        y_train = y[train_idx]
        # Labels der zweiten Teilmenge extrahieren
        y_test = y[test_idx]

        # Positiven Klassenanteil der ersten Teilmenge berechnen
        train_pos_rate = y_train.mean()
        # Positiven Klassenanteil der zweiten Teilmenge berechnen
        test_pos_rate = y_test.mean()

        # Abweichung beider Teilmengen vom Gesamtanteil berechnen
        score = abs(train_pos_rate - overall_pos_rate) + abs(test_pos_rate - overall_pos_rate)

        # Bisher besten Split aktualisieren, falls der aktuelle Kandidat ausgewogener ist
        if score < best_score:
            best_score = score
            best_train_idx = train_idx
            best_test_idx = test_idx

    # Gewählten Split und Balance-Score in der Konsole ausgeben
    print(
        f"{split_name}: GroupShuffleSplit mit {n_candidates} Kandidaten verwendet. "
        f"Balance-Score={best_score:.4f}"
    )

    # Beste gefundene Indizes zurückgeben
    return best_train_idx, best_test_idx



def create_split_csv(
    video_dir=DEFAULT_VIDEO_DIR,
    split_csv=DEFAULT_SPLIT_CSV,
    test_size=0.20,
    val_size=0.20,
    seed=0,
    force=False
):
    """
    Erstellt eine Split-Datei mit train-, val- und test-Zuweisungen.

    Zusätzlich kann später der kombinierte Split trainval verwendet werden.

    Args:
        video_dir (str): Ordner mit allen Videos.
        split_csv (str): Zielpfad der Split-Datei.
        test_size (float): Anteil des Testsets am Gesamtdatensatz.
        val_size (float): Anteil des Validierungssets am Gesamtdatensatz.
        seed (int): Zufallsseed für reproduzierbare Splits.
        force (bool): Wenn True, wird eine bestehende Split-Datei überschrieben.

    Returns:
        pd.DataFrame: DataFrame mit Split-Zuweisungen.
    """

    # Zielordner für die Split-Datei erstellen, falls er noch nicht existiert
    os.makedirs(os.path.dirname(split_csv), exist_ok=True)

    # Aktuellen Videobestand aus dem Videoordner einlesen
    current_df = scan_video_folder(video_dir)

    # Bestehende Split-Datei verwenden, wenn sie existiert und nicht überschrieben werden soll
    if os.path.exists(split_csv) and not force:
        # Bestehende Split-Datei laden
        existing_df = pd.read_csv(split_csv)

        # Aktuelle Videopfade normalisieren und als Menge speichern
        current_paths = set(os.path.normpath(p) for p in current_df["filepath"].tolist())
        # Videopfade aus der bestehenden Split-Datei normalisieren und als Menge speichern
        existing_paths = set(os.path.normpath(p) for p in existing_df["filepath"].tolist())

        # Prüfen, ob Videobestand und Split-Datei noch zusammenpassen
        if current_paths != existing_paths:
            # Videos bestimmen, die neu im Ordner liegen, aber nicht in der CSV stehen
            missing_in_csv = current_paths - existing_paths
            # CSV-Einträge bestimmen, deren Datei nicht mehr auf dem Datenträger existiert
            missing_on_disk = existing_paths - current_paths

            # Aussagekräftigen Fehler auslösen, damit der Split nicht unbemerkt inkonsistent ist
            raise ValueError(
                "\nDie vorhandene split_csv passt nicht mehr zum Videoordner.\n"
                f"Split-Datei: {split_csv}\n\n"
                f"Neue Videos ohne Split-Eintrag: {len(missing_in_csv)}\n"
                f"CSV-Einträge ohne Datei auf Disk: {len(missing_on_disk)}\n\n"
                "Wenn neue Videos hinzugefügt wurden, data/splits.csv löschen"
                "oder create_split_csv(..., force=True) aufrufen.\n"
            )

        # Bestehende und gültige Split-Datei weiterverwenden
        print(f"Bestehende Split-Datei wird verwendet: {split_csv}")
        # Bestehenden Split zurückgeben
        return existing_df

    # Kopie der aktuellen Metadaten für die Split-Erzeugung verwenden
    df = current_df.copy()
    # Split-Spalte zunächst mit Platzhalterwert initialisieren
    df["split"] = "unset"

    # Ersten gruppierten Split in trainval und test erzeugen
    trainval_idx, test_idx = _safe_group_split(
        df,
        test_size=test_size,
        seed=seed,
        split_name="TrainVal/Test"
    )

    # Test-Indizes im DataFrame markieren
    df.loc[test_idx, "split"] = "test"

    # Teilmenge für trainval erzeugen und Originalindizes erhalten
    trainval_df = df.iloc[trainval_idx].copy().reset_index()
    
    # Relative Validierungsgröße bezogen auf den trainval-Anteil berechnen
    relative_val_size = val_size / (1.0 - test_size)

    # Zweiten gruppierten Split innerhalb von trainval in train und val erzeugen
    inner_train_idx, inner_val_idx = _safe_group_split(
        trainval_df,
        test_size=relative_val_size,
        seed=seed + 1,
        split_name="Train/Val"
    )

    # Originalindizes der Trainingssamples aus dem trainval-DataFrame zurückholen
    original_train_indices = trainval_df.iloc[inner_train_idx]["index"].to_numpy()
    # Originalindizes der Validierungssamples aus dem trainval-DataFrame zurückholen
    original_val_indices = trainval_df.iloc[inner_val_idx]["index"].to_numpy()

    # Trainingssamples im ursprünglichen DataFrame markieren
    df.loc[original_train_indices, "split"] = "train"
    # Validierungssamples im ursprünglichen DataFrame markieren
    df.loc[original_val_indices, "split"] = "val"

    # Sicherstellen, dass jedes Sample genau einem Split zugeordnet wurde
    if (df["split"] == "unset").any():
        raise RuntimeError("Einige Samples wurden keinem Split zugewiesen.")

    # Split-Datei ohne zusätzlichen Index speichern
    df.to_csv(split_csv, index=False)

    # Pfad der erzeugten Split-Datei ausgeben
    print(f"\nSplit-Datei erstellt: {split_csv}")
    # Split-Statistik zur Kontrolle ausgeben
    print_split_statistics(df)

    # DataFrame mit Split-Zuweisungen zurückgeben
    return df


def load_split_csv(
    video_dir=DEFAULT_VIDEO_DIR,
    split_csv=DEFAULT_SPLIT_CSV,
    test_size=0.20,
    val_size=0.20,
    seed=0
):
    """
    Lädt die Split-Datei oder erstellt sie automatisch, falls sie noch nicht existiert.

    Args:
        video_dir (str): Ordner mit allen Videos.
        split_csv (str): Pfad zur Split-Datei.
        test_size (float): Testanteil, falls eine neue Split-Datei erzeugt wird.
        val_size (float): Validierungsanteil, falls eine neue Split-Datei erzeugt wird.
        seed (int): Zufallsseed für eine neue Split-Datei.

    Returns:
        pd.DataFrame: Geladene oder neu erzeugte Split-Tabelle.
    """

    # Split-Datei erzeugen, falls sie noch nicht existiert
    if not os.path.exists(split_csv):
        return create_split_csv(
            video_dir=video_dir,
            split_csv=split_csv,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
            force=True
        )

    # Bestehende Split-Datei laden
    df = pd.read_csv(split_csv)
    # Dateipfade normalisieren, um plattformabhängige Pfadprobleme zu vermeiden
    df["filepath"] = df["filepath"].apply(os.path.normpath)

    # Geladene Split-Tabelle zurückgeben
    return df


def print_split_statistics(df):
    """
    Gibt Klassenverteilungen und ID-Überlappungen je Split aus.

    Args:
        df (pd.DataFrame): Split-Tabelle mit Labels und Split-Zuordnung.
    """

    # Überschrift für die Split-Statistik ausgeben
    print("\nSplit-Statistik:")

    # Die drei Hauptsplits nacheinander auswerten
    for split in ["train", "val", "test"]:
        # Teilmenge für den aktuellen Split auswählen
        part = df[df["split"] == split]

        # Leere Splits separat behandeln
        if len(part) == 0:
            print(f"{split}: leer")
            continue

        # Gesamtzahl der Samples im Split berechnen
        num_total = len(part)
        # Anzahl positiver Samples berechnen
        num_pos = int(part["yawning"].sum())
        # Anzahl negativer Samples berechnen
        num_neg = num_total - num_pos
        # Anzahl eindeutiger IDs im Split berechnen
        num_ids = part["id"].nunique()

        # Statistik formatiert ausgeben
        print(
            f"{split:5s}: total={num_total:4d}, "
            f"yawning={num_pos:4d}, "
            f"non-yawning={num_neg:4d}, "
            f"ids={num_ids:4d}"
        )

    # IDs des Trainingssplits als Menge speichern
    train_ids = set(df[df["split"] == "train"]["id"].astype(str))
    # IDs des Validierungssplits als Menge speichern
    val_ids = set(df[df["split"] == "val"]["id"].astype(str))
    # IDs des Testsplits als Menge speichern
    test_ids = set(df[df["split"] == "test"]["id"].astype(str))

    # Schnittmenge zwischen train und val berechnen
    overlap_train_val = train_ids.intersection(val_ids)
    # Schnittmenge zwischen train und test berechnen
    overlap_train_test = train_ids.intersection(test_ids)
    # Schnittmenge zwischen val und test berechnen
    overlap_val_test = val_ids.intersection(test_ids)

    # Überschrift für die Overlap-Prüfung ausgeben
    print("\nID-Overlap-Prüfung:")
    # Anzahl gemeinsamer IDs zwischen train und val ausgeben
    print(f"train ∩ val : {len(overlap_train_val)}")
    # Anzahl gemeinsamer IDs zwischen train und test ausgeben
    print(f"train ∩ test: {len(overlap_train_test)}")
    # Anzahl gemeinsamer IDs zwischen val und test ausgeben
    print(f"val ∩ test  : {len(overlap_val_test)}")


def get_metadata_for_split(
    split,
    video_dir=DEFAULT_VIDEO_DIR,
    split_csv=DEFAULT_SPLIT_CSV
):
    """
    Gibt die Metadaten für einen gewünschten Split zurück.

    Unterstützte Werte:
        train
        val
        test
        trainval
        all

    Args:
        split (str): Gewünschter Split.
        video_dir (str): Ordner mit allen Videos.
        split_csv (str): Pfad zur Split-Datei.

    Returns:
        pd.DataFrame: Metadaten des gewünschten Splits.
    """

    # Split-Datei laden oder bei Bedarf erzeugen
    df = load_split_csv(video_dir=video_dir, split_csv=split_csv)

    # Alle Daten zurückgeben, falls split="all" gewählt wurde
    if split == "all":
        out = df.copy()

    # Trainings- und Validierungsdaten gemeinsam zurückgeben
    elif split == "trainval":
        out = df[df["split"].isin(["train", "val"])].copy()

    # Einen der drei Hauptsplits zurückgeben
    elif split in ["train", "val", "test"]:
        out = df[df["split"] == split].copy()

    # Ungültige Split-Namen mit klarer Fehlermeldung abfangen
    else:
        raise ValueError(
            f"Unbekannter Split: {split}. "
            f"Erlaubt sind: train, val, test, trainval, all"
        )

    # Index zurücksetzen, damit spätere Subsets konsistent adressiert werden
    out = out.reset_index(drop=True)

    # Fehler auslösen, falls der gewählte Split leer ist
    if len(out) == 0:
        raise ValueError(f"Split '{split}' enthält keine Videos.")

    # Gefilterte Metadaten zurückgeben
    return out


def get_image_paths(split):
    """
    Kompatibilitätsfunktion für ältere Codestellen.

    Früher wurden separate Ordner data/train, data/val und data/test gelesen.
    Jetzt werden die entsprechenden Einträge aus data/splits.csv verwendet.

    Args:
        split (str): Gewünschter Split.

    Returns:
        pd.DataFrame: Metadaten des gewünschten Splits.
    """
    # Metadaten über die neue Split-Logik zurückgeben
    return get_metadata_for_split(split)


def load_images_from_path(file_path, num_frames):
    """
    Lädt gleichmäßig verteilte Frames aus einem Video.

    Args:
        file_path (str): Pfad zur Videodatei.
        num_frames (int): Anzahl der zu ladenden Frames.

    Returns:
        torch.Tensor: Tensor mit den extrahierten Videoframes.
    """

    # VideoDecoder für die angegebene Videodatei initialisieren
    decoder = VideoDecoder(file_path)

    # Gesamtanzahl der Frames aus den Videometadaten auslesen
    total_frames = decoder.metadata.num_frames

    # Ungültige oder leere Videos abfangen
    if total_frames is None or total_frames <= 0:
        raise ValueError(f"Video enthält keine gültige Frameanzahl: {file_path}")

    # Gleichmäßig verteilte Frame-Indizes über das gesamte Video erzeugen
    indices = torch.linspace(0, total_frames - 1, num_frames).long()
    # Frames an den berechneten Positionen laden
    frames = decoder.get_frames_at(indices=list(indices)).data
    # Geladene Frames zurückgeben
    return frames


def get_labels_from_dataset(dataset):
    """
    Extrahiert Labels aus verschiedenen Dataset-Typen, ohne Videos zu laden.

    Unterstützt:
        YawDDDataset
        Subset
        ConcatDataset

    Args:
        dataset: Dataset oder Dataset-Wrapper.

    Returns:
        list: Liste der Labels als numerische Werte.
    """

    # Labels direkt zurückgeben, falls es sich um das eigene Dataset handelt
    if isinstance(dataset, YawDDDataset):
        return list(dataset.labels)

    # Labels aus dem Elterndataset anhand der Subset-Indizes extrahieren
    if isinstance(dataset, Subset):
        parent_labels = get_labels_from_dataset(dataset.dataset)
        return [parent_labels[int(i)] for i in dataset.indices]

    # Labels aller Teildatensätze eines ConcatDataset sammeln
    if isinstance(dataset, ConcatDataset):
        labels = []
        # Jedes enthaltene Dataset einzeln auswerten
        for ds in dataset.datasets:
            labels.extend(get_labels_from_dataset(ds))
        # Zusammengeführte Label-Liste zurückgeben
        return labels

    # Fallback für unbekannte Dataset-Typen initialisieren
    labels = []

    # Alle Samples laden und Labels extrahieren
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(float(label))

    # Gesammelte Labels zurückgeben
    return labels


class YawDDDataset(Dataset):
    """
    PyTorch-Dataset für YawDD-artige Videodaten aus einem gemeinsamen Videoordner.

    Rückgabe:
        frames: Tensor mit Form (T, C, H, W)
        label: Tensor mit Wert 0.0 oder 1.0
    """

    def __init__(
        self,
        split,
        num_frames,
        train=True,
        video_dir=DEFAULT_VIDEO_DIR,
        split_csv=DEFAULT_SPLIT_CSV
    ):
        """
        Initialisiert das Dataset für einen bestimmten Split.

        Args:
            split (str): Gewünschter Split, z. B. train, val, test oder trainval.
            num_frames (int): Anzahl der Frames pro Video.
            train (bool): Aktiviert Trainingsaugmentationen, falls True.
            video_dir (str): Ordner mit allen Videos.
            split_csv (str): Pfad zur Split-Datei.
        """

        # Metadaten für den gewünschten Split laden
        df = get_metadata_for_split(
            split=split,
            video_dir=video_dir,
            split_csv=split_csv
        )

        # DataFrame für spätere Analysen oder Debugging speichern
        self.df = df

        # Liste aller Videopfade speichern
        self.image_paths = df["filepath"].tolist()
        # Labels als Float-Werte speichern
        self.labels = df["yawning"].astype(float).tolist()
        # IDs als Strings speichern, damit Gruppierungen konsistent bleiben
        self.ids = df["id"].astype(str).tolist()
        # Aktivitätsbezeichnungen speichern
        self.activities = df["activity"].tolist()

        # Splitnamen speichern
        self.split = split
        # Trainingsmodus speichern
        self.train = train
        # Anzahl der zu ladenden Frames speichern
        self.num_frames = num_frames

        # ImageNet-Mittelwerte für die Normalisierung definieren
        self.mean = [0.485, 0.456, 0.406]
        # ImageNet-Standardabweichungen für die Normalisierung definieren
        self.std = [0.229, 0.224, 0.225]


    def __len__(self):
        """
        Gibt die Anzahl der Samples im Dataset zurück.

        Returns:
            int: Anzahl der Labels und damit Anzahl der Videos.
        """
        return len(self.labels)


    def _frame_to_pil(self, frame):
        """
        Wandelt einen einzelnen Frame robust in ein PIL-Bild um.

        Unterstützte Tensorformen:
            (C, H, W)
            (H, W, C)

        Args:
            frame: Einzelner Videoframe.

        Returns:
            PIL.Image: Konvertierter Frame.
        """

        # Prüfen, ob der Frame als PyTorch-Tensor vorliegt
        if isinstance(frame, torch.Tensor):

            # Sicherstellen, dass der Frame drei Dimensionen besitzt
            if frame.ndim != 3:
                raise ValueError(f"Frame hat unerwartete Form: {frame.shape}")

            # Falls der Frame im HWC-Format vorliegt, in CHW-Format umwandeln
            if frame.shape[0] not in (1, 3, 4) and frame.shape[-1] in (1, 3, 4):
                frame = frame.permute(2, 0, 1)

            # Tensor in PIL-Bild umwandeln
            return TF.to_pil_image(frame)

        # Nicht-Tensoren direkt mit torchvision in PIL-Bilder umwandeln
        return TF.to_pil_image(frame)

    def _normalize_pil(self, img):
        """
        Konvertiert ein PIL-Bild in einen normalisierten Tensor.

        Args:
            img: PIL-Bild.

        Returns:
            torch.Tensor: Normalisierter Bildtensor.
        """
        # PIL-Bild in Tensor mit Wertebereich [0, 1] umwandeln
        tensor = TF.to_tensor(img)
        # Tensor mit ImageNet-Statistiken normalisieren
        tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
        # Normalisierten Tensor zurückgeben
        return tensor

    def _transform_clip_train(self, image_sequence):
        """
        Wendet Trainingsaugmentationen konsistent auf alle Frames eines Clips an.

        Die gleichen zufälligen Parameter werden für alle Frames eines Videos verwendet,
        damit keine künstlichen zeitlichen Sprünge innerhalb der Sequenz entstehen.

        Args:
            image_sequence: Sequenz geladener Videoframes.

        Returns:
            torch.Tensor: Transformierter Clip mit Form (T, C, H, W).
        """

        # Alle Frames in PIL-Bilder umwandeln
        pil_frames = [self._frame_to_pil(frame) for frame in image_sequence]

        # Alle Frames auf eine einheitliche Zwischengröße bringen
        pil_frames = [TF.resize(img, [256, 341]) for img in pil_frames]

        # Zufällige Entscheidung für horizontalen Flip pro Clip treffen
        do_flip = random.random() < 0.5

        # Horizontalen Flip auf alle Frames anwenden, falls ausgewählt
        if do_flip:
            pil_frames = [TF.hflip(img) for img in pil_frames]

        # Einen zufälligen Rotationswinkel pro Clip bestimmen
        angle = random.uniform(-10, 10)
        # Rotation mit gleichem Winkel auf alle Frames anwenden
        pil_frames = [TF.rotate(img, angle) for img in pil_frames]

        
        # Zufälligen Helligkeitsfaktor pro Clip bestimmen
        brightness_factor = random.uniform(0.8, 1.2)
        # Zufälligen Kontrastfaktor pro Clip bestimmen
        contrast_factor = random.uniform(0.8, 1.2)
        # Zufälligen Sättigungsfaktor pro Clip bestimmen
        saturation_factor = random.uniform(0.8, 1.2)
        # Zufälligen Farbtonfaktor pro Clip bestimmen
        hue_factor = random.uniform(-0.05, 0.05)

        # Farbanpassungen als Funktionen definieren
        color_ops = [
            lambda img: TF.adjust_brightness(img, brightness_factor),
            lambda img: TF.adjust_contrast(img, contrast_factor),
            lambda img: TF.adjust_saturation(img, saturation_factor),
            lambda img: TF.adjust_hue(img, hue_factor),
        ]

        # Reihenfolge der Farbanpassungen zufällig variieren
        random.shuffle(color_ops)

        # Alle Farbanpassungen nacheinander auf alle Frames anwenden
        for op in color_ops:
            pil_frames = [op(img) for img in pil_frames]

        # Alle Frames zentral auf 224x224 zuschneiden
        pil_frames = [TF.center_crop(img, [224, 224]) for img in pil_frames]

        # Alle Frames in normalisierte Tensoren umwandeln
        tensors = [self._normalize_pil(img) for img in pil_frames]

        # Einzelne Frame-Tensoren zu einem Clip-Tensor stapeln
        return torch.stack(tensors)

    def _transform_clip_val(self, image_sequence):
        """
        Wendet deterministische Vorverarbeitung für Validierung und Test an.

        Args:
            image_sequence: Sequenz geladener Videoframes.

        Returns:
            torch.Tensor: Transformierter Clip mit Form (T, C, H, W).
        """

        # Alle Frames in PIL-Bilder umwandeln
        pil_frames = [self._frame_to_pil(frame) for frame in image_sequence]
        # Alle Frames auf eine einheitliche Zwischengröße bringen
        pil_frames = [TF.resize(img, [256, 341]) for img in pil_frames]
        # Alle Frames zentral auf 224x224 zuschneiden
        pil_frames = [TF.center_crop(img, [224, 224]) for img in pil_frames]
        # Alle Frames in normalisierte Tensoren umwandeln
        tensors = [self._normalize_pil(img) for img in pil_frames]

        # Einzelne Frame-Tensoren zu einem Clip-Tensor stapeln
        return torch.stack(tensors)

    def __getitem__(self, idx):
        """
        Lädt ein Video, extrahiert Frames, transformiert diese und gibt Label zurück.

        Args:
            idx (int): Index des gewünschten Samples.

        Returns:
            tuple: (frames, label)
                frames: Tensor mit Form (T, C, H, W)
                label: Tensor mit Wert 0.0 oder 1.0
        """
        # Gleichmäßig verteilte Frames aus dem ausgewählten Video laden
        image_sequence = load_images_from_path(
            self.image_paths[idx],
            num_frames=self.num_frames
        )

        # Trainingsaugmentationen oder deterministische Vorverarbeitung anwenden
        if self.train:
            images = self._transform_clip_train(image_sequence)
        else:
            images = self._transform_clip_val(image_sequence)

        # Label des aktuellen Videos als Float-Tensor erzeugen
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Transformierten Videoclip und zugehöriges Label zurückgeben
        return images, label