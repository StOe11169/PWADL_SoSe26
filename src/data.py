import os
import random
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from torchcodec.decoders import VideoDecoder


DEFAULT_VIDEO_DIR = os.path.join("data", "videos")
DEFAULT_SPLIT_CSV = os.path.join("data", "splits.csv")


def scan_video_folder(video_dir=DEFAULT_VIDEO_DIR):
    """
    Scannt data/videos rekursiv nach Videodateien.

    Erwartetes Dateinamenformat:
        ID-info_labels-activity.mp4

    Beispiel:
        001-Male-yawning.mp4
        001-Male-normal.mp4
        150-own-yawning.mp4
    """

    if not os.path.exists(video_dir):
        raise FileNotFoundError(
            f"Videoordner nicht gefunden: {video_dir}\n"
            f"Bitte lege deine Videos unter data/videos/ ab."
        )

    file_paths = []
    file_names = []

    for dirpath, dirnames, filenames in os.walk(video_dir):
        dirnames.sort()

        for fname in sorted(filenames):
            if fname.lower().endswith((".mp4", ".avi", ".mov")):
                full_path = os.path.normpath(os.path.join(dirpath, fname))
                name_without_ext = os.path.splitext(fname)[0]

                file_paths.append(full_path)
                file_names.append(name_without_ext)

    if len(file_paths) == 0:
        raise FileNotFoundError(f"Keine Videos in {video_dir} gefunden.")

    rows = []

    for fn, fp in zip(file_names, file_paths):
        parts = fn.split("-", maxsplit=2)

        if len(parts) != 3:
            raise ValueError(
                f"Dateiname hat nicht das erwartete Format "
                f"'ID-info_labels-activity': {fn}"
            )

        sample_id, info_labels, activity = parts

        yawning = 1.0 if "yawning" in activity.lower() else 0.0

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

    df = pd.DataFrame(rows)

    # Reproduzierbare Reihenfolge
    df = df.sort_values(["id", "filename"]).reset_index(drop=True)

    return df


def _safe_group_split(df, test_size, seed, split_name):
    """
    Erstellt einen gruppierten Split.

    Ziel:
        - gleiche ID kommt nicht in beide Teilmengen
        - Klassenverteilung möglichst ähnlich

    Vorgehen:
        Es werden viele GroupShuffleSplit-Kandidaten erzeugt.
        Danach wird der Kandidat gewählt, dessen yawning-Anteil
        in beiden Teilmengen am ähnlichsten ist.

    Vorteil:
        Funktioniert auch ohne StratifiedGroupShuffleSplit.
    """

    from sklearn.model_selection import GroupShuffleSplit

    y = df["yawning"].astype(int).to_numpy()
    groups = df["id"].astype(str).to_numpy()
    X = np.zeros(len(df))

    overall_pos_rate = y.mean()

    best_train_idx = None
    best_test_idx = None
    best_score = float("inf")

    # Mehrere zufällige Gruppensplits ausprobieren und den besten wählen
    n_candidates = 200

    for i in range(n_candidates):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed + i
        )

        train_idx, test_idx = next(splitter.split(X, y, groups))

        y_train = y[train_idx]
        y_test = y[test_idx]

        train_pos_rate = y_train.mean()
        test_pos_rate = y_test.mean()

        # Score: Abweichung beider Teilmengen vom Gesamtanteil
        score = abs(train_pos_rate - overall_pos_rate) + abs(test_pos_rate - overall_pos_rate)

        if score < best_score:
            best_score = score
            best_train_idx = train_idx
            best_test_idx = test_idx

    print(
        f"{split_name}: GroupShuffleSplit mit {n_candidates} Kandidaten verwendet. "
        f"Balance-Score={best_score:.4f}"
    )

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
    Erstellt data/splits.csv.

    Es entstehen die Splits:
        train
        val
        test

    Zusätzlich kann das Dataset später mit split="trainval"
    train + val zusammen verwenden.

    test_size:
        Anteil des Gesamtdatensatzes für Test.

    val_size:
        Anteil des Gesamtdatensatzes für Val.

    Beispiel bei test_size=0.2 und val_size=0.2:
        ca. 60 % train
        ca. 20 % val
        ca. 20 % test
    """

    os.makedirs(os.path.dirname(split_csv), exist_ok=True)

    current_df = scan_video_folder(video_dir)

    if os.path.exists(split_csv) and not force:
        existing_df = pd.read_csv(split_csv)

        current_paths = set(os.path.normpath(p) for p in current_df["filepath"].tolist())
        existing_paths = set(os.path.normpath(p) for p in existing_df["filepath"].tolist())

        if current_paths != existing_paths:
            missing_in_csv = current_paths - existing_paths
            missing_on_disk = existing_paths - current_paths

            raise ValueError(
                "\nDie vorhandene split_csv passt nicht mehr zum Videoordner.\n"
                f"Split-Datei: {split_csv}\n\n"
                f"Neue Videos ohne Split-Eintrag: {len(missing_in_csv)}\n"
                f"CSV-Einträge ohne Datei auf Disk: {len(missing_on_disk)}\n\n"
                "Wenn du neue Videos hinzugefügt hast, lösche data/splits.csv "
                "oder rufe create_split_csv(..., force=True) auf.\n"
            )

        print(f"Bestehende Split-Datei wird verwendet: {split_csv}")
        return existing_df

    df = current_df.copy()
    df["split"] = "unset"

    # 1. TrainVal/Test Split
    trainval_idx, test_idx = _safe_group_split(
        df,
        test_size=test_size,
        seed=seed,
        split_name="TrainVal/Test"
    )

    df.loc[test_idx, "split"] = "test"

    trainval_df = df.iloc[trainval_idx].copy().reset_index()
    # reset_index erzeugt Spalte "index", die auf Original-df zeigt.

    # 2. Innerer Train/Val Split
    # val_size ist bezogen auf Gesamtdaten.
    # Relative Val-Größe innerhalb von TrainVal:
    relative_val_size = val_size / (1.0 - test_size)

    inner_train_idx, inner_val_idx = _safe_group_split(
        trainval_df,
        test_size=relative_val_size,
        seed=seed + 1,
        split_name="Train/Val"
    )

    original_train_indices = trainval_df.iloc[inner_train_idx]["index"].to_numpy()
    original_val_indices = trainval_df.iloc[inner_val_idx]["index"].to_numpy()

    df.loc[original_train_indices, "split"] = "train"
    df.loc[original_val_indices, "split"] = "val"

    if (df["split"] == "unset").any():
        raise RuntimeError("Einige Samples wurden keinem Split zugewiesen.")

    df.to_csv(split_csv, index=False)

    print(f"\nSplit-Datei erstellt: {split_csv}")
    print_split_statistics(df)

    return df


def load_split_csv(
    video_dir=DEFAULT_VIDEO_DIR,
    split_csv=DEFAULT_SPLIT_CSV,
    test_size=0.20,
    val_size=0.20,
    seed=0
):
    """
    Lädt data/splits.csv oder erstellt sie automatisch.
    """

    if not os.path.exists(split_csv):
        return create_split_csv(
            video_dir=video_dir,
            split_csv=split_csv,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
            force=True
        )

    df = pd.read_csv(split_csv)
    df["filepath"] = df["filepath"].apply(os.path.normpath)

    return df


def print_split_statistics(df):
    """
    Gibt Klassenverteilung je Split aus.
    """

    print("\nSplit-Statistik:")

    for split in ["train", "val", "test"]:
        part = df[df["split"] == split]

        if len(part) == 0:
            print(f"{split}: leer")
            continue

        num_total = len(part)
        num_pos = int(part["yawning"].sum())
        num_neg = num_total - num_pos
        num_ids = part["id"].nunique()

        print(
            f"{split:5s}: total={num_total:4d}, "
            f"yawning={num_pos:4d}, "
            f"non-yawning={num_neg:4d}, "
            f"ids={num_ids:4d}"
        )

    train_ids = set(df[df["split"] == "train"]["id"].astype(str))
    val_ids = set(df[df["split"] == "val"]["id"].astype(str))
    test_ids = set(df[df["split"] == "test"]["id"].astype(str))

    overlap_train_val = train_ids.intersection(val_ids)
    overlap_train_test = train_ids.intersection(test_ids)
    overlap_val_test = val_ids.intersection(test_ids)

    print("\nID-Overlap-Prüfung:")
    print(f"train ∩ val : {len(overlap_train_val)}")
    print(f"train ∩ test: {len(overlap_train_test)}")
    print(f"val ∩ test  : {len(overlap_val_test)}")


def get_metadata_for_split(
    split,
    video_dir=DEFAULT_VIDEO_DIR,
    split_csv=DEFAULT_SPLIT_CSV
):
    """
    Gibt Metadaten für einen Split zurück.

    Unterstützte split-Werte:
        train
        val
        test
        trainval
        all
    """

    df = load_split_csv(video_dir=video_dir, split_csv=split_csv)

    if split == "all":
        out = df.copy()

    elif split == "trainval":
        out = df[df["split"].isin(["train", "val"])].copy()

    elif split in ["train", "val", "test"]:
        out = df[df["split"] == split].copy()

    else:
        raise ValueError(
            f"Unbekannter Split: {split}. "
            f"Erlaubt sind: train, val, test, trainval, all"
        )

    out = out.reset_index(drop=True)

    if len(out) == 0:
        raise ValueError(f"Split '{split}' enthält keine Videos.")

    return out


def get_image_paths(split):
    """
    Kompatibilitätsfunktion für deinen bisherigen Code.

    Früher wurden data/train, data/val, data/test gelesen.
    Jetzt werden die Splits aus data/splits.csv verwendet.
    """

    return get_metadata_for_split(split)


def load_images_from_path(file_path, num_frames):
    """
    Lädt num_frames gleichmäßig verteilte Frames aus einem Video.
    """

    decoder = VideoDecoder(file_path)

    total_frames = decoder.metadata.num_frames

    if total_frames is None or total_frames <= 0:
        raise ValueError(f"Video enthält keine gültige Frameanzahl: {file_path}")

    indices = torch.linspace(0, total_frames - 1, num_frames).long()

    frames = decoder.get_frames_at(indices=list(indices)).data

    return frames


def get_labels_from_dataset(dataset):
    """
    Extrahiert Labels aus YawDDDataset, Subset oder ConcatDataset,
    ohne Videos laden zu müssen.
    """

    if isinstance(dataset, YawDDDataset):
        return list(dataset.labels)

    if isinstance(dataset, Subset):
        parent_labels = get_labels_from_dataset(dataset.dataset)
        return [parent_labels[int(i)] for i in dataset.indices]

    if isinstance(dataset, ConcatDataset):
        labels = []
        for ds in dataset.datasets:
            labels.extend(get_labels_from_dataset(ds))
        return labels

    labels = []

    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(float(label))

    return labels


class YawDDDataset(Dataset):
    """
    Dataset für YawDD-artige Videodaten aus einem einzigen Ordner data/videos.

    Rückgabe:
        frames: Tensor mit Form (T, C, H, W)
        label: Tensor float, 0.0 oder 1.0
    """

    def __init__(
        self,
        split,
        num_frames,
        train=True,
        video_dir=DEFAULT_VIDEO_DIR,
        split_csv=DEFAULT_SPLIT_CSV
    ):
        df = get_metadata_for_split(
            split=split,
            video_dir=video_dir,
            split_csv=split_csv
        )

        self.df = df

        self.image_paths = df["filepath"].tolist()
        self.labels = df["yawning"].astype(float).tolist()
        self.ids = df["id"].astype(str).tolist()
        self.activities = df["activity"].tolist()

        self.split = split
        self.train = train
        self.num_frames = num_frames

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.labels)

    def _frame_to_pil(self, frame):
        """
        Robuste Umwandlung eines Frames nach PIL.

        Unterstützt:
            Tensor (C, H, W)
            Tensor (H, W, C)
        """

        if isinstance(frame, torch.Tensor):
            if frame.ndim != 3:
                raise ValueError(f"Frame hat unerwartete Form: {frame.shape}")

            # Falls Frame HWC ist, nach CHW wandeln.
            if frame.shape[0] not in (1, 3, 4) and frame.shape[-1] in (1, 3, 4):
                frame = frame.permute(2, 0, 1)

            return TF.to_pil_image(frame)

        return TF.to_pil_image(frame)

    def _normalize_pil(self, img):
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
        return tensor

    def _transform_clip_train(self, image_sequence):
        """
        Wendet dieselben zufälligen Augmentationen auf alle Frames eines Videos an.

        Wichtig:
            ColorJitter wird hier manuell umgesetzt, weil
            transforms.ColorJitter.get_params(...) je nach torchvision-Version
            ein Tuple statt eines callable Transform-Objekts zurückgeben kann.

        Dadurch vermeiden wir:
            TypeError: 'tuple' object is not callable
        """

        pil_frames = [self._frame_to_pil(frame) for frame in image_sequence]

        # Resize: torchvision erwartet [height, width]
        pil_frames = [TF.resize(img, [256, 341]) for img in pil_frames]

        # Zufälliger horizontaler Flip, aber konsistent für alle Frames
        do_flip = random.random() < 0.5

        if do_flip:
            pil_frames = [TF.hflip(img) for img in pil_frames]

        # Zufällige Rotation, aber gleicher Winkel für alle Frames
        angle = random.uniform(-10, 10)
        pil_frames = [TF.rotate(img, angle) for img in pil_frames]

        # ============================================================
        # Manueller ColorJitter, konsistent für den gesamten Clip
        # ============================================================

        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        saturation_factor = random.uniform(0.8, 1.2)
        hue_factor = random.uniform(-0.05, 0.05)

        color_ops = [
            lambda img: TF.adjust_brightness(img, brightness_factor),
            lambda img: TF.adjust_contrast(img, contrast_factor),
            lambda img: TF.adjust_saturation(img, saturation_factor),
            lambda img: TF.adjust_hue(img, hue_factor),
        ]

        # Reihenfolge wie bei ColorJitter zufällig machen
        random.shuffle(color_ops)

        for op in color_ops:
            pil_frames = [op(img) for img in pil_frames]

        # CenterCrop auf 224x224
        pil_frames = [TF.center_crop(img, [224, 224]) for img in pil_frames]

        tensors = [self._normalize_pil(img) for img in pil_frames]

        return torch.stack(tensors)

    def _transform_clip_val(self, image_sequence):
        """
        Deterministische Transformation für Validierung/Test.
        """

        pil_frames = [self._frame_to_pil(frame) for frame in image_sequence]

        pil_frames = [TF.resize(img, [256, 341]) for img in pil_frames]
        pil_frames = [TF.center_crop(img, [224, 224]) for img in pil_frames]

        tensors = [self._normalize_pil(img) for img in pil_frames]

        return torch.stack(tensors)

    def __getitem__(self, idx):
        image_sequence = load_images_from_path(
            self.image_paths[idx],
            num_frames=self.num_frames
        )

        if self.train:
            images = self._transform_clip_train(image_sequence)
        else:
            images = self._transform_clip_val(image_sequence)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return images, label