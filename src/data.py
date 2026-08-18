import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from torchcodec.decoders import VideoDecoder

from sklearn.model_selection import GroupShuffleSplit


def get_dataframe():
    file_paths = []
    file_names = []

    folder_path = "data"

    for dirpath, _, filenames in os.walk(folder_path):
        for fname in filenames:
            file_paths.append(os.path.join(dirpath, fname))
            file_names.append(os.path.splitext(fname)[0])

    df = pd.DataFrame(
        [fn.split("-") for fn in file_names],
        columns=["id", "info_labels", "activity"]
    )

    df["filepath"] = file_paths
    df["yawning"] = [
        1.0 if "yawning" in x.lower() else 0.0
        for x in df["activity"]
    ]

    return df


def create_splits(random_state=42):
    """
    Creates:
        Train: 70%
        Val:   15%
        Test:  15%

    Groups are based on subject IDs.
    """

    df = get_dataframe()

    groups = df["id"]

    # First split: 70% train, 30% temp
    gss1 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.70,
        random_state=random_state
    )

    train_idx, temp_idx = next(
        gss1.split(df, groups=groups)
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Second split: split remaining 30% into 15% val + 15% test
    gss2 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.50,
        random_state=random_state
    )

    val_idx, test_idx = next(
        gss2.split(
            temp_df,
            groups=temp_df["id"]
        )
    )

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }


def load_images_from_path(file_path, num_frames):
    decoder = VideoDecoder(file_path)

    indices = torch.linspace(
        0,
        decoder.metadata.num_frames - 1,
        num_frames
    ).long()

    return decoder.get_frames_at(
        indices=list(indices)
    ).data


class YawDDDataset(Dataset):
    def __init__(
        self,
        split,               # "train", "val", or "test"
        num_frames,
        random_state=42
    ):
        splits = create_splits(random_state=random_state)
        df = splits[split]

        self.image_paths = df["filepath"].tolist()
        self.labels = df["yawning"].tolist()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 341)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.num_frames = num_frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_sequence = load_images_from_path(
            self.image_paths[idx],
            num_frames=self.num_frames
        )

        images = [
            self.transform(frame)
            for frame in image_sequence
        ]

        label = self.labels[idx]

        return torch.stack(images), torch.tensor(label)