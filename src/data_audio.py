import torch
from torch.utils.data import Dataset
from src.utils_audio import pad_or_truncate_audio, normalize_audio, load_audio_from_video


class AudioYawDDDataset(Dataset):
    """ audio only dataset
    Returns: {
        "audio": Tensor[audio_num_samples],
        "labels": Tensor[],
        "filepath": str
    }"""

    def __init__(self, df, cfg):
        self.filepaths = df["filepath"].tolist()
        self.labels = df["yawning"].tolist()
        self.sample_rate = cfg["audio_sample_rate"]
        self.num_samples = cfg["audio_num_samples"]
        self.mono = cfg["audio_mono"]
        self.do_normalize = cfg["audio_normalize"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        filepath = self.filepaths[index]

        waveform = load_audio_from_video(filepath=filepath, sample_rate=self.sample_rate, mono=self.mono,)
        waveform = pad_or_truncate_audio(waveform=waveform, num_samples=self.num_samples)

        if self.do_normalize:
            waveform = normalize_audio(waveform)

        label = torch.tensor(self.labels[index], dtype=torch.float32)

        return{
            "audio": waveform,
            "labels": label,
            "filepath": filepath,
        }
