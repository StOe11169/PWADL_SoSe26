import torch
from torch.utils.data import Dataset
from src.utils_audio import normalize_audio, load_audio_from_video, sample_audio_clips


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
        #audio settings
        self.sample_rate = cfg["audio_sample_rate"]
        self.mono = cfg["audio_mono"]
        self.do_normalize = cfg["audio_normalize"]
        #sample multiple short clips from a video
        self.num_audio_clips = cfg["num_audio_clips"]
        self.clip_seconds = cfg["audio_clip_seconds"]
        #nr of waveform samples per clip
        self.samples_per_clip = int(self.sample_rate * self.clip_seconds)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        #decode video into waveform before sampling
        waveform = load_audio_from_video(filepath=filepath, sample_rate=self.sample_rate, mono=self.mono,)
        #sample clips from video, fixed amount keeps samples stackable by data loader
        clips = sample_audio_clips(waveform=waveform, num_clips=self.num_audio_clips, samples_per_clip=self.samples_per_clip)
        
        #normalize each clip, silence remains zero
        if self.do_normalize:
            clips = torch.stack([normalize_audio(clip) for clip in clips])

        label = torch.tensor(self.labels[index], dtype=torch.float32)
        #Output shape before batching: [num_clips, samples_per_clip]
        return{#[num:clips, sample_per_clip]
            "audio": clips,
            "labels": label,
            "filepath": filepath}
