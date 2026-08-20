import pandas as pd
import torch
from argparse import Namespace
from torch.utils.data import DataLoader

from src.config import build_config
from src.data_audio import AudioYawDDDataset
from src.models.audio.yamnet import YamNetLikeAudioClassifier
from src.utils import get_device


class FixedTrial:
    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_float(self, name, low, high, step=None, log=False):
        return low

    def suggest_int(self, name, low, high):
        return low


args = Namespace(
    data="dummy",
    epochs=1,
    num_frames=32,
)

cfg = build_config(FixedTrial(), args)

df = pd.DataFrame({
    "filepath": [
        "tests/fixtures/audio_dummy/001-driver-yawning.mp4",
        "tests/fixtures/audio_dummy/002-driver-normal.mp4",
    ],
    "yawning": [1.0, 0.0],
})

dataset = AudioYawDDDataset(df, cfg)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
)

device = get_device()

model = YamNetLikeAudioClassifier(dropout=cfg["dropout"], sample_rate=cfg["audio_sample_rate"]).to(device)

model.eval()

batch = next(iter(loader))
audio = batch["audio"].to(device)

with torch.no_grad():
    logits = model(audio)

print("Audio input shape:", audio.shape)
print("Logits shape:", logits.shape)
print("Logits:", logits)

assert logits.shape == (2,)

print("YAMNet-like forward dummy test passed.")