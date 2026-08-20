import os
import torch.nn.functional as F
from torchcodec.decoders import AudioDecoder

def load_audio_from_video(filepath, sample_rate=16000, mono=True):
    """
    Loads audio from a video file using TorchCodec.

    Uses two attempts:
    1. Let TorchCodec select the best audio stream.
    2. Fallback to stream_index=0 if best stream selection fails.
    """

    num_channels = 1 if mono else None

    try:
        decoder = AudioDecoder(filepath, sample_rate=sample_rate, num_channels=num_channels,)
    except Exception:
        # Fallback if TorchCodec cannot infer the best audio stream
        decoder = AudioDecoder(filepath, stream_index=0, sample_rate=sample_rate, num_channels=num_channels,)

    samples = decoder.get_all_samples()
    waveform = samples.data.float()

    if mono and waveform.dim() == 2:
        waveform = waveform.squeeze(0)

    return waveform

def is_excluded_audio_path(filepath, exlude_path_parts):
    #helper to ignore non audio videos, current hard code -> adjust later
    #Returns True for paths to be ignored, false otherwise
    pathname = os.path.normpath(filepath).lower()
    for part in exlude_path_parts:
        if part.lower() in pathname:
            return True

    return False

def filter_audio_dataframe(df, exclude_path_parts=None):
    #Remove videos without audio
    exclude_path_parts = exclude_path_parts or []

    keep_mask = df["filepath"].apply(lambda p: not is_excluded_audio_path(p, exclude_path_parts)) #what files to keep?
    filtered_df = df[keep_mask].reset_index(drop=True)
    print(f"Audio filtering: kept {len(filtered_df)} / {len(df)} samples")
    print(f"Audio filtering: removed {len(df) - len(filtered_df)} samples")
    return filtered_df


def has_decodable_audio(filepath, sample_rate=16000, mono=True):
    """
    Returns True if TorchCodec can decode audio from filepath.
    """
    try:
        waveform = load_audio_from_video(
            filepath=filepath,
            sample_rate=sample_rate,
            mono=mono,
        )

        if waveform.numel() == 0:
            return False

        return True

    except Exception as e:
        print(f"[NO AUDIO / DECODE FAILED] {filepath}")
        print(f"  Error: {e}")
        return False

def filter_decodable_audio_dataframe(df, cfg):
    """
    Removes:
    -Known no-audio folders, e.g. Mirror
    -Files where TorchCodec cannot decode audio

    only for first audio pipeline validation.
    """

    df = filter_audio_dataframe(df, exclude_path_parts=cfg["audio_exclude_path_parts"],)

    keep_rows = []

    for _, row in df.iterrows():
        filepath = row["filepath"]

        if has_decodable_audio(
            filepath=filepath,
            sample_rate=cfg["audio_sample_rate"],
            mono=cfg["audio_mono"],
        ):
            keep_rows.append(row)

    filtered_df = type(df)(keep_rows).reset_index(drop=True)

    print(f"Decode filtering: kept {len(filtered_df)} / {len(df)} samples")
    print(f"Decode filtering: removed {len(df) - len(filtered_df)} samples")

    return filtered_df

def pad_or_truncate_audio(waveform, num_samples):
    """
    Converts waveform to fixed length.
    Input: waveform Tensor[num_samples] or Tensor[channels, num_samples]
    Output: Tensor[num_samples]
    """
    if waveform.dim() ==2:
        #convert to mono
        waveform = waveform.mean(dim=0)

    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D waveform after mono conversion, got shape {waveform.shape}")

    current_samples = waveform.shape[0]

    if current_samples > num_samples:
        return waveform[:num_samples]

    if current_samples < num_samples:
        pad_amount = num_samples - current_samples
        return F.pad(waveform, (0, pad_amount))

    return waveform

def normalize_audio(waveform, eps=1e-8):
    #normalize waveform peaks to [-1,1]
    max_abs = waveform.abs().max().item()
    if max_abs < eps:
        return waveform
    return waveform / max_abs