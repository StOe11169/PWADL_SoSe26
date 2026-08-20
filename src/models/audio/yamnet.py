# https://www.codegenes.net/blog/yamnet-pytorch/
# because am lazy
# converting to have it use waveform input as I cannot be bothered to create the spectrogramms for all files
# also: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet

import torch
import torch.nn as nn
 
def hz_to_mel(freq):
    #convert frequency to mel-scale for creating log-mel-spectrogram
    #https://en.wikipedia.org/wiki/Mel_scale
    return 2595.0 * torch.log10(torch.tensor(1.0) + freq / 700.0)

def mel_to_hz(mel):
    #inverse of hz_to_mel()
    return 700.0*(10.0 ** (mel / 2595.0) - 1.0)

def create_mel_filterbank(sample_rate, n_fft, n_mels, f_min=0.0, f_max=None):
    """
    filterbank = array of bandpass filters to seperate input signal into multiple components
    #n_fft = number of fasst fourier transforms
    outputs Tensor[n_freq_bins, n_mels]
    """

    if f_max is None:
        f_max = sample_rate / 2

    n_freq_bins = n_fft // 2+1

    mel_min = hz_to_mel(torch.tensor(f_min))
    mel_max = hz_to_mel(torch.tensor(f_max))

    mel_points =torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    freq_bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
    freq_bins = torch.clamp(freq_bins, min=0, max=n_freq_bins -1)

    filterbank = torch.zeros(n_freq_bins, n_mels)

    for m in range(1, n_mels + 1):
        left = freq_bins[m - 1]
        center = freq_bins[m]
        right = freq_bins[m + 1]

        if center > left:
            filterbank[left:center, m - 1] = torch.linspace(0, 1, center - left)

        if right > center:
            filterbank[center:right, m - 1] = torch.linspace(1, 0, right - center)

    return filterbank

class LogMelSpectrogram(nn.Module):
    """waveform to log-mel spectrogram.
    Input:Tensor[B, num_samples]
    Output:Tensor[B, 1, n_mels, time]
    """

    def __init__(self,sample_rate=16000, n_fft=1024, win_length=400,hop_length=160,n_mels=64,):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        window = torch.hann_window(win_length)
        mel_filterbank = create_mel_filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
        )

        self.register_buffer("window", window)
        self.register_buffer("mel_filterbank", mel_filterbank)

    def forward(self, waveform):
        if waveform.dim() != 2:
            raise ValueError(f"Expected waveform shape [B, N], got {waveform.shape}")

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )

        power_spec = stft.abs().pow(2)              # [B, freq, time]
        mel_spec = torch.matmul(
            power_spec.transpose(1, 2),             # [B, time, freq]
            self.mel_filterbank,                   # [freq, mel]
        )                                          # [B, time, mel]

        log_mel = torch.log(mel_spec + 1e-6)
        log_mel = log_mel.transpose(1, 2)          # [B, mel, time]
        log_mel = log_mel.unsqueeze(1)             # [B, 1, mel, time]

        return log_mel


class DepthwiseSeparableConv(nn.Module):
   #Depthwise separable convolution block
   #compare: https://www.codegenes.net/blog/yamnet-pytorch/

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class YamNetLikeAudioClassifier(nn.Module):
    """
    YAMNet-like binary audio classifier
    Note: not actually googles pre-trained Yamnet, but has the same idea:
      waveform -> log-mel spectrogram -> depthwise separable CNN
    Output matches the visual model: logits shape [B].
    """

    def __init__(
        self,
        dropout=0.3,
        sample_rate=16000,
        n_mels=64,
    ):
        super().__init__()

        self.frontend = LogMelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
        )

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, waveform):
        x = self.frontend(waveform)
        x = self.features(x)
        x = self.pool(x)
        logits = self.cls_head(x).squeeze(-1)

        return logits