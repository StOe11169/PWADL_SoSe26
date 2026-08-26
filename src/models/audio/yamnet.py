# Now using: https://github.com/w-hc/torch_audioset.git
# https://www.codegenes.net/blog/yamnet-pytorch/
# converting to have it use waveform input as I cannot be bothered to create the spectrogramms for all files
# also: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet

import torch
import torch.nn as nn
from torch_audioset.yamnet.model import yamnet
from torch_audioset.data.torch_input_processing import WaveformToInput

class YamNetAudioClassifier(nn.Module):
    """
    pretrained yamnet backbone with binary yawning classifier
    input: waveform [batch, num_clips, samples_per_clip]
    output: logits [batch]

    https://github.com/w-hc/torch_audioset.git
    """

    def __init__(self, dropout=0.3, sample_rate=16000,freeze_backbone = True):
        super().__init__()

        self.sample_rate = sample_rate
        self.freeze_backbone = freeze_backbone

        #wav -> log-mel
        self.frontend = WaveformToInput()

        #load pretrained weights
        #Note: First run needs internet connection to download weights
        self.backbone = yamnet(pretrained=True)
        
        #Note: yamnet maps 1024 features -> 521 classes
        embedding_size = self.backbone.classifier.in_features

        #replace yamnet classifier (521 classes) with just the 1024 vector embedding
        self.backbone.classifier = nn.Identity()

        #simple binary classification head
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_size, 1))

        #train only the new classification head
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, waveform):
        #fail early to not waste time
        if waveform.ndim != 3:
            raise ValueError("Expected waveform shape [batch, num_clips, samples_per_clip], got {tuple(waveform.shape)}")

        video_embeddings = []

        #process each videos audio separetly
        # waveform: [batch, num_clips, samples_per_clip]
        #merges patches into clips, then clips into video embeddings
        for video_clips in waveform:
            clip_embeddings = []

            #process one videos sampled clips at a time
            #video_clips: [num_clips, samples_per_clip]
           

            for clip in video_clips:
                # [samples] -> [1, samples]
                clip = clip.unsqueeze(0) #frontend uses cpu; move clips back to model device

                #convert clip to YAMNet-compatible patches
                patches, _ = self.frontend.wavform_to_log_mel(clip.cpu(), self.sample_rate)
                patches = patches.to(waveform.device)

                #avoid autograd for frozen backbone
                if self.freeze_backbone:
                    self.backbone.eval()

                    with torch.no_grad():
                        #[num_patches, 1024]
                        embeddings = self.backbone(patches)
                else:
                    embeddings = self.backbone(patches)

                #average YAMNet patches within this specific audio 
                #[num_patches, 1024] -> [1024]
                #Note: mean pooling removes order from patches and clips, same as with frames
                clip_embedding = embeddings.mean(dim=0)
                clip_embeddings.append(clip_embedding)

            #average sampled clips into one representation for the hole video
            #stack only after all videos are processed as stacking inside the loop would break the batches
            #[num_clips, 1024] -> [1024]
            video_embedding = torch.stack(clip_embeddings).mean(dim=0)
            video_embeddings.append(video_embedding)

        # [B, 1024]
        video_embeddings = torch.stack(video_embeddings)

        # [B, 1024] -> [B]
        return self.classifier(video_embeddings).squeeze(-1)


""" 
Misread note on torch audio and made these myself uncessarily, but keeping them for reference
def hz_to_mel(freq):
    #convert frequency to mel-scale for creating log-mel-spectrogram
    #https://en.wikipedia.org/wiki/Mel_scale
    return 2595.0 * torch.log10(torch.tensor(1.0) + freq / 700.0)

def mel_to_hz(mel):
    #inverse of hz_to_mel()
    return 700.0*(10.0 ** (mel / 2595.0) - 1.0)

def create_mel_filterbank(sample_rate, n_fft, n_mels, f_min=0.0, f_max=None):
    #filterbank = array of bandpass filters to seperate input signal into multiple components
    #n_fft = number of fasst fourier transforms
    #outputs Tensor[n_freq_bins, n_mels]
    
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





class LogMelSpectrogram(nn.Module):
    #waveform to log-mel spectrogram
    #waveform -> Short-Time Fourier Transform -> power spectrum -> mel filterbank
    #Input:Tensor[B, num_samples]
    #Output:Tensor[B, 1, n_mels, time]
    
    def __init__(self,sample_rate=16000, n_fft=1024, win_length=400,hop_length=160,n_mels=64,):
        super().__init__()
        self.mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate,n_fft=n_fft,win_length=win_length,hop_length=hop_length,n_mels=n_mels,
            power=2.0, center=True, norm=None, mel_scale="htk")
        
    def forward(self, waveform):
    #batches 1D audio into [batch, samples]
        if waveform.dim() != 2:
            raise ValueError(f"Expected waveform shape [B, N], got {waveform.shape}")
        # Output[B, n_mels, time]
        mel_spec = self.mel_spectrogram(waveform)

        log_mel = torch.log(mel_spec + 1e-6)

        # CNN expects image-like channel dimension
        # [B, 1, n_mels, time]
        return log_mel.unsqueeze(1)
"""

