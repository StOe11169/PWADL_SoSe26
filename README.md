
# Multimodal Yawning Detection from Video and Audio

Praktisches Wissenschaftliches Arbeiten mit Deep Learning (PWADL) - SoSe26

**Author:** [Stefan Oelbracht]  
**Repository:** [PWADL_SoSe26 - StOe_Multimodal branch](https://github.com/StOe11169/PWADL_SoSe26/tree/StOe_Multimodal)  
**Implementation documented:** commit [ed63ed6](https://github.com/StOe11169/PWADL_SoSe26/tree/ed63ed6fceca70cec29c3f422d6adbd7ced0fd8b)  
**Project status:** The pipelines are implemented and tested, but  models have not yet been trained. All numerical result fields below are therefore placeholders and will be added at a later date.

## Table of contents

- [1. Problem statement](#1-problem-statement)
- [2. System overview](#2-system-overview)
- [3. Datasets](#3-datasets)
- [4. Preprocessing and augmentation](#4-preprocessing-and-augmentation)
- [5. Model architecture](#5-model-architecture)
- [6. Experimental design](#6-experimental-design)
- [7. Installation and execution](#7-installation-and-execution)
- [8. Evaluation metrics](#8-evaluation-metrics)
- [9. Results template](#9-results-template)
- [10. Discussion](#10-discussion)
- [11. Limitations and threats to validity](#11-limitations-and-threats-to-validity)
- [12. Repository structure](#12-repository-structure)
- [13. References](#13-references)

## 1. Problem statement

This repository examines the effects of using multiple modes to detect yawning in a videosuequence and wether the added mode (audio) improves the ability to detect yawning.
Yawning being a visible and importantly only sometimes audible event that can be used to detect fatigue in drivers. Yawning itself is not a clear predictor of fatigue and drowsiness, but can aid in its detection. A practical fatigue detection system should additionaly use  eye closure, blink behaviour, head motion, or physiological measurements.
The narrower task of this project is **binary video-level yawning classification**. More specifically:

> Given a video containing a person, predict whether the clip contains yawning by combining visual facial motion and the accompanying sound.

The central question examined here is:

> **Does late fusion of independently trained visual and audio classifiers improve subject-independent yawning detection over either modality alone?**

Such this question can be divided further:

1. How accurately can yawning be detected from sampled RGB frames?
2. How accurately can it be detected from the audio track alone?
3. Does weighted late fusion improve performance on the same held-out participants?
4. Is an apparent fusion gain supported by the results, or does the fused model mainly reproduce one modality?

The positive class (1) is **yawning** and the negative class (0) combines **normal** and **talking** clips. A filename containing “Talking&Yawning” is therefore positive. This makes talking an important hard negative because both talking and yawning can involve an open mouth, a distinction already emphasized in the original YawDD work this project is based on ([Abtahi et al., 2014](https://doi.org/10.1145/2557642.2563678)).

## 2. System overview

The implementation is purposefully modular. Each can be run and trained in a standalone way. Each encoder produces one raw binary logit per video. The logits are aligned by filepath and combined only after the unimodal predictions have been produced.

~~~mermaid
flowchart TD
    V["Input video"]
    V --> VP["Visual preprocessing"]
    V --> AP["Audio preprocessing"]
    VP --> VM["ResNet-18 + attention pooling"]
    AP --> AM["YAMNet + mean pooling"]
    
    VM --> VL["Visual logit"]
    AM --> AL["Audio logit"]
    VL --> F["Weighted late fusion"]
    AL --> F
    F --> Y["Yawning probability and label"]
~~~

Three experiment modes are available:

| Mode | Input | Model output |
| --- | --- | --- |
| Visual | Uniformly sampled RGB frames | One visual logit per video |
| Audio | Four one-second clips sampled across a 16 kHz mono waveform | One audio logit per video |
| Multimodal | The same video processed by both pipelines | Weighted sum of visual and audio logits |

Late fusion was selected because it preserves two independently testable pipelines, makes unimodal ablations straightforward, and does not require the heterogeneous visual and audio feature sequences to have the same temporal resolution. This is a deliberately simple fusion baseline within the wider taxonomy of multimodal representation, alignment, and fusion methods described by [Baltrušaitis et al. (2019)](https://doi.org/10.1109/TPAMI.2018.2798607). This also makes it possible to determine which modality is the prominent indicator of the two. An information early-fusion would obstruct.

## 3. Datasets

### 3.1 YawDD visual dataset

YawDD was introduced specifically for yawning detection by [Abtahi et al. (2014)](https://doi.org/10.1145/2557642.2563678). The complete publication describes two in-vehicle views:

| Subset | Camera position | Published size | Structure |
| --- | --- | ---: | --- |
| Mirror / Case I | Below the front mirror | 322 videos | Three or four separate clips per participant, including normal, talking/singing, and yawning |
| Dash / Case II | On the dashboard | 29 videos | One longer clip per participant containing multiple mouth conditions |

The recordings were made in a parked car under varying illumination, with participants of different genders and facial characteristics, including glasses and sunglasses. The [official dataset page](https://www.site.uottawa.ca/~shervin/yawning/) links to the current open-access distribution. 

Additonally own videos where produced using the participants of this course. They where recorded in the same way as described by Abtahi et al..

### 3.2 Audio-capable project data

As the published YawDD files are distributed without audio. Therefore own videos where produced using the participants of this course. They where recorded in the same way as described by Abtahi et al.. just with audio. While both Dashboard and Mirrow views are available in the YawDD Dataset and the the self produced one. During training the mirror view from the YawDD Dataset and the dashboard view from the selfmade dataset where used. Unfortunatly I can not recall any specific reason for this decision. Each participant was recorded in each of the following conditions: 
| Eyewear condition    | Yawning recordings | Non-yawning recordings | Applicability                         |
| -------------------- | -----------------: | ---------------------: | ------------------------------------- |
| Without glasses      |                  2 |                      2 | All participants                      |
| With regular glasses |                  2 |                      2 | Participants who usually wore glasses |
| With sunglasses      |                  2 |                      2 | All participants                      |

The recordings were made consecutivly in a parked car during a partially cloudy day.Specifically Thursday,11 June 2026 in Gelsenkirchen, Germany.
The distribution of of participants of the self made, un-augmented dataset used here, as well as the technical details of the un-augmented dataset are:

|  |  |
| --- | --- |
| Number of participants | 5 |
| Number of videos | 48 |
| Positive / negative videos | **24 / 24** |
| Recording device and microphone | **TBD** |
| Video resolution and frame rate | **3840x2160px ; 29,97frames/s** |
| Original sample rate | **48.000kHz** |
| Clip duration distribution | **TBD** |


After recording, the custom videos were initially converted with VLC Media Player to approximate the resolution, frame rate, and bitrate of the YawDD videos. Because TorchCodec could not reliably seek within or decode the resulting MP4 files, the final copies were subsequently normalized using FFmpeg. This normalization regenerated presentation timestamps, shifted negative timestamps to zero, re-encoded the video as H.264 with YUV 4:2:0 pixel format, retained the audio as AAC, and rebuilt the MP4 container index. This normalization step was separate from data augmentation and did not change the labels.
To increase the limited dataset all videos where augmented once (doubling the dataset size) using the augment_dataset.py script found in this repository and treated as new participants.
This knowingly introduces data leakage, as the random transformations can not be so extreme that the participants cant't be recoginzed as the same person, while still producing a usefull video. This was deemed a worty compromise in order to gain sufficient data for the nested-cross validation.

### 3.3 Labels and filename convention

The data loader recursively scans the local directory named **data** for MP4, AVI, and MOV files. It expects every filename stem to contain exactly three hyphen-separated fields:

~~~text
<participant-id>-<participant-information>-<activity>.<extension>
~~~

Examples:

~~~text
001-FemaleNoGlasses-Normal.avi
001-FemaleNoGlasses-Talking.avi
001-FemaleNoGlasses-Yawning.avi
023-MaleGlasses-Talking&Yawning.avi
~~~

The label is derived from the activity field:

$$y_i =
\begin{cases} 
1, & \text{if “yawning” occurs in the activity string},\\
0, & \text{otherwise}.
\end{cases}$$


The label applies to the complete video even though the actual yawn occupies only a short interval. This is a known source of label noise and should be kept in mind when interpreting attention weights or failure cases. A different approach (insert yawdd paper with csv here)

### 3.4 Data layout

The repository does not include the training data. The original YAWD-Dataset can be found here: https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset
The self produced videos are not available to the public.
 A compatible layout of the data folder would be:

~~~text
data/
├── Mirror/
│   ├── 001-FemaleNoGlasses-Normal.avi
│   ├── 001-FemaleNoGlasses-Talking.avi
│   └── 001-FemaleNoGlasses-Yawning.avi
└── CustomAudioVisual/
    ├── 101-Participant-Normal.mp4
    └── 101-Participant-Yawning.mp4
~~~

Important: the command-line option named **--data** is currently written into the experiment configuration but does not change the scanned root; **main.py always calls get_all_data_paths("data")**. Stage only the intended experiment files under data, or change the loader before running from a different root.

### 3.5 Subject-independent splitting

Multiple clips from one participant are correlated. Random video-level splitting could place the same face in training and testing, allowing identity and background cues to inflate the measured performance. The implementation therefore uses the parsed participant ID as a group and [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) to keep groups disjoint while approximately preserving the class distribution.

The current nested cross-validation structure is:

- **Outer evaluation:** 5 subject-grouped folds.
- **Inner model selection:** 3 subject-grouped folds within each outer training partition.
- **Final epoch selection:** one grouped training/validation split taken from a separate 5-fold `StratifiedGroupKFold` applied only to the outer training partition.
- **Outer test data:** used once to evaluate the selected final model for that outer fold.

Although the final splitter is configured with two folds, the implementation calls `next(...)` and therefore uses only its first split; it does not train two additional final models.

Nested cross-validation is important here because using the same cross-validation results both to select hyperparameters and to report performance creates optimistic bias. This risk and the role of nested evaluation are demonstrated by [Varma and Simon (2006)](https://doi.org/10.1186/1471-2105-7-91) and [Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html).

One repository specific thing to note is that grouping only uses the subjects ID. In the Mirror naming scheme, a male and a female participant can share the same number. They are consequently placed in the same fold even though they are different people. This is avoids data leakage, but it reduces the effective number of groups and can make stratification less flexible. A final version should preferably use unambiguous participant key such as gender plus numeric ID.

## 4. Preprocessing and augmentation

### 4.1 Visual preprocessing

For every video, \(T\) frame indices are selected by linear spacing from the first to the last frame. The current command-line default is \(T=64\). TorchCodec decodes the frames, after which each frame is:

1. converted to a PIL image;
2. resized to 256 × 341 pixels;
3. centre-cropped to 224 × 224 pixels;
4. converted to a channel-first floating-point tensor; and
5. normalized with ImageNet mean $(0.485, 0.456, 0.406)$ and standard deviation $(0.229, 0.224, 0.225)$.

These dimensions and normalization match the standard preprocessing associated with pretrained Torchvision ResNet-18 weights. Uniform sampling results in a fixed memory use and coveres the full clip, but it can miss yawns or only capture them partialy. This type of sampling was chosen with the eventual use-case in mind. To have the model run on edge devices in cars with limited ressources.

### 4.2 Audio preprocessing

The audio loader uses TorchCodec to decode the complete audio stream and resample it to 16 kHz mono. It then samples four one-second clips distributed linearly across the complete recording. Number and length of the aduio clips was chosen in order not to sample to much of a given video, lasting aproximatly ~30s.

For a waveform containing $N_i$ samples, the default configuration uses
$$
C = 4,
\quad
L = 16\,000
$$
clips and samples per clip. When $N_i > L$, clip start positions are selected by linearly spacing four indices between 0 and $N_i-L$. Each resulting segment therefore has exactly one second of audio. This works similarly to how the visual pipeline samples frames. The length can be adjusted in the configuration.

For testing purposes using dummy files the file is zero-padded to \(L\) samples and repeated four times, if the recodring is no longer than one second.Each sampled clip is peak-normalized independently when its maximum absolute amplitude is non-zero. The resulting tensor has the shape

$$
A_i \in \mathbb{R}^{4 \times 16\,000}.
$$

This sampling allows to cover the whole video, while keeping memory and compute costs constant, again similar to the visual pipeline. It can nevertheless miss a short yawn that falls between the sampled intervals. Event-centred sampling or sliding-window inference would provide stronger temporal coverage.

Sixteen-kilohertz mono audio and log-mel input remain consistent with the official [YAMNet preprocessing specification](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet).

The configuration contains an audio missing-policy field, but the training workflow still filters primarily by path substring. It does not invoke the available audio-decodability check before cross-validation. Every retained audio or multimodal file must therefore contain a valid audio stream.

### 4.3 Offline augmentation utility

The optional **src/augment_dataset.py** utility applies:

- horizontal flipping;
- rotation in approximately $[-5^\circ,5^\circ]$;
- small brightness and contrast changes;
- Gaussian noise; and
- optional Gaussian blur.

The augmentation scanner now handles MP4 and AVI files independently of the main dataset loader. It removes a trailing `-converted` suffix before parsing filenames and preserves the input subdirectory hierarchy in the augmented output directory. MOV files are not processed by this utility, they are also not present in the current dataset.
Flip, rotation, brightness, contrast, noise strength, and blur selection are sampled once per video so that the principal transformation does not flicker between frames. The noise realization itself varies by frame, but its strength is consistent across the video. Applying coherent spatial transformations is preferable to independently warping each frame because a video model should not learn artificial temporal discontinuities. See [Shorten and Khoshgoftaar (2019)](https://doi.org/10.1186/s40537-019-0197-0) for image augmentation and [Cauli and Reforgiato Recupero (2022)](https://doi.org/10.3390/fi14030093) for video-specific augmentation considerations.

**Leakage warning:** the offline utility assigns augmented copies new IDs and is not integrated into the fold-specific training loader. Do not combine original and augmented copies before cross-validation if they can be assigned to different folds. The scientifically safe alternatives are:

1. perform augmentation on-the-fly only for the training partition of each fold; or
2. preserve the original participant group for every augmented derivative and guarantee that all derivatives remain in the same fold.
No augmentation is currently applied during the training loop.

## 5. Model architecture

### 5.1 Visual branch

The visual classifier uses an ImageNet-pretrained ResNet-18 without its final classification layer. ResNet-18 is a comparatively compact residual network and offers a practical transfer-learning baseline for a small video dataset ([He et al., 2016](https://doi.org/10.1109/CVPR.2016.90)).

Each frame is processed independently into a 512-dimensional feature. A learned attention module assigns a scalar weight to every frame, and the weighted sum is passed through the binary classification head:

| Module | Input | Output | Function |
| --- | --- | --- | --- |
| ResNet-18 feature extractor | $B \times T \times 3 \times 224 \times 224$ | $B \times T \times 512$ | Extracts one spatial appearance vector per frame |
| Attention scorer | $B \times T \times 512$ | $B \times T \times 1$ | Scores the relevance of each sampled frame |
| Weighted pooling | Features and normalized scores | $B \times 512$ | Produces one clip representation |
| Classification head | $B \times 512$ | $B$ | Layers 512 → 256 → 128 → 1 with batch normalization, ReLU, and dropout |

Let $X_i = (x_{i1},\ldots,x_{iT})$ be the sampled frames for clip $i$. The frame features are

$$
h_{it} = R_{\theta}(x_{it}) \in \mathbb{R}^{512}.
$$

The attention score and normalized weight are

$$
e_{it} = w_2^\top \tanh(W_1 h_{it} + b_1) + b_2,
$$

$$
\alpha_{it} = \frac{\exp(e_{it})}{\sum_{k=1}^{T}\exp(e_{ik})}.
$$

The clip representation is

$$
\bar{h}_i = \sum_{t=1}^{T}\alpha_{it} h_{it} \in \mathbb{R}^{512},
$$

and the classification head produces the visual logit

$$
\ell_{v,i} = C_{\phi}(\bar{h}_i) \in \mathbb{R}.
$$

Attention pooling is appropriate for a weakly labelled clip because only some frames may display a yawn. It's related to attention-based multiple-instance aggregation ([Ilse et al., 2018](https://proceedings.mlr.press/v80/ilse18a.html)). However, this implementation has no positional encoding, recurrence, or temporal convolution. The weighted sum is permutation-invariant, so it selects informative frames but **does not model frame order or motion direction**. Simply put: it learns the most important frames, but not how they relate to time. It could for example not differentiate between a closing and opening mouth. CNN-LSTM yawning detectors such as [Zhang and Su (2017)](https://doi.org/10.1109/SSCI.2017.8285343) provide a relevant recurrent baseline. Inflated 3D convolution is another stronger video baseline because it learns joint spatial-temporal filters, although at substantially higher computational cost ([Carreira and Zisserman, 2017](https://doi.org/10.1109/CVPR.2017.502)).
The main reasons for using the Resnet archictecture where the pretrained weights and its comparatively low computational costs.

### 5.2 Audio branch

The audio classifier uses a PyTorch port of YAMNet. The official YAMNet is a MobileNet-v1-based network pretrained to recognize 521 AudioSet event classes. AudioSet contains large-scale human-labelled sound events and provides a useful starting point for transfer learning ([Gemmeke et al., 2017](https://doi.org/10.1109/ICASSP.2017.7952261)), similar to the pretrained Resnet.

Each video is represented by \(C=4\) sampled audio clips:

$$
A_i = (a_{i1},\ldots,a_{iC})
\in \mathbb{R}^{C \times L},
\qquad
C=4,
\quad
L=16\,000.
$$

For each clip $a_{ic}$, the YAMNet frontend produces overlapping log-mel patches. After removal of the original 521-class output layer, the frozen YAMNet backbone produces a 1024-dimensional representation for every patch:

$$
g_{ick} = A_{\psi}(p_{ick})
\in \mathbb{R}^{1024}.
$$

The patch embeddings are first averaged within each sampled clip:

$$
\tilde{g}_{ic} = \frac{1}{K_{ic}} \sum_{k=1}^{K_{ic}} g_{ick}
$$

The four clip representations are then averaged to obtain one representation for the complete video:

$$
\bar{g}_i=\frac{1}{C} \sum_{c=1}^{C}\tilde{g}_{ic}
$$

Finally, a dropout-plus-linear head produces the audio logit:

$$
\ell_{a,i}=w_a^\top\operatorname{Dropout}(\bar{g}_i)+b_a
$$

The YAMNet backbone currently remains frozen and only the binary classification head is trained. Sampling several short intervals increases temporal coverage without processing the complete waveform, but averaging discards clip order and can dilute a localized sound event.
Similar to how the Resnet does not preserve this order, which is one of the reasons YamNet was chosen for the aduio pipeline in addition to the pretrained weights and comparetivly low computational costs.
> 

### 5.3 Late fusion

For a fixed visual weight $\lambda \in [0,1]$, the fused logit is

$$
\ell_{f,i} = \lambda\ell_{v,i} + (1-\lambda)\ell_{a,i}.
$$

The final probability and label are

$$
\hat{p}_i = \sigma(\ell_{f,i}) = \frac{1}{1+\exp(-\ell_{f,i})},
$$

$$
\hat{y}_i = \mathbb{1}[\hat{p}_i > 0.5]
           = \mathbb{1}[\ell_{f,i} > 0].
$$

The default is $\lambda = 0.5$. Because neural-network logits can have different scales and may be poorly calibrated ([Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html)), equal numerical weights do not necessarily mean equal information. The fusion weight must be fixed in advance or selected only inside the inner cross-validation loop. Selecting it from outer-test performance would leak test information.

The implementation also records

$$
c_{v,i} = \lambda\ell_{v,i},
\qquad
c_{a,i} = (1-\lambda)\ell_{a,i},
$$

and their absolute shares. These values describe the magnitude of the two terms in the fusion equation. They are **not causal feature importance** and cannot replace a proper ablation experiment.

### 5.4 Training objective

Both trainable branches are binary classifiers; late fusion itself has no learned parameters. For a model logit $\ell_i$ and ground-truth label $y_i \in \{0,1\}$, the weighted binary cross-entropy is

$$
\mathcal{L}_i =-\left[w_+ y_i \log\sigma(\ell_i)+ (1-y_i)\log(1-\sigma(\ell_i))
\right]
$$

The positive-class weight is calculated separately from the training data of each inner fold:

$$
w_+ = \frac{N_-}{N_+},
$$

where $N_-$ and $N_+$ are the numbers of negative and positive videos in that fold’s training partition. Validation and test labels are not used in this calculation. The weight is recalculated for the final training split and the class counts and resulting ratio are recorded with the experiment configuration.

PyTorch’s fused [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) is used for numerical stability. 
Training and validation losses are reported as sample-weighted means rather than sums of batch means, making the logged loss values comparable across the configured batch sizes.

## 6. Experimental design

### 6.1 Nested model selection

For every outer fold:

1. keep one participant-disjoint fold untouched as the outer test set;
2. create the same eight inner participant-disjoint folds for all Optuna trials;
3. maximize mean validation F1 across the eight inner folds;
4. choose the best completed trial;
5. train a fresh final model on the outer training data, using a grouped validation split for early model selection; and
6. evaluate the selected checkpoint once on the outer test fold.

In multimodal mode this process is run independently for the visual and audio branches, using the same participant splits. Their selected models are then evaluated on the same outer-test videos and fused by filepath.

Optuna’s median pruner stops unpromising trials after a warm-up, reducing compute while retaining a define-by-run search space ([Akiba et al., 2019](https://doi.org/10.1145/3292500.3330701)).

### 6.2 Hyperparameter search space

| Parameter | Search or fixed value |
| --- | --- |
| Frames per video | Command line; default 64 |
| Epochs per training run | Command line; default 50 |
| Optuna trials per study | Command line; default 10 |
| Batch size | {4, 8} |
| Dropout | 0.2 to 0.6 in steps of 0.1 |
| Optimizer | AdamW or SGD |
| Learning rate | Log-uniform from $10^{-5}$ to $10^{-3}$ |
| Weight decay | Log-uniform from $10^{-6}$ to $10^{-2}$ |
| SGD momentum | 0.0 to 0.95 |
| Scheduler | None, exponential, or step |
| Exponential gamma | 0.85 to 0.99 |
| Step size | 2 to 10 epochs |
| Step gamma | 0.1 to 0.9 |
| Positive-class weight | Training-fold ratio $N_-/N_+$, recalculated for every inner and final training split |
| Gradient clipping | Maximum norm 1.0 |
| Data-loader workers | 0 |
| Visual fusion weight | Command line; default 0.5 |
| Audio clips per video | Fixed at 4 |
| Duration per audio clip | Fixed at 1.0 s, or 16,000 samples |

AdamW is included because it decouples weight decay from the adaptive gradient update ([Loshchilov and Hutter, 2019](https://openreview.net/forum?id=Bkg6RiCqY7)). SGD is retained as a conventional comparison. Dropout regularizes the two classifier heads ([Srivastava et al., 2014](https://www.jmlr.org/papers/v15/srivastava14a.html)). Small batches are partly required by video memory; prior work has also documented a generalization gap in some large-batch regimes ([Keskar et al., 2017](https://openreview.net/forum?id=H1oyRlYgg)).

The visual ResNet backbone is fine-tuned end-to-end with one optimizer. The audio YAMNet backbone is frozen. Despite earlier project notes considering warm-up phases and separate optimizers, those strategies are **not present in the documented commit** and are therefore not claimed here.

### 6.3 Reproducibility controls

- Python, NumPy, and PyTorch seeds are set to 0.
- Cross-validation split seeds are set to 42.
- CuDNN deterministic mode is enabled and benchmarking is disabled when CUDA is available.
- The best validation-F1 checkpoint is saved for each run.
- Hyperparameters and per-trial inner-fold F1 values are written to JSON.
- Training curves and validation confusion matrices are written to TensorBoard.

Exact reproducibility can still be affected by library versions, GPU kernels, media decoding, filesystem traversal order, and hardware. The main dataset scanner does not currently sort the paths returned by `os.walk`, so the initial sample ordering can differ between systems.

The shuffled training loader may drop its final batch when that batch would contain only one video, because the visual classification head contains `BatchNorm1d`. Training metrics are therefore calculated with a separate non-shuffled loader over the complete training dataset using `drop_last=False`. Validation metrics likewise use the complete, non-shuffled validation dataset.

## 7. Installation and execution

### 7.1 Prerequisites

- Python 3.12 is recommended for the current branch and pinned 2026 dependency set.
- FFmpeg must be available to TorchCodec and to the offline augmentation utility.
- A CUDA-capable GPU is strongly recommended for nested visual training.
- Internet access is required on the first run to download pretrained ResNet-18 and YAMNet weights.
- Git must be available during installation because `torch-audioset` is installed directly from a pinned Git commit.

Check FFmpeg:

~~~bash
ffmpeg -version
~~~

### 7.2 Clone and create the environment

~~~bash
git clone --branch StOe_Multimodal --single-branch \
  https://github.com/StOe11169/PWADL_SoSe26.git
cd PWADL_SoSe26

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

The code also directly imports NumPy, pandas, scikit-learn, Matplotlib, Seaborn, and TensorBoard, but these packages are not currently pinned explicitly in `requirements.txt`. Install them separately:

~~~bash
python -m pip install numpy pandas scikit-learn matplotlib seaborn tensorboard
~~~

Record and pin their final versions before running the experiments used in the report.

`torch-audioset` is already installed by `requirements.txt` from the pinned Git commit `daed697ddcd1d2f7304cc3c2b1b4295ac7e304e4`. A separate editable clone is therefore not required and could unintentionally replace the documented version.

On systems with an NVIDIA GPU, ensure that the installed PyTorch version includes CUDA support. Select the appropriate Windows or Linux CUDA command from the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) and verify the installation with `torch.cuda.is_available()`.

Record and pin the final versions before running the experiments used in the report.
On Windows PowerShell, activate the environment with:

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

TorchCodec depends on a compatible PyTorch and FFmpeg combination. Consult the [TorchCodec compatibility documentation](https://github.com/meta-pytorch/torchcodec) if media decoders fail to load.

### 7.3 Pipeline tests

The three committed dummy videos contain synthetic visual motion and sine-wave audio. They test decoding, shapes, filepath alignment, logit fusion, and output creation; they do **not** test yawning accuracy.

~~~bash
python test_pipelines.py --mode visual
python test_pipelines.py --mode audio
python test_pipelines.py --mode multimodal --visual-weight 0.5

# Run all three
python test_pipelines.py --mode all --visual-weight 0.5
~~~


### 7.4 Run experiments
The current command-line defaults are 64 frames per video, 50 epochs per training run, and 10 Optuna trials. The experiment code uses five outer folds and eight inner folds. Smaller values may be used for development runs, but results from reduced runs should be clearly identified and should not be reported as the final experiment.
For testing purposes it is also advisable to reduce the number of folds. The fold counts are currently hard-coded in `run_experiment()` and `run_multimodal_experiment()` in `src/experiment.py`. All `StratifiedGroupKFold` instances require `n_splits >= 2`. The final training/validation stage uses only the first split returned by its five-fold splitter.
Visual-only nested cross-validation:

~~~bash
python main.py \
  --mode visual \
  --num_frames 32 \
  --epochs 10 \
  --n_trials 8
~~~

Audio-only nested cross-validation:

~~~bash
python main.py \
  --mode audio \
  --epochs 10 \
  --n_trials 8 \
  --audio_exclude_path_parts Mirror
~~~

Multimodal nested cross-validation with equal logit weights:

~~~bash
python main.py \
  --mode multimodal \
  --num_frames 32 \
  --epochs 10 \
  --n_trials 8 \
  --visual-weight 0.5 \
  --audio_exclude_path_parts Mirror
~~~

For the modality comparison idealy use identical eligible videos, subject groups, outer folds, epoch budgets, and search budgets. A visual model trained on YawDD plus custom clips is not directly comparable to an audio model trained only on custom clips. The clean comparison is visual-only, audio-only, and fused evaluation on the same audio-capable subset.

### 7.5 Monitoring

`main.py` creates a mode-specific, timestamped study directory and starts TensorBoard on port 6006 for that directory. It verifies that the TensorBoard process started successfully, attempts to open a browser, and terminates the process when the experiment ends or raises an exception.

Study directories follow this pattern:

~~~text
logs/study_<mode>_<YYYYMMDD_HHMMSS>/
~~~

Examples include:

~~~text
logs/study_visual_20260826_143015/
logs/study_audio_20260826_150221/
logs/study_multimodal_20260826_161407/
~~~

In multimodal mode, each outer fold contains separate `visual`, `audio`, and `fusion` subdirectories.

After an experiment has ended, all studies can be inspected manually with:

~~~bash
tensorboard --logdir logs --port 6006
~~~

### 7.6 Output files

| Output | Meaning |
| --- | --- |
| best_model_trial_*.pth | Checkpoint with the best validation F1 |
| checkpoint_trial_*.pth | Latest epoch, model, and optimizer state |
| trial_*_summary.json | Trial configuration and inner-fold F1 scores |
| tensorboard_trial_* | Loss, F1, precision, recall, learning rate, and validation confusion matrices |
| fusion_predictions.csv | Per-video visual, audio, and fused logits and predictions |
| fusion_summary.json | Fused metrics and weighted-logit magnitude summary |
| `outer_cv_summary.json` | Completed outer-fold F1 scores, mean, standard deviation, and completion status |
| `console.log` | Complete console output, warnings, runtime, and error traceback |
| `outer_test_predictions.csv` | Standalone per-video labels, logits, probabilities, and predictions |
| `outer_cv_summary.json` | Per-fold and aggregate accuracy, precision, recall, and F1 |

After every successfully completed outer fold, the experiment overwrites `outer_cv_summary.json` in the study root. This preserves partial progress if a later fold fails. The file contains:

- `mode`
- `expected_folds`
- `completed_folds`
- `complete`
- `fold_f1`
- `mean_f1`
- `std_f1`

For unimodal experiments, the outer summary persists F1 only. Accuracy, precision, and recall are calculated for each outer-test fold but are not currently printed or saved.

For multimodal experiments, each completed outer fold additionally stores `fusion_predictions.csv` and `fusion_summary.json` under `outer_fold_<fold>/fusion/`. These files contain per-video predictions and the fold’s fused accuracy, precision, recall, F1, and contribution diagnostics. The study-level `outer_cv_summary.json` aggregates only fused F1.
**Note: `fold_f1` stores scores in completion order but does not store the corresponding outer-fold identifiers. Final results should therefore only be taken from a summary where `complete` is `true`; otherwise the missing fold cannot be identified from this file alone.**

## 8. Evaluation metrics

Let TP, TN, FP, and FN denote true positives, true negatives, false positives, and false negatives for the yawning class.

$$
\operatorname{Accuracy} =
\frac{TP+TN}{TP+TN+FP+FN}
$$

$$
\operatorname{Precision} =
\frac{TP}{TP+FP}
$$

$$
\operatorname{Recall} =
\frac{TP}{TP+FN}
$$

$$
F_1 =
\frac{2\cdot\operatorname{Precision}\cdot\operatorname{Recall}}
{\operatorname{Precision}+\operatorname{Recall}}
$$

The primary selection metric is positive-class F1 because it balances false alarms and missed yawns and is more informative than accuracy alone when the classes are unequal. The interpretation of these measures and their dependence on the confusion matrix is discussed by [Sokolova and Lapalme (2009)](https://doi.org/10.1016/j.ipm.2009.03.002).

The workflow is configured for five outer folds. The current implementation reports the arithmetic mean and NumPy’s population standard deviation (`numpy.std` with its default `ddof=0`):

$$
\bar{m} = \frac{1}{K}\sum_{k=1}^{K}m_k,
\qquad
\sigma_{\mathrm{CV}}=\sqrt{\frac{1}{K}\sum_{k=1}^{K}(m_k-\bar{m})^2}
$$

For a complete experiment, $K=5$. The `complete` field in `outer_cv_summary.json` must be `true` before the mean and standard deviation are treated as final results. If a fold is skipped because no Optuna trial completes, the file contains a partial summary calculated from the remaining completed folds.

## 9. Results template

> **Current status:** No trained-model results are available. 

### 9.1 Execution environment

| Item | Final value |
| --- | --- |
| Git commit | **TBD** |
| Experiment date | **TBD** |
| Operating system | **TBD** |
| CPU and RAM | **TBD** |
| GPU and VRAM | **TBD** |
| Python | **TBD** |
| PyTorch / Torchvision / TorchCodec | **TBD** |
| CUDA / cuDNN | **TBD** |
| FFmpeg | **TBD** |

### 9.2 Final dataset after filtering

| Experiment subset | Participants / groups | Videos | Positive | Negative | Excluded |
| --- | ---: | ---: | ---: | ---: | ---: |
| YawDD visual | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| Custom audio-capable | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| Common multimodal evaluation subset | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |


### 9.3 Selected hyperparameters

Report best cfg but note that a separate final deployment study would be needed to confirm it.

| Outer fold | Mode | Batch | Dropout | Optimizer | LR | Weight decay | Scheduler | Best epoch |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- | ---: |
| 1 | Visual | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| 1 | Audio | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 5 | Audio | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

### 9.4 Aggregate outer-test performance

| Model | Outer-test F1, mean ± population SD | Source |
| --- | ---: | --- |
| Visual: ResNet-18 + attention | **TBD** | `outer_cv_summary.json` |
| Audio: frozen YAMNet + linear head | **TBD** | `outer_cv_summary.json` |
| Multimodal: weighted late fusion | **TBD** | `outer_cv_summary.json` |

Accuracy, precision, and recall are saved for each multimodal fold in `fusion_summary.json`, but are not persisted for unimodal outer-test folds. TensorBoard validation metrics must not be reported as outer-test results.

### 9.5 Paired fold-level modality ablation

Note: Use the exact same outer-test videos for all three rows in a fold.
The multimodal workflow does not directly save separate visual-only and audio-only F1 values for the outer-test fold. They must either be calculated from the modality logits in `fusion_predictions.csv` or obtained from separate unimodal experiments using identical folds and videos.

| Outer fold | Visual F1 | Audio F1 | Fused F1 | Fused − visual | Fused − audio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| 2 | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| 3 | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| 4 | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| 5 | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| Mean ± SD | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

Note: A high mean visual absolute-logit share is descriptive, but only a consistent paired F1 improvement demonstrates predictive value from fusion.

### 9.6 Fusion diagnostics

| Quantity | Final value |
| --- | ---: |
| Fixed or inner-selected visual weight $\lambda$ | **TBD** |
| Audio weight $1-\lambda$ | **TBD** |
| Mean absolute visual logit share | **TBD** |
| Mean absolute audio logit share | **TBD** |
| Number of test clips where fusion corrects visual | **TBD** |
| Number of test clips where fusion harms visual | **TBD** |
| Number of test clips where modalities disagree | **TBD** |

ToDo: inspect disagreement cases qualitatively with categories like visible but acoustically silent yawns, loud speech mistaken for a yawn, covered mouths, off-axis faces, background speech, low audio level, and yawns that fall between the four sampled one-second audio intervals.
Correction, harm, and disagreement counts are not currently included in `fusion_summary.json`. They must be derived from `fusion_predictions.csv` by thresholding the visual, audio, and fused logits at zero.

### 9.7 Training curves and result visualizations

ToDo: Export

1. training and validation loss/F1 curves for the selected model;
2. one outer-test confusion matrix per model or an explicitly defined pooled out-of-fold matrix;
3. a fold-level plot comparing visual, audio, and fused F1; 
4. optionally, visual attention weights over sampled frames.

Template:
~~~markdown
<!--
![Training and validation curves](docs/figures/training_curves.png)
![Out-of-fold confusion matrices](docs/figures/confusion_matrices.png)
![Outer-fold modality comparison](docs/figures/modality_f1_by_fold.png)
![Visual attention weights](docs/figures/attention_weights.png)
-->
~~~

ToDo: Caption every plot with the dataset subset, split level, metric aggregation, and model configuration. Note: validation confusion matrix is not a test result

### 9.8 Runtime and resource requirements

| Mode | Total nested-CV time | Mean time per training run | Peak GPU memory | Peak RAM | Checkpoint size |
| --- | ---: | ---: | ---: | ---: | ---: |
| Visual | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| Audio | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| Multimodal complete study | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

ToDo: report per-video inference time and whether decoding is included. Training time from the older template must not be reused because the current nested protocol trains many more models.

These measurements matter if the work is later moved from an offline study to an in-vehicle or mobile prototype: prior deployment-oriented drowsiness work shows that model size and latency can become first-class design constraints ([Jabbar et al., 2018](https://doi.org/10.1016/j.procs.2018.04.060)).

### 9.9 Comparison with prior work

It should be notet that reported values from the literature can only serve as context, not a leaderboard. The label granularity, camera view, preprocessing, participant split, and metric definition differ across papers, they are therfore not directly comparible among each other and with this project.

| Work | Core approach | Reported result | Relationship to this project |
| --- | --- | --- | --- |
| [Omidyeganeh et al., 2016](https://doi.org/10.1109/TIM.2015.2507378) | Viola–Jones face/mouth detection and temporal mouth-change measurement on embedded hardware | Correct yawning detection rate reported as 65% for mirror and 75% for dash | Lightweight hand-engineered baseline; different metric and protocol |
| [Zhang and Su, 2017](https://doi.org/10.1109/SSCI.2017.8285343) | CNN features followed by LSTM temporal modelling | 88.6% accuracy reported by the paper | Motivates an order-aware temporal baseline |
| [Saurav et al., 2020](https://doi.org/10.1007/978-3-030-44689-5_17) | Mouth-region features from a pretrained CNN, followed by 1D CNN and bidirectional LSTM | Evaluated on manually annotated clips from YawDD and NTHU-DDD | Closely related order-aware yawn classifier; protocol and clip extraction differ |
| [Kielty et al., 2023](https://doi.org/10.1117/12.2680327) | Event-camera representation with CNN, self-attention, and recurrence | 95.9% precision and 94.7% recall on their unseen-subject test; 89.9% and 91.0% on simulated public data | Strong subject-disjoint temporal comparison, but a different sensor representation and dataset |
| [Lu et al., 2023](https://doi.org/10.1109/TITS.2023.3285923) | Key-frame selection, face frontalization, head-pose and facial-action fusion | State-of-the-art YawDD results reported in the paper | Addresses pose and face localization omitted by the present model |
| This work | ResNet-18 attention + YAMNet + weighted late fusion | **TBD** | Adds an audio ablation and nested subject-grouped evaluation |

The fairest internal baselines are more important than cross-paper rank:

- mean pooling instead of visual attention;
- visual attention versus an order-aware GRU, LSTM, or temporal convolution;
- visual-only versus audio-only versus fused predictions on identical folds;
- fixed equal fusion versus an inner-selected and optionally calibrated fusion weight.

## 10. Discussion

To be completed after training. The statements below define the questions to answer.

### 10.1 Did multimodal fusion help?

-mean paired change from visual to fused F1 and the number of outer folds improved. Note: small average gain accompanied by large fold variance should be described as inconclusive. If fusion helps only when the visual branch is uncertain, show representative cases. If the fused model is worse, inspect logit scale, audio quality, event timing, and background speech before concluding that sound is uninformative.

### 10.2 Which modality dominated?

Comparing

1. unimodal and fused outer-fold metrics
2. per-video correction
3. weighted-logit magnitudes as a non-causal diagnostic

It should be noted that we can not infer modality importance solely from the configured weight or the mean absolute share. A large-magnitude but poorly calibrated branch can numerically dominate without adding accuracy.

### 10.3 What worked well?

Underpin following assesments with evidence from chapter 9

- subject-grouped nested validation keeps model selection separate from performance estimation;
- shared multimodal folds make paired comparison possible;
- pretrained encoders reduce the amount of task-specific data required;
- modular late fusion enables transparent unimodal ablations;
- checkpoints, JSON summaries, per-video fusion output, and TensorBoard improve traceability.

### 10.4 What was difficult?
Expand after training:
- YawDD supplies visual data but not usable audio, requiring a smaller custom multimodal dataset;
- clip-level filename labels are temporally imprecise;
- nested optimization is computationally expensive;
- video decoding and 64 ResNet passes per sample create high memory and I/O cost;
- yawning sound may be weak, silent, confused with speech, or absent from all four sampled one-second intervals;
- equal fusion of uncalibrated logits may not balance the branches.

## 11. Limitations and threats to validity

1. **No final empirical results yet.** The architecture and protocol are documented, but effectiveness remains unmeasured until the outer-fold experiments are complete.
2. **Dataset mismatch across modalities.** YawDD supports the visual branch, whereas multimodal evaluation depends on separate custom recordings. Claims about the value of sound must use the common audio-capable subset.
3. **Weak clip labels and sparse temporal sampling.** A positive filename does not identify yawn onset and offset. Uniform visual sampling and four sparse one-second audio intervals can miss the labelled event.
4. **Attention is not temporal dynamics.** Visual pooling can emphasize frames but is invariant to their order. It cannot distinguish opening from closing motion by sequence direction.
5. **No explicit face or mouth detector.** The centre crop includes the full scene and may devote capacity to background, camera, identity, or illumination cues.
6. **Frozen generic audio representation.** YAMNet was trained for broad AudioSet events, not specifically for yawning. Training only the head may underfit.
7. **Raw-logit fusion.** Different calibration and scale between encoders can distort a weighted sum. Fusion weights and calibration must be chosen without outer-test access.
8. **Augmentation is not fold-safe by default.** The offline augmentation script is not called by training and will leak augmented if originals and copies are combined with different group IDs.
9. **Audio filtering is path-based.** The configured missing-audio policy is not enforced by decoding checks in the main workflow.
10. **Participant-ID collisions.** Grouping male and female participants who share a numeric ID is conservative but reduces the independent group count.
11. **External validity.** YawDD was recorded in parked vehicles, and custom data may be collected under limited conditions. Performance cannot be assumed to generalize to moving vehicles, different cameras, languages, microphones, noise, demographics, or spontaneous rather than acted yawns.
12. **Safety.** The project is a research classifier, not a validated driver-safety device. It must not be used as the sole basis for safety-critical decisions.

## 12. Repository structure

~~~text
.
├── main.py                         # CLI and experiment entry point
├── requirements.txt               # Pinned Python dependencies
├── Jupyter_Notebooks/
│   └── annotationen_yawdd.ipynb   # Exploratory YawDD annotation analysis
├── src/
│   ├── augment_dataset.py         # Hierarchy-preserving offline video augmentation
│   ├── config.py                  # Optuna search space and fixed settings
│   ├── data.py                    # File parsing, visual dataset, and split helpers
│   ├── data_audio.py              # Four evenly distributed audio clips per video
│   ├── evaluation.py              # Metrics and raw-logit prediction
│   ├── experiment.py              # Nested CV and experiment dispatch
│   ├── fusion.py                  # Weighted late fusion and output files
│   ├── training.py                # Training loop and checkpointing
│   ├── utils.py                   # Seeds, device, optimizer, scheduler, TensorBoard
│   ├── utils_audio.py             # Decoding, clip sampling, padding, and normalization
│   └── models/
│       ├── visual/model.py        # ResNet-18 attention classifier
│       └── audio/yamnet.py        # Frozen YAMNet with clip/video mean pooling
├── test_pipelines.py              # ├── test_pipelines.py              # Batch-size-two preprocessing, inference, and fusion tests
└── tests/                         # Three synthetic audio-video fixtures
~~~

## 13. References

1. Abtahi, S., Omidyeganeh, M., Shirmohammadi, S., & Hariri, B. (2014). [YawDD: A yawning detection dataset](https://doi.org/10.1145/2557642.2563678). _Proceedings of the 5th ACM Multimedia Systems Conference_, 24–28.
2. Omidyeganeh, M., Shirmohammadi, S., Abtahi, S., et al. (2016). [Yawning detection using embedded smart cameras](https://doi.org/10.1109/TIM.2015.2507378). _IEEE Transactions on Instrumentation and Measurement, 65_(3), 570–582.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep residual learning for image recognition](https://doi.org/10.1109/CVPR.2016.90). _Proceedings of CVPR_, 770–778.
4. Ilse, M., Tomczak, J. M., & Welling, M. (2018). [Attention-based deep multiple instance learning](https://proceedings.mlr.press/v80/ilse18a.html). _Proceedings of ICML_, 2127–2136.
5. Gemmeke, J. F., Ellis, D. P. W., Freedman, D., et al. (2017). [Audio Set: An ontology and human-labeled dataset for audio events](https://doi.org/10.1109/ICASSP.2017.7952261). _Proceedings of ICASSP_, 776–780.
6. Google Research. [YAMNet: A pretrained audio event classifier](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet). Official model documentation and preprocessing specification.
7. w-hc. [torch_audioset](https://github.com/w-hc/torch_audioset). PyTorch port used by the audio branch.
8. Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). [Multimodal machine learning: A survey and taxonomy](https://doi.org/10.1109/TPAMI.2018.2798607). _IEEE Transactions on Pattern Analysis and Machine Intelligence, 41_(2), 423–443.
9. Varma, S., & Simon, R. (2006). [Bias in error estimation when using cross-validation for model selection](https://doi.org/10.1186/1471-2105-7-91). _BMC Bioinformatics, 7_, 91.
10. Cawley, G. C., & Talbot, N. L. C. (2010). [On over-fitting in model selection and subsequent selection bias in performance evaluation](https://www.jmlr.org/papers/v11/cawley10a.html). _Journal of Machine Learning Research, 11_, 2079–2107.
11. Shorten, C., & Khoshgoftaar, T. M. (2019). [A survey on image data augmentation for deep learning](https://doi.org/10.1186/s40537-019-0197-0). _Journal of Big Data, 6_, 60.
12. Cauli, N., & Reforgiato Recupero, D. (2022). [Survey on videos data augmentation for deep learning models](https://doi.org/10.3390/fi14030093). _Future Internet, 14_(3), 93.
13. Sokolova, M., & Lapalme, G. (2009). [A systematic analysis of performance measures for classification tasks](https://doi.org/10.1016/j.ipm.2009.03.002). _Information Processing & Management, 45_(4), 427–437.
14. Saito, T., & Rehmsmeier, M. (2015). [The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets](https://doi.org/10.1371/journal.pone.0118432). _PLOS ONE, 10_(3), e0118432.
15. Loshchilov, I., & Hutter, F. (2019). [Decoupled weight decay regularization](https://openreview.net/forum?id=Bkg6RiCqY7). _International Conference on Learning Representations_.
16. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). [Dropout: A simple way to prevent neural networks from overfitting](https://www.jmlr.org/papers/v15/srivastava14a.html). _Journal of Machine Learning Research, 15_, 1929–1958.
17. Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2017). [On large-batch training for deep learning: Generalization gap and sharp minima](https://openreview.net/forum?id=H1oyRlYgg). _International Conference on Learning Representations_.
18. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). [Optuna: A next-generation hyperparameter optimization framework](https://doi.org/10.1145/3292500.3330701). _Proceedings of KDD_, 2623–2631.
19. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). [On calibration of modern neural networks](https://proceedings.mlr.press/v70/guo17a.html). _Proceedings of ICML_, 1321–1330.
20. Zhang, W., & Su, J. (2017). [Driver yawning detection based on long short term memory networks](https://doi.org/10.1109/SSCI.2017.8285343). _IEEE Symposium Series on Computational Intelligence_, 1–5.
21. Kielty, P., Dilmaghani, M. S., Ryan, C., Lemley, J., & Corcoran, P. (2023). [Neuromorphic sensing for yawn detection in driver drowsiness](https://doi.org/10.1117/12.2680327). _Proceedings of SPIE 12701, Fifteenth International Conference on Machine Vision_.
22. Lu, Y., Liu, C., Chang, F., Liu, H., & Huan, H. (2023). [JHPFA-Net: Joint head pose and facial action network for driver yawning detection across arbitrary poses in videos](https://doi.org/10.1109/TITS.2023.3285923). _IEEE Transactions on Intelligent Transportation Systems, 24_(11), 11850–11863.
23. Meta PyTorch. [TorchCodec documentation](https://github.com/meta-pytorch/torchcodec). Media-decoding implementation reference.
24. PyTorch. [Torchvision ResNet-18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). Pretrained weights and canonical preprocessing.
25. Carreira, J., & Zisserman, A. (2017). [Quo vadis, action recognition? A new model and the Kinetics dataset](https://doi.org/10.1109/CVPR.2017.502). _Proceedings of CVPR_, 4724–4733.
26. Saurav, S., Mathur, S., Sang, I., et al. (2020). [Yawn detection for driver's drowsiness prediction using bi-directional LSTM with CNN features](https://doi.org/10.1007/978-3-030-44689-5_17). _Intelligent Human Computer Interaction_, 189–200.
27. Jabbar, R., Al-Khalifa, K., Kharbeche, M., et al. (2018). [Real-time driver drowsiness detection for Android application using deep neural networks techniques](https://doi.org/10.1016/j.procs.2018.04.060). _Procedia Computer Science, 130_, 400–407.

---

_This project was developed for the university course “Praktisches Wissenschaftliches Arbeiten mit Deep Learning.”_
