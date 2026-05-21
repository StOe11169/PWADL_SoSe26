import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

from torchvision import transforms
from torchcodec.decoders import VideoDecoder

def get_image_paths(split):
    file_paths = []
    file_names = []
    folder_path = os.path.join("data", split)
    for dirpath, dirnames, filenames in os.walk(folder_path):
            for fname in filenames:
                file_paths.append(dirpath+'/'+fname)
                file_names.append(fname[:-4])

    df = pd.DataFrame([fn.split('-') for fn in file_names], columns=['id', 'info_labels', 'activity'])

    df['filepath'] = file_paths
    df['yawning'] = ['1' if 'yawning' in g.lower() else '0' for g in df['activity']]

    return df

def load_images_from_path(file_path, steps):
    # Get video frames
    decoder = VideoDecoder(file_path)

    # Select frame indices
    indices = torch.linspace(0, decoder.metadata.num_frames - 1, steps=steps).long()

    # Get raw image frames
    return decoder.get_frames_at(indices=list(indices)).data

def transform(file_path):
     return

class CustomDataset(Dataset):
    def __init__(self, split_type): # is called only once
        df_image_paths = get_image_paths(split_type)

        self.image_paths = df_image_paths['filepath'].tolist() # data_paths for efficient data handling with large datasets
        self.labels = df_image_paths['yawning'].tolist()


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): # is called multiple times during training and evaluation and should be written efficiently
        # load image from path
    #     # apply transforms (resize, CenterCrop, normalization, ToTensor, data augmentation operations, ...)
        return #images[idx], labels[idx]