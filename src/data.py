import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchcodec.decoders import VideoDecoder


def get_image_paths(split):
    file_paths = []
    file_names = []
    folder_path = os.path.join("data", split)
    for dirpath, _, filenames in os.walk(folder_path):
            for fname in filenames:
                file_paths.append(dirpath+'/'+fname)
                file_names.append(fname[:-4])

    df = pd.DataFrame([fn.split('-') for fn in file_names], columns=['id', 'info_labels', 'activity'])

    df['filepath'] = file_paths
    df['yawning'] = [1.0 if 'yawning' in g.lower() else 0.0 for g in df['activity']]

    return df


def load_images_from_path(file_path, num_frames):
    # Get video frames
    decoder = VideoDecoder(file_path)

    # Select frame indices
    indices = torch.linspace(0, decoder.metadata.num_frames - 1, num_frames).long()

    # Get raw image frames
    return decoder.get_frames_at(indices=list(indices)).data


class YawDDDataset(Dataset):
    def __init__(self, split, num_frames): # is called only once
        df_image_paths = get_image_paths(split)

        self.image_paths = df_image_paths['filepath'].tolist() # data_paths for efficient data handling with large datasets
        self.labels = df_image_paths['yawning'].tolist()

        # transforms
        self.transform = transforms.Compose([
             transforms.ToPILImage(), 
             transforms.Resize((256, 341)),            # resize
             transforms.CenterCrop(224),        # crop
             transforms.ToTensor(),             # back to C×H×W tensor
             transforms.Normalize(
                  mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]
             )
        ])

        self.num_frames = num_frames        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): # is called multiple times during training and evaluation and should be written efficiently
        # load one image sequence from path
        image_sequence = load_images_from_path(self.image_paths[idx], num_frames=self.num_frames)
        images = [self.transform(frame) for frame in image_sequence] 

        # get corresponding label
        label = self.labels[idx]

        return torch.stack(images), torch.tensor(label)