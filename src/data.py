import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import cv2


from torchvision import transforms
#from torchcodec.decoders import VideoDecoder

#conf = OmegaConf.load("config.yml")
#NUM_FRAMES = conf.training.num_frames

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
    df['yawning'] = [1.0 if 'yawning' in g.lower() else 0.0 for g in df['activity']]

    return df

def load_images_from_path(file_path, num_frames):
    # Open Video with OpenCV
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise IOError(f"Konnte Video nicht öffnen: {file_path}")
        
    # Read Frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate indices
    indices = torch.linspace(0, total_frames - 1, num_frames).long().tolist()
    
    frames = []
    
    for idx in indices:
        # Jump to Frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # Fallback
            if len(frames) > 0:
                frames.append(frames[-1]) # Duplicate last successful frame
            else:
                # Empty Frame, if first is failing
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 224
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 224
                import numpy as np
                frames.append(np.zeros((height, width, 3), dtype=np.uint8))

    cap.release()
    return frames


def transform(file_path):
     return

class CustomDataset(Dataset):
    def __init__(self, split_type, num_frames): # is called only once
        df_image_paths = get_image_paths(split_type)

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
        image_sequence = load_images_from_path(self.image_paths[idx], num_frames=self.num_frames) #Possible error at num_frames
        images = [self.transform(frame) for frame in image_sequence] 

        # get corresponding label
        label = self.labels[idx]

        return torch.stack(images), torch.tensor(label)