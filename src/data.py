import os
import shutil
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from torchvision import transforms


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

def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Splits data based on subject IDs to prevent data leakage.
    Ensures the same person never appears in different splits.
    """
    # Create a unique subject group column (e.g., "1-Female" or "1-Male")
    df['subject_group'] = df['id'] + '-' + df['info_labels']
    
    # 1. First Split: Separate Test subjects from the rest
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['subject_group']))
    
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    # 2. Second Split: Separate Validation subjects from Remaining Train subjects
    adjusted_val_size = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['subject_group']))
    
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    
    return train_df, val_df, test_df

def prepare_and_split_data(raw_folder_path="data/raw", output_base_path="data", data_fraction=1.0, random_state=42):
    """
    Scans the raw data folder, sub-samples the dataset based on data_fraction (0.0 to 1.0) using a subject-grouped split,
    clears any old split directories, applies the train/val/test split, and copies the video files.
    """
    file_paths = []
    file_names = []
    
    for dirpath, dirnames, filenames in os.walk(raw_folder_path):
        for fname in filenames:
            file_paths.append(os.path.join(dirpath, fname))
            file_names.append(fname[:-4])

    if not file_names:
        print(f"No videos found in '{raw_folder_path}'.")
        return

    # Build initial full DataFrame
    df = pd.DataFrame([fn.split('-') for fn in file_names], columns=['id', 'info_labels', 'activity'])
    df['raw_filepath'] = file_paths
    df['yawning'] = [1.0 if 'yawning' in g.lower() else 0.0 for g in df['activity']]

    # --- 1. OPTIONAL DATA FRACTION SUBSAMPLING (GROUPED) ---
    if data_fraction < 1.0:
        if data_fraction <= 0.0:
            raise ValueError("data_fraction must be greater than 0.0")
        
        df['subject_group'] = df['id'] + '-' + df['info_labels']
        # Use GroupShuffleSplit to reduce the dataset size without splitting the same person's videos
        gss_fraction = GroupShuffleSplit(n_splits=1, test_size=1.0 - data_fraction, random_state=random_state)
        keep_idx, _ = next(gss_fraction.split(df, groups=df['subject_group']))
        df = df.iloc[keep_idx].reset_index(drop=True)
        print(f"Sub-sampled dataset to {data_fraction * 100:.1f}%. Utilizing {len(df)} videos.")

    # --- 2. CLEAR EXISTING DIRECTORIES ---
    split_names = ['train', 'val', 'test']
    for name in split_names:
        target_dir = os.path.join(output_base_path, name)
        if os.path.exists(target_dir):
            print(f"Clearing old directory: {target_dir}")
            shutil.rmtree(target_dir) # Completely deletes the folder and all its contents
        os.makedirs(target_dir, exist_ok=True)

    # --- 3. APPLY TRAIN/VAL/TEST SPLIT ---
    train_df, val_df, test_df = split_data(df, random_state=random_state)

    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    print("\n--- Split Statistics (Subject-Grouped) ---")
    for name, split_df in splits.items():
        unique_subjects = split_df['id'].str.cat(split_df['info_labels'], sep='-').nunique()
        print(f" {name.upper()}: {len(split_df)} videos from {unique_subjects} unique subjects.")

    # --- 4. COPY FILES INTO TARGET DIRECTORIES ---
    print("\nCopying files...")
    for split_name, split_df in splits.items():
        target_dir = os.path.join(output_base_path, split_name)
        for _, row in split_df.iterrows():
            source_file = row['raw_filepath']
            filename = os.path.basename(source_file)
            dest_file = os.path.join(target_dir, filename)
            
            # Changed from shutil.move to shutil.copy to preserve the raw data folder
            shutil.copy(source_file, dest_file)
            
    print("Data preparation and split completed successfully!")

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