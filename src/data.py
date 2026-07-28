import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchcodec.decoders import VideoDecoder
from sklearn.model_selection import train_test_split, GroupShuffleSplit



#Data Handling Functions
"""
def get_image_paths(split): #image paths are pre order according to their "split" i.e training, test or validation data
    file_paths = []
    file_names = []
    folder_path = os.path.join("data", split) #create pahth for sub-folder in data accoridng to split
    for dirpath, _, filenames in os.walk(folder_path): #Go through every directory and file and ad it to file_names and file_path
            for fname in filenames:
                file_paths.append(dirpath+'/'+fname)
                file_names.append(fname[:-4])

    #Create a dataframe from filenames
    #Changes 001-driver-yawning.mp4 into
    #id     info_labels     activity
    #001    driver          yawning
    df = pd.DataFrame(
         [fn.split('-') for fn in file_names], 
         columns=['id', 'info_labels', 'activity']
         )

    df['filepath'] = file_paths
    df['yawning'] = [1.0 if 'yawning' in g.lower() else 0.0 for g in df['activity']] #convert activity into binary classification label

    return df
"""
def get_all_data_paths(root="data"):
    """
    Scan dataset directory and returns a DataFrame with
    filepath,id, info_labels, activity, yawning binary label
    
    Assumes filenames follow pattern:
        <id>-<info>-<activity>.mp4
    e.g:
        001-driver-yawning.mp4
    """

    file_paths = []
    file_names = []

    # Walk through all subdirectories and collect videos
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith((".mp4", ".avi", ".mov")):  # only consider video files
                file_paths.append(dirpath+'/'+fname)
                #full_path = os.path.join(dirpath, fname)
                #file_paths.append(full_path)
                file_names.append(os.path.splitext(fname)[0]) #remove extension names

    # Split filenames
    # Example: "001-driver-yawning" → ['001', 'driver', 'yawning']
    df = pd.DataFrame([fn.split('-') for fn in file_names], columns=['id', 'info_labels', 'activity'])

    # Add full file paths
    df['filepath'] = file_paths

    # Convert activity into binary label
    df['activity'] = df['activity'].astype(str) #convert to string
    # Create binary label: 1.0 if 'yawning' appears in activity 
    # pd.notna + str() to safely handle missing or non-string values
    df['yawning'] = df['activity'].apply(lambda x: 1.0 if pd.notna(x) and 'yawning' in str(x).lower() else 0.0)


    # Optional: ensure consistent ordering (for reproducibility)
    #df = df.sort_values(by='filepath').reset_index(drop=True)

    return df

def create_splits(df, test_size=0.2, val_size=0.1, seed=42):
    """
    Splits dataset into train, validation, and test sets.

    Parameters:
    - df: DataFrame returned by get_all_data_paths()
    - test_size: fraction of total data used for test set
    - val_size: fraction of total data used for validation set
    - seed: random seed for reproducibility

    Returns:
    - train_df, val_df, test_df (all disjoint)

    Notes:
    - Splitting is done at VIDEO LEVEL
    - test set size is determined by 1-test_size - val_size
    """

    #Split into train+val and test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['yawning'],   # maintain class distribution
        random_state=seed
    )

    #Split train+val into train and validation
    # Adjust validation size relative to remaining data
    val_relative_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df['yawning'],
        random_state=seed
    )

    # Data leakage sanity check
    assert set(train_df.filepath).isdisjoint(val_df.filepath)
    assert set(train_df.filepath).isdisjoint(test_df.filepath)
    assert set(val_df.filepath).isdisjoint(test_df.filepath)

    return train_df, val_df, test_df

def create_group_splits(df, output_dir, file_col = 'filepath', test_size = 0.15, val_size = 0.15, seed=42):
    """
    Splits dataset into train, validation, and test sets.

    Parameters:
    - df: DataFrame returned by get_all_data_paths()
    - test_size: fraction of total data used for test set
    - val_size: fraction of total data used for validation set
    - seed: random seed for reproducibility

    Returns:
    - train_df, val_df, test_df (all disjoint)

    Notes:
    - Splitting is done via ID
    - test set size is determined by 1-train_size - val_size
    """

    #Split into train+val and into test set
    gss = GroupShuffleSplit(n_splits= 1, test_size= test_size, random_state= seed)
    train_val_idx, test_idx = next(gss.split(df, groups=df['id']))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df      = df.iloc[test_idx].reset_index(drop=True)

    #Split train+val again into train and val
    val_relative_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=seed)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups= train_val_df['id']))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    #Sanity check for leakage
    assert set(train_df['id']).isdisjoint(val_df['id'])
    assert set(train_df['id']).isdisjoint(test_df['id'])
    assert set(val_df['id']).isdisjoint(test_df['id'])

    print("Train IDs:", len(train_df['id'].unique()))
    print("Val IDs:", len(val_df['id'].unique()))
    print("Test IDs:", len(test_df['id'].unique()))

    print("Train samples:", len(train_df))
    print("Val samples:", len(val_df))
    print("Test samples:", len(test_df))

    #Create metadata for all sets and save to csv
    metadata = []
    for set_name, df_set in [('train', train_df), ('val', val_df), ('test', test_df)]:
        #Group by ID and collect into a list
        grouped = df_set.groupby('id')[file_col].apply(list).reset_index()
        grouped['set'] = set_name
        metadata.append(grouped)

    #Combine sets into df
    metadata_df = pd.concat(metadata, ignore_index=True)

    output = os.path.join(os.getcwd(), output_dir, f"sample_distribution")
    metadata_df.to_csv(output)

    return train_df, val_df, test_df

def load_images_from_path(file_path, num_frames): #get fixed number of frames from a video
    # Convert video into singular frames
    decoder = VideoDecoder(file_path) 

    # Select frames from linearly spaced indices across the whole video/frame sequence, returns 1D tensor, long just for memory
    indices = torch.linspace(0, decoder.metadata.num_frames - 1, num_frames).long()

    #Return only selected raw images frames
    return decoder.get_frames_at(indices=list(indices)).data



#--------------------------------------------------------------------------------------------#

#Dataset Class

class YawDDDataset(Dataset):
    def __init__(self, df, num_frames): # is called only once
        #df_image_paths = get_image_paths(df) #get images paths as dataframe

       # store data directly from dataframe
        self.image_paths = df['filepath'].tolist()
        self.labels = df['yawning'].tolist()

        # transform images to appropriate size upon initialisation
        self.transform = transforms.Compose([
             transforms.ToPILImage(), 
             transforms.Resize((256, 341)),     # resize
             transforms.CenterCrop(224),        # crop
             transforms.ToTensor(),             # back to C×H×W tensor
             #Data Augmentation should be added here
             transforms.Normalize(
                  mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]
             )
        ])

        self.num_frames = num_frames        

    def __len__(self): #return number of entries/labels in the dataset. allwos for batching/iteration
        return len(self.labels)

    def __getitem__(self, idx): #get dataset entrie at idx
        # load one image sequence from path
        image_sequence = load_images_from_path(self.image_paths[idx], num_frames=self.num_frames)
        #apply transofrmation
        images = [self.transform(frame) for frame in image_sequence] 

        # get corresponding label
        label = self.labels[idx]

        return torch.stack(images), torch.tensor(label) #torch.stack converts list of tensors into one, torch.tensor turns scalar labels into a tensorw