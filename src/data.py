import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchcodec.decoders import VideoDecoder


#Data Handling Functions
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


def load_images_from_path(file_path, num_frames): #get fixed number of frames from a video
    # Convert video into singular frames
    decoder = VideoDecoder(file_path) 

    # Select frames from linearly spaced indices across the whole video/frame sequence
    indices = torch.linspace(0, decoder.metadata.num_frames - 1, num_frames).long()

    #Return only selected raw images frames
    return decoder.get_frames_at(indices=list(indices)).data



#--------------------------------------------------------------------------------------------#

#Dataset Class

class YawDDDataset(Dataset):
    def __init__(self, split, num_frames): # is called only once
        df_image_paths = get_image_paths(split) #get images paths as dataframe

        self.image_paths = df_image_paths['filepath'].tolist() # data_paths for efficient data handling with large datasets, converts df to list
        self.labels = df_image_paths['yawning'].tolist() 

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
        images = [self.transform(frame) for frame in image_sequence] 

        # get corresponding label
        label = self.labels[idx]

        return torch.stack(images), torch.tensor(label) #torch.stack converts list of tensors into one, torch.tensor turns scalar labels into a tensorw