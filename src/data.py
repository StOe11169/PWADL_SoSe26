import torch
from torch.utils.data import Dataset


class YawDDDataset(Dataset):
    def __init__(self): # is called only once
        # self.image_paths # data_paths for efficient data handling with large datasets
        # self.labels
        pass

    # def __len__(self):
    #     return len(self.labels)

    # def __getitem__(self, idx): # is called multiple times during training and evaluation and should be written efficiently
    #     # load image from path
    #     # apply transforms (resize, CenterCrop, normalization, ToTensor, data augmentation operations, ...)
    #     return #images[idx], labels[idx]