import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class his_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __getitem__(self, index):
        return self.data[index],self.label[index]
    def __len__(self):
        return self.data.shape[0]


        
