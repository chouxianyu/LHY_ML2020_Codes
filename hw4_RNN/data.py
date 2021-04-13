"""
自定义了dataset的'__init__', '__getitem__', '__len__'，使其可被dataloader直接调用
"""


import torch
from torch.utils.data import Dataset


class TwitterDataset(Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)
    __len__ will return the number of data
    """
    def __init__(self, X, y):
        self.data = X
        self.label = y
    
    def __getitem__(self, idx):
        if self.label is None:
            return self.data[idx]
        else:
            return self.data[idx], self.label[idx]
    
    def __len__(self):
        return len(self.data)
