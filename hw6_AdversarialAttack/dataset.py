from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
import torch


"""读取图片和标签"""
class AdverDataset(Dataset):
    def __init__(self, img_dir_path, label_csv_path, normalize_mean, normalize_std, cuda=False):
        self.cuda = cuda
        self.img_dir_path = img_dir_path
        self.label_csv_path = label_csv_path
        self.transform = transforms.Compose([                
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std, False) # Normalize到[0,1]
        ])
        self.img_names = sorted(os.listdir(img_dir_path))
        df = pd.read_csv(self.label_csv_path)
        df = df.loc[:, 'TrueLabel'].to_numpy()
        self.labels = torch.from_numpy(df).long()

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir_path, self.img_names[idx]))
        img = self.transform(img)
        label = self.labels[idx]
        if self.cuda:
            return img.cuda(), label.cuda()
        else:
            return img, label

    def __len__(self):
        return len(self.img_names)
