import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class ImgDataset(Dataset):
    def __init__(self, dir_path, cuda):
        self.cuda = cuda
        self.img_names = sorted(os.listdir(dir_path))
        self.img_paths = [os.path.join(dir_path, img_name) for img_name in self.img_names]
        self.transform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, i):
        X = self.transform(Image.open(self.img_paths[i]))
        Y = int(self.img_names[i].split('_')[0])
        if self.cuda:
            return X.cuda(), Y.cuda()
        else:
            return X, Y

    def get_batch(self, indices):
        # 这个方法用于取出某几张图片以方便进行可视化
        images = []
        labels = []
        for index in indices:
          image, label = self.__getitem__(index)
          images.append(image)
          labels.append(label)
        return torch.stack(images), torch.tensor(labels)
