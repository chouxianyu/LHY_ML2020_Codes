import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os

class FoodDataset(Dataset):
    def __init__(self, dir_path, transform, cuda=False):
        self.cuda = cuda
        self.transform = transform
        self.x = []
        self.y = []
        img_names = sorted(os.listdir(dir_path))
        for img_name in img_names:  # glob返回匹配到的所有文件的路径
            img_path = os.path.join(dir_path, img_name)
            label = int(img_name.split("_")[0])

            image = Image.open(img_path)
            # Get File Descriptor
            image_fp = image.fp
            image.load()
            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            self.x.append(image)
            self.y.append(label)

    def __getitem__(self, idx):
        image = self.transform(self.x[idx])
        label = torch.torch.tensor(self.y[idx], dtype=torch.int64)
        if self.cuda:
            image = image.cuda()
            label = label.cuda()
        return image, label
    
    def __len__(self):
        return len(self.x)


trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])


def get_dataloader(dir_path='../hw3_CNN/data', mode='training', batch_size=32, cuda=False):

    assert mode in ['training', 'testing', 'validation']

    dataset = FoodDataset(
        f'{dir_path}/{mode}',
        transform=trainTransform if mode == 'training' else testTransform, cuda=cuda)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader
