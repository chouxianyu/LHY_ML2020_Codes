import os
import cv2
import time
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.pooling import MaxPool1d, MaxPool2d
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np


"""加载数据"""
def read_files(dir_path): # 读取文件夹中的所有图片
    filenames = sorted(os.listdir(dir_path))
    x = np.zeros((len(filenames), 128, 128, 3), dtype=np.uint8) # (N,H,W,C)
    y = np.zeros((len(filenames)), dtype=np.uint8)
    for i, filename in enumerate(filenames):
        img = cv2.imread(os.path.join(dir_path, filename))
        x[i, : , :] = cv2.resize(img, (128, 128))
        y[i] = int(filename.split("_")[0])
    return x, y

train_x, train_y = read_files("./data/training")
val_x, val_y = read_files("./data/validation")
print("Data Loaded")
print("Size of training data : %d" % len(train_x))
print("Size of validation data : %d" % len(val_x))


"""数据变换（训练时进行数据增强）"""
train_transform = transforms.Compose([
    transforms.ToPILImage(mode=None), # 将图片格式转换成PIL格式
    transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
    transforms.RandomRotation(15), # 随机旋转图片
    transforms.ToTensor(), # 转换成torch中的tensor并将值normalize到[0.0,1.0]
])
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


"""加载数据"""
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        return X

batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, val_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


"""定义模型"""
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1) # torch.nn只支持mini-batches而不支持单个sample，第1个维度是mini-batch中图片（特征）的索引，即将每张图片都展开
        return self.fc(x)


"""训练并测试模型"""
model = Model() # model = Model().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 30
for epoch in range(epochs):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # batch_loss.backward()的gradient会累加，所以每个batch都需要置零
        pred = model(data[0]) # pred = model(data[0].cuda())
        batch_loss = criterion(pred, data[1]) # batch_loss = criterion(pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(pred.detach().numpy(), axis=1) == data[1].numpy())
        # train_acc += np.sum(np.argmax(pred.cpu().detach().numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            pred = model[data[0]] # pred = model(data[0].cuda())
            batch_loss = criterion(pred, data[1]) # batch_loss = criterion(pred, data[1].cuda())
            val_acc += np.sum(np.argmax(pred.detach().numpy(), axis=1) == data[1].numpy())
            # val_acc += np.sum(np.argmax(pred.cpu().detach().numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
    
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch+1, epochs, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
