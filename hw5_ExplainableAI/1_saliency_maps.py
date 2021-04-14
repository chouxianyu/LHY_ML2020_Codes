import torch
import matplotlib.pyplot as plt
from dataset import ImgDataset
from model import Model


"""Plot Saliency Map"""


cuda = False
img_indices = [83, 4218, 4707, 8598]  # 选择数据集中的几张图片

# 加载模型
if cuda:
    model = torch.load('../hw3_CNN/model.pth')
else:
    model = torch.load('../hw3_CNN/model.pth', map_location='cpu')
# 选择数据集
training_dataset = ImgDataset('../hw3_CNN/data/training', cuda)
images, labels = training_dataset.get_batch(img_indices)

# 计算Loss并求导
model.eval()
images.requires_grad_() # 追踪loss关于images的梯度
y_pred = model(images)
loss_func = torch.nn.CrossEntropyLoss()
loss = loss_func(y_pred, labels)
loss.backward()
# 取出loss关于images的梯度
if cuda:
    saliencies = images.grad.abs().detach().cpu()
else:
    saliencies = images.grad.abs().detach()
saliencies = torch.stack([(item - item.min()) / (item.max() - item.min()) for item in saliencies]) # normalize

# 画出图片及其关于loss的梯度
fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for row, target in enumerate([images, saliencies]):
  for column, img in enumerate(target):
    axs[row][column].imshow(img.permute(1, 2, 0).detach().numpy())
plt.show()
plt.close()
