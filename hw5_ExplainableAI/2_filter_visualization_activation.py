import torch
import matplotlib.pyplot as plt
from dataset import ImgDataset
from model import Model
from torch.optim import Adam

# config
cuda = False
img_indices = [83, 4218, 4707, 8598]  # 选择数据集中的几张图片
layer_idx, filter_idx = 7, 50  # 选择网络中某层的某个filter
learning_rate = 0.1
iterations = 100

# normalize
def normalize(imgs):
    return (imgs - imgs.min()) / (imgs.max() - imgs.min())

# 加载模型
if cuda:
    model = torch.load('../hw3_CNN/model.pth')
else:
    model = torch.load('../hw3_CNN/model.pth', map_location='cpu')
print(model)
# 选择数据集
training_dataset = ImgDataset('../hw3_CNN/data/training', cuda)
images, _ = training_dataset.get_batch(img_indices)


"""Filter Visualization：挑几张图片看看某个filter的输出"""
layer_output = None
model.eval()
# 在hook_func中，我們把第layer_idx层的output通过变量layer_output保存下来
def hook_func(model, input, output):
    global layer_output
    layer_output = output
# 开启hook：当forward过了第layer_idx层后，要先调用我们定义的hook_func，然后才可以继续forward下一层
hook_handle = model.cnn[layer_idx].register_forward_hook(hook_func)
# forward
if cuda:
    model(images.cuda())
else:
    model(images)
# 取出filter的输出
if cuda:
    filter_output = layer_output[:, filter_idx, :, :].detach().cpu()
else:
    filter_output = layer_output[:, filter_idx, :, :].detach()
# Plot Filter Visualization
fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.detach().permute(1, 2, 0))
for i, img in enumerate(filter_output):
  axs[1][i].imshow(normalize(img))
plt.title('Filter Visualization')
plt.show()
plt.close()


"""Filter Activation：看看什么图片可以最大程度地activate该filter"""
if cuda:
    images = images.cuda()
images.requires_grad_()  # 从random noise开始或者从一张数据集中的图片开始都可以
optimizer = Adam([images], lr=learning_rate)  # 定义优化器，只使用梯度更新输入图片
for iter in range(iterations):
    optimizer.zero_grad()
    model(images)
    loss = -layer_output[:, filter_idx, :, :].sum()  # 使得该filter的输出越大越好
    loss.backward()  # 计算梯度(loss关于部分model parameter和输入图片的梯度)
    optimizer.step()  # 更新参数(根据optimizer的定义，所以只会更新images这个变量)
# 关闭hook：hook register之后forward时就都会调用hook_func，所以hook使用完之后就要及时关闭
hook_handle.remove()
# 取出最能activate指定filter的图片
if cuda:
    filter_activation = images.detach().cpu().squeeze()[0]
else:
    filter_activation = images.detach().squeeze()[0]

# Plot Filter Activation
plt.imshow(normalize(filter_activation.permute(1, 2, 0)))
plt.title('Filter Activation')
plt.show()
plt.close()
