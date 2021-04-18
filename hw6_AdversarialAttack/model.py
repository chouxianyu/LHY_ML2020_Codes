import torchvision.models as models
from dataset import AdverDataset
import torch
import torch.nn.functional as F
"""使用FGSM实现黑盒攻击"""


class Attacker:
    def __init__(self, img_dir_path='./data/images', label_csv_path='./data/labels.csv', cuda=False):
        self.cuda = cuda
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        self.dataset = AdverDataset(img_dir_path, label_csv_path, self.normalize_mean, self.normalize_std, self.cuda)
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False
        )
        self.model = models.vgg16(pretrained=True) # 选择Proxy Network
        self.model.eval()
        if self.cuda:
            self.model = self.model.cuda()

    def perturb(self, image, eps, grad):
        """使用FGSM进行攻击"""
        sign_grad = grad.sign()  # 找出梯度的方向
        perturbed_image = image + eps * sign_grad  # 扰动图像
        return perturbed_image

    def attack(self, epsilon):
        pred_wrong_cnt = 0 # Proxy Network预测错误的图片数量
        attack_fail_cnt = 0  # 攻击失败的图片数量
        attack_succeed_cnt = 0  # 攻击成功的图片数量
        attack_results = []
        for (image, label) in self.data_loader:
            img = image # 备份一下image
            image.requires_grad = True
            preds = self.model(image)
            pred = preds.max(1, keepdim=True)[1]
            if pred.item() != label.item():  # 如果Proxy Network预测错误则不攻击
                pred_wrong_cnt += 1
                continue
            loss = F.nll_loss(preds, label) # 计算loss
            self.model.zero_grad()
            loss.backward() # 计算梯度
            grad = image.grad.data # 取出loss关于image的梯度
            perturbed_image = self.perturb(image, epsilon, grad) # 用FGSM扰动图片
            perturbed_pred = self.model(perturbed_image).max(1, keepdim=True)[1]
            if perturbed_pred.item() == label.item(): # 攻击失败
                attack_fail_cnt += 1
            else: # 攻击成功
                attack_succeed_cnt += 1
                if (len(attack_results)<5):
                    std = torch.tensor(self.normalize_std).view(3, 1, 1)
                    mean = torch.tensor(self.normalize_mean).view(3, 1, 1)
                    if self.cuda:
                        img = img.cpu()
                        perturbed_image = perturbed_image.cpu()
                    img = img*std+mean
                    img = img.squeeze().detach().numpy()
                    perturbed_image = perturbed_image*std+mean
                    perturbed_image = perturbed_image.squeeze().detach().numpy()
                    attack_results.append((pred.item(), perturbed_pred.item(), img, perturbed_image))
        accuracy = (attack_fail_cnt / (pred_wrong_cnt + attack_fail_cnt + attack_succeed_cnt)) # 经过攻击后Proxy Network的准确率
        print('Original Proxy Network\t Accuracy: {}'.format((attack_fail_cnt+attack_succeed_cnt)/self.dataset.__len__()))
        print('After Attack(epsilon: {})\t Accrucy: {}'.format(epsilon, accuracy))
        return attack_results, accuracy
