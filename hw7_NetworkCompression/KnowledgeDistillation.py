from dataset import get_dataloader
import torchvision.models as models
from nets import StudentNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


# config
batch_size = 4
cuda = True
epochs = 200
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
print(batch_size, cuda, epochs, os.environ['CUDA_VISIBLE_DEVICES'])


# 定义Knowledge Distillation中的损失函数
def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 让student的logits做log_softmax后对目标概率(teacher的logits/T后softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss
    # 关于为什么要对student进行logsoftmax：https://github.com/peterliht/knowledge-distillation-pytorch/issues/2
    # 《Distilling the Knowledge in a Neural Network》：https://arxiv.org/abs/1503.02531


# 运行一个epoch
def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空梯度
        optimizer.zero_grad()
        # 获取数据
        inputs, hard_labels = batch_data
        # Teacher不用反向传播，所以使用torch.no_grad()
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            # 使用前面定义的融合soft label&hard label的损失函数：loss_fn_kd，T=20是原论文设定的参数值
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()
        else:
            # 只是做validation的话，就不用计算梯度
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)

        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num


if __name__ == '__main__':
    # 加载数据
    train_dataloader = get_dataloader('../hw3_CNN/data', 'training', batch_size, cuda)
    valid_dataloader = get_dataloader('../hw3_CNN/data', 'validation', batch_size, cuda)
    print('Data Loaded')

    # 加载网络
    teacher_net = models.resnet18(pretrained=False, num_classes=11)
    teacher_net.load_state_dict(torch.load(f'./weights/teacher_resnet18.bin'))
    student_net = StudentNet(base=16)
    if cuda:
        teacher_net = teacher_net.cuda()
        student_net = student_net.cuda()
    print('Model Loaded')

    # 开始训练(Knowledge Distillation)
    print('Training Started')
    optimizer = optim.AdamW(student_net.parameters(), lr=1e-3)
    teacher_net.eval()
    now_best_acc = 0
    for epoch in range(epochs):
        student_net.train()
        train_loss, train_acc = run_epoch(train_dataloader, update=True)
        student_net.eval()
        valid_loss, valid_acc = run_epoch(valid_dataloader, update=False)

        # 存下最好的model
        if valid_acc > now_best_acc:
            now_best_acc = valid_acc
            torch.save(student_net.state_dict(), './weights/student_model.bin')
        print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc))
