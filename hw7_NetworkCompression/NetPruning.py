from dataset import get_dataloader
from nets import StudentNet
import torch
import torch.nn as nn
import torch.optim as optim
import os


def network_slimming(old_model, new_model):
    old_params = old_model.state_dict()
    new_params = new_model.state_dict()

    # 只保留每一层中的部分卷积核
    selected_idx = []
    for i in range(8):  # 只对模型中CNN部分(8个Sequential)进行剪枝
        gamma = old_params[f'cnn.{i}.1.weight']
        new_dim = len(new_params[f'cnn.{i}.1.weight'])
        ranking = torch.argsort(gamma, descending=True)
        selected_idx.append(ranking[:new_dim])

    now_processing = 1  # 当前在处理哪一个Sequential，索引为0的Sequential不需处理
    for param_name, weights in old_params.items():
        # 如果是CNN层，则根据gamma仅复制部分参数；如果是FC层或者该参数只有一个数字(例如batchnorm的tracenum等等)就直接全部复制
        if param_name.startswith('cnn') and weights.size() != torch.Size([]) and now_processing != len(selected_idx):
            # 当处理到Pointwise Convolution时，则代表正在处理的Sequential已处理完毕
            if param_name.startswith(f'cnn.{now_processing}.3'):
                now_processing += 1

            # Pointwise Convolution的参数会受前一个Sequential和后一个Sequential剪枝情况的影响，因此需要特别处理
            if param_name.endswith('3.weight'):
                # 不需要删除最后一个Sequential中的Pointwise卷积核
                if len(selected_idx) == now_processing:
                    # selected_idx[now_processing-1]指当前Sequential中保留的通道的索引
                    new_params[param_name] = weights[:,selected_idx[now_processing-1]]
                # 除了最后一个Sequential，每个Sequential中卷积核的数量(输出通道数)都要和后一个Sequential匹配。
                else:
                    # Pointwise Convolution中Conv2d(x,y,1)的weight的形状是(y,x,1,1)
                    # selected_idx[now_processing]指后一个Sequential中保留的通道的索引
                    # selected_idx[now_processing-1]指当前Sequential中保留的通道的索引
                    new_params[param_name] = weights[selected_idx[now_processing]][:,selected_idx[now_processing-1]]
            else:
                new_params[param_name] = weights[selected_idx[now_processing]]
        else:
            new_params[param_name] = weights
    
    # 返回新模型
    new_model.load_state_dict(new_params)
    return new_model


def run_epoch(dataloader):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空 optimizer
        optimizer.zero_grad()
        # 获取数据
        inputs, labels = batch_data
  
        logits = new_net(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_hit += torch.sum(torch.argmax(logits, dim=1) == labels).item()
        total_num += len(inputs)
        total_loss += loss.item() * len(inputs)

    return total_loss / total_num, total_hit / total_num


# config
batch_size = 4
cuda = True
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
prune_count = 5
prune_rate = 0.95
finetune_epochs = 5
print(batch_size, cuda, os.environ['CUDA_VISIBLE_DEVICES'], prune_count, prune_rate, finetune_epochs)


if __name__ == '__main__':
    # 加载数据
    train_dataloader = get_dataloader('../hw3_CNN/data', 'training', batch_size, cuda)
    valid_dataloader = get_dataloader('../hw3_CNN/data', 'validation', batch_size, cuda)
    print('Data Loaded')

    # 加载网络
    old_net = StudentNet()
    if cuda:
        old_net = old_net.cuda()
    old_net.load_state_dict(torch.load('./weights/student_model.bin'))

    # 开始剪枝并finetune：独立剪枝prune_count次，每次剪枝的剪枝率按prune_rate逐渐增大，剪枝后微调finetune_epochs个epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(old_net.parameters(), lr=1e-3)

    now_width_mult = 1
    for i in range(prune_count):
        now_width_mult *= prune_rate # 增大剪枝率
        new_net = StudentNet(width_mult=now_width_mult)
        if cuda:
            new_net = new_net.cuda()
        new_net = network_slimming(old_net, new_net)
        now_best_acc = 0
        for epoch in range(finetune_epochs):
            new_net.train()
            train_loss, train_acc = run_epoch(train_dataloader)
            new_net.eval()
            valid_loss, valid_acc = run_epoch(valid_dataloader)
            # 每次剪枝时存下最好的model
            if valid_acc > now_best_acc:
                now_best_acc = valid_acc
                torch.save(new_net.state_dict(), f'./weights/pruned_{now_width_mult}_student_model.bin')
            print('rate {:6.4f} epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(now_width_mult, 
                epoch, train_loss, train_acc, valid_loss, valid_acc))
