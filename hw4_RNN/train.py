"""
这份代码用来训练RNN
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import evaluation


def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train()
    criterion = nn.BCELoss()
    t_batch = len(train) # 训练集batch数量
    v_batch = len(valid) # 验证集batch数量
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0 # 该epoch的loss和acc
        # 训练
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float) # 因为要计算loss，所以类型要是float
            optimizer.zero_grad() # loss.backward()的gradient会累加，所以每个batch都需要置零
            outputs = model(inputs) # 得到输出
            outputs = outputs.squeeze() # 删除为1的dimension
            loss = criterion(outputs, labels) # 计算loss
            loss.backward() # 计算gradient
            optimizer.step() # 更新参数
            correct = evaluation(outputs, labels) # 计算training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))

        # 验证
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float) # 因为要计算loss，所以类型要是float
                outputs = model(inputs)
                outputs = outputs.squeeze()  # 删除为1的dimension
                loss = criterion(outputs, labels) # 计算loss
                correct = evaluation(outputs, labels)  # 计算validation accuracy
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                # 如果 validation 的结果优于之前所有的结果，就把该模型存下来
                best_acc = total_acc
                #torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, "{}/ckpt.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        print('-----------------------------------------------')
        model.train()
