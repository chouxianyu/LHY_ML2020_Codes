"""
这份代码用来对testing_data.txt进行预测
"""
import torch


def testing(test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= 0.5] = 1  # 大于等于0.5为正面
            outputs[outputs < 0.5] = 0  # 小于0.5为负面
            ret_output += outputs.int().tolist()

    return ret_output
