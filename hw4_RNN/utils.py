"""
这份代码定义一些常用到的函数
"""
import torch

def load_training_data(path):
    with open(path, mode='r', encoding='UTF-8') as f:
        if 'training_label' in path:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
            x = [line[2:] for line in lines]
            y = [line[0] for line in lines]
            return x, y
        elif 'training_nolabel' in path:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
            return x


def load_testing_data(path='./data/testing_data.txt'):
    with open(path, mode='r', encoding='UTF-8') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sentence.split(' ') for sentence in X]
        return X


def evaluation(outputs, labels):
    outputs[outputs >= 0.5] = 1  # 大于等于0.5为正面
    outputs[outputs<0.5] = 0 # 小于0.5为负面
    correct_cnt = torch.sum(torch.eq(outputs, labels)).item()
    return correct_cnt
