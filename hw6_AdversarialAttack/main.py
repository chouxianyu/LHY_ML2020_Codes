from PIL.Image import Image
import pandas as pd
from model import Attacker
import matplotlib.pyplot as plt
import numpy as np
import os


# config
cuda = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 加载label名称，即category
category = pd.read_csv("./data/categories.csv")
category = category.loc[:, 'CategoryName'].to_numpy()

# 攻击
attacker = Attacker(cuda=cuda)
epsilons = [0.1, 0.01]
accuracies, examples = [], []
for eps in epsilons:
    ex, acc = attacker.attack(eps)
    accuracies.append(acc)
    examples.append(ex)

# 攻击样例可视化
cnt = 0
plt.figure(figsize=(30, 30))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        pred, perturbed_pred, image, perturbed_image = examples[i][j]
        # 原图
        cnt+=1
        plt.subplot(len(epsilons), len(examples[0]) * 2, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("eps: {}".format(epsilons[i]), fontsize=14)
        plt.title("original: {}".format(category[pred].split(',')[0]))
        plt.imshow(np.transpose(image, (1, 2, 0)))
        # 攻击结果
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]) * 2, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title("attack: {}".format(category[perturbed_pred].split(',')[0]))
        plt.imshow(np.transpose(perturbed_image, (1, 2, 0)))
plt.tight_layout()
plt.show()
