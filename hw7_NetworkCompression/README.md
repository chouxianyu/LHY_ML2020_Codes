# 任务描述

通过Architecture Design、Knowledge Distillation、Network Pruning和Weight Quantization这4种模型压缩策略，用一个非常小的model完成homework3中食物图片分类的任务。

## 1.Architecture Design

MobileNet提出了Depthwise & Pointwise Convolution。我们在这里实现MobileNet v1这个比较小的network，后续使用Knowledge Distillation策略训练它，然后对它进行剪枝和量化。

## 2.Knowledge Distillation

将ResNet18作为Teacher Net(使用torchvision中的ResNet18，仅将num_classes改成11，加载助教训练好的Accuracy约为88.4%的参数)，将上一步(1.Architecture Design)设计的小model作为Student Net，使用Knowledge_Distillation策略训练Student Net。

Loss计算方法为$Loss = \alpha T^2 \times KL(\frac{\text{Teacher's Logits}}{T} || \frac{\text{Student's Logits}}{T}) + (1-\alpha)(\text{Original Loss})$，关于为什么要对student进行logsoftmax可见https://github.com/peterliht/knowledge-distillation-pytorch/issues/2

论文《Distilling the Knowledge in a Neural Network》：https://arxiv.org/abs/1503.02531

## 3.Network Pruning

对上一步(2.Knowledge_Distillation)训练好的Student Net做剪枝。

根据论文《Learning Efficient Convolutional Networks through Network Slimming》，论文链接：https://arxiv.org/abs/1708.06519
BatchNorm层中的gamma值和一些特定卷积核（或者全连接层的一个神经元）相关联，因此可以使用BatchNorm层中的gamma值判断相关通道的重要性。

Student Net中CNN部分有几个结构相同的Sequential，其结构、权重名称、实现代码、权重形状如下表所示。

|  #   | name      | meaning               | code                              | weight shape |
| :--: | :-------- | :-------------------- | --------------------------------- | ------------ |
|  0   | cnn.{i}.0 | Depthwise Convolution | nn.Conv2d(x, x, 3, 1, 1, group=x) | (x, 1, 3, 3) |
|  1   | cnn.{i}.1 | Batch Normalization   | nn.BatchNorm2d(x)                 | (x)          |
|  2   |           | ReLU6                 | nn.ReLU6                          |              |
|  3   | cnn.{i}.3 | Pointwise Convolution | nn.Conv2d(x, y, 1),               | (y, x, 1, 1) |
|  4   |           | MaxPooling            | nn.MaxPool2d(2, 2, 0)             |              |

独立剪枝prune_count次，每次剪枝的剪枝率按prune_rate逐渐增大，剪枝后微调finetune_epochs个epoch。

## 4.Weight Quantization

对第二步(2.Knowledge_Distillation)训练好的Student Net做量化（用更少的bit表示一个value）。

torch预设的FloatTensor是32bit，而FloatTensor最低可以是16bit。

如何将32bit转成8bit的int呢？对每个weight进行min-max normalization，然后乘以$2^8-1$再四舍五入成整数，这样就可以转成uint8了。

# 数据集描述

数据集为homework3中食物图片分类数据集。

11个图片类别，训练集中有9866张图片，验证集中有3430张图片，测试集中有3347张图片。

训练集和验证集中图片命名格式为`类别_编号.jpg`，编号不重要。

# 代码

https://github.com/chouxianyu/LHY_ML2020_Codes/tree/master/hw7_NetworkCompression
