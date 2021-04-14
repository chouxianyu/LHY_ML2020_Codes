# 任务描述

在homework3中我们通过CNN实现了食物图片分类，这次作业的任务就是探究这个CNN的可解释性，具体如下

1. **Saliency Map**

	按照《Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps》，计算每个像素对最终分类结果的重要性。

	我們把一张图片输入到model，将model的输出与label进行对比计算得到loss，因此与loss相关的变量有image、model parameter和label这3者。

	通常情況下，我們训练模型时希望找到一组好的model parameter来拟合image和label，因此loss在backward时我们只在乎loss关于model parameter的梯度。但在数学上image本身也是continuous tensor，我们可以**在model parameter和label都固定的情况下计算loss关于image的梯度，这个梯度代表稍微改变image的某个pixel value会对loss产生什么影响，我们习惯把这个影响的程度解读为该pixel对于结果的重要性（每个pixel都有自己的梯度）**。

	因此将loss关于一张图片中每个pixel的梯度计算并画出来，就可以看出该图中哪些像素是model在计算结果时的重要依据。那如何用代码实现我们的这个想法呢？非常简单，在一般训练中我们都是在forward后计算模型输出与标签之间的loss，然后进行loss的backward，其实在PyTorch中这个backword计算的是loss对**model parameter**的梯度，因此我們只需要用一行代码`images.requires_grad_()`使得**image**也要被计算梯度。

2. **Filter Visualization**

	基于Gradient Ascent，实现Activation maximization，找到最能够激活某个filter的图片，以观察模型学到了什么。

	这里我们想要知道某一个filter到底学习到了什么，我们需要做两件事情：**①Filter Visualization：挑几张图片看看某个filter的输出；②Filter Activation：看看什么图片可以最大程度地activate该filter**。

	在代码实现方面，我们一般是直接把图片输入到model，然后直接forward，那要如何取出model中某层的输出呢？虽然我们可以直接修改model的forward函数使其返回某层的输出，但这样比较麻烦，还可能会因此改动其它部分的代码。因此PyTorch提供了方便的解决方法：**hook**。

3. **LIME**

	绿色代表一个component和结果正相关，红色则代表该component和结果负相关。

	《"Why Should I Trust You?”: Explaining the predictions of Any Classifier》

	注：根据助教的示例，我遇到了一个BUG`KeyError: 'Label not in explanation'`，暂未解决……

# 数据集描述

使用homework3使用的数据集以及训练出的CNN模型。

# 其它

https://github.com/utkuozbulak/pytorch-cnn-visualizations