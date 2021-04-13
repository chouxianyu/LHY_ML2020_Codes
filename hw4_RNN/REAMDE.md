- 任务描述

	通过RNN实现文本情感分类(Text Sentiment Classification)。

- 数据集描述

	输入是1个句子，输出是0(负面)或1(正面)。

	训练集：标注数据20万，无标注数据120万

	测试集：20万(无标注)

- 数据格式

	- training_label.txt：`label +++$+++ sentence`，其中`+++$+++`只是分隔符
	- training_nolabel.txt：每一行就是一个句子，没有label
	- testing_data.txt：

- 数据预处理

	一个句子(sentence)中有多个word，我们需要通过**Word Embedding**(我的其它文章里有介绍)用一个vector表示一个word， 然后使用RNN得到一个表示该sentence的vector。

- 半监督学习

	这里使用一种半监督学习方法：**Self-Training**(我的其它文章里有介绍)。使用有标签数据训练好模型，然后对无标签数据进行预测，并根据预测结果对无标签数据进行标注("伪标签")并继续训练模型

- 第三方库

	使用Python第三方库`gensim`实现word2vec模型，以进行Word Embedding。

- 代码

