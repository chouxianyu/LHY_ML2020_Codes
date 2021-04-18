- 任务描述

  选择一个Proxy Network实现**Black Box** Attack，通过FGSM(Fast Gradient Sign Method)实现Non-targeted Adversial Attack。

- 数据集描述

  有200张图片，命名格式为`编号.png`，尺寸为224×224。

  categories.csv：1000个类别，索引为[0,999]，

  labels.csv：每张图片的信息(包括类别索引)

- 评估指标

	- 所有输入图片$x^0$和攻击图片$x'$的L-infinity的平均值
	- 攻击的成功率

- 代码

	https://github.com/chouxianyu/LHY_ML2020_Codes/tree/master/hw6_AdversarialAttack
	
- 结果

	```python
	Original Proxy Network	 Accuracy: 0.865
	After Attack(epsilon: 0.1)	 Accrucy: 0.03
	Original Proxy Network	 Accuracy: 0.865
	After Attack(epsilon: 0.01)	 Accrucy: 0.27
	```

	使用预训练的VGG16作为Proxy Network，可知在攻击前Proxy Nerwork的准确率为0.865，而攻击后准确率为0.03(epsilon为0.1)、0.27(epsilon为0.01)

