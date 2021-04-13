- 任务描述（Task Description）

	现在有某地空气质量的观测数据，请使用线性回归拟合数据，预测PM2.5。

- 数据集描述（Dataset Description）

	- train.csv

		该文件中是2014年每月前20天每小时的观察数据，每小时的数据是18个维度的（其中之一是PM2.5）。

	- test.csv

		该文件中包含240组数据，每组数据是连续9个小时的所有观测数据（同样是18个维度）。

		请预测每组数据对应的第10个小时的PM2.5数值。

- 结果格式

	要求上交结果的格式为CSV文件。

	第一行必须是`id,value`。

	从第二行开始每行分别为id值及预测的PM2.5数值，两者用逗号间隔

- 总结

	- 数据处理
		- 将数据处理、转换成什么形式，要根据数据集格式、任务来确定。
		- 要熟练掌握pandas、numpy等数据处理工具，特别是要知道它们能实现什么功能。

- 参考链接

	https://colab.research.google.com/drive/131sSqmrmWXfjFZ3jWSELl8cm0Ox5ah3C