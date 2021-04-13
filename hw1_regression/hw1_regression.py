import pandas as pd
import numpy as np
import csv

## 读取数据
data = pd.read_csv('./train.csv', encoding = 'big5') # 读取训练集
# print(data.describe())

## 数据预处理
data = data.iloc[:, 3:] # 不需要使用前三列的表头，所以删除
data[data == 'NR'] = 0 # 将非数值NR改为0
raw_data = data.to_numpy() # pandas转numpy数组，形状是4320(=18*20*12)*24
# print(raw_data.shape)

## 修改数据格式
# 数据格式为12(month)*18(features)*480(=24*20hours)，即12个月、每个月有480小时的数据（18维）
month_data = {} # 字典
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[ : , 24 * day : 24 * (day + 1)] = raw_data[(month * 20 + day) * 18 : (month * 20 + day + 1) * 18, : ]
    month_data[month] = sample

## 修改数据格式
# 数据格式为每个月有连续的480个小时，每10个小时形成1个object，每个月就有471个object，12个月就有471*12个oeject，每个object包括x(18*9的featrues)和y(1个PM2.5数值)。
x = np.empty([471*12, 18*9], dtype=float) # 471*12行，一行是一个object的x
y = np.empty([471*12, 1], dtype=float) # 471*12行，一行是一个object的y
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14: # 最后一个10小时从第20天14小时开始，防止越界
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) # reshape时的(1, -1)指：1行、列数自动计算
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] # 取对应的第10个小时的PM2.5的值
# print(x, y)

## 标准化
#关于标准化，可以看这篇文章https://www.cnblogs.com/chouxianyu/p/13872444.html
mean_x = np.mean(x, axis=0) # 平均值，axis=0指沿着列计算平均值，即计算每列的平均值
std_x = np.std(x, axis=0) # 标准差，axis=0指沿着列计算平均值，即计算每列的标准差
# print(mean_x.shape, std_x.shape)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

## 训练
dim = 18 * 9 + 1 # 这个+1是为了保存偏置
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([471 * 12, 1]), x), axis=1).astype(float) # axis=1表示将两个数组按行拼接，向x中添加1是为了让其与weight中的偏置相乘
learning_rate = 100 # 学习率
iter_time = 1000 # 迭代次数
adagrad = np.zeros([dim, 1])
eps = 1e-10  # eps是避免Adagrad分母为0而加的
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12) # RMSE
    if (t % 100 == 0):
        print(t, loss)
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) # dim*1
    adagrad += gradient ** 2
    w -= learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)
print('Training Done')

## 测试
# 读取数据
test_data = pd.read_csv('./test.csv',header=None, encoding='big5')
test_data = test_data.iloc[ : , 2:] # 去除表头（前两列）
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18 * 9]) # 240个object，一行是一个object的x
# 修改数据格式
for i in range(240):
    test_x[i, :] = test_data[i * 18 : (i + 1) * 18, : ].reshape(1, -1) # 格式和训练集一样
# 标准化
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i, j] = (test_x[i, j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float) # axis=1表示将两个数组按行拼接，向x中添加1是为了让其与weight中的偏置相乘

## 预测
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
# print('ans_y.shape', ans_y.shape)
with open('answer.csv', mode='w', newline='') as answer_file:
    csv_writer = csv.writer(answer_file)
    csv_writer.writerow(['id', 'value'])
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        # print(row)
