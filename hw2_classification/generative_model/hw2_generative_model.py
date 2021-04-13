import numpy as np


## 文件路径
X_train_fpath = '../data/X_train.csv'
Y_train_fpath = '../data/Y_train.csv'
X_test_fpath = '../data/X_test.csv'
output_fpath = 'output.csv'


## 函数定义
# 归一化
def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data
    if specified_column is None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], axis=0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], axis=0).reshape(1, -1)
    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
    return X, X_mean, X_std

# 训练集划分
def _train_valid_split(X, Y, valid_ratio=0.25):
    # This function splits data into training set and validation set.
    train_size = int(len(X) * (1 - valid_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

# sigmoid函数
def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-8, 1 - ( 1e-8))

# forward
def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)

# 预测
def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)

# 计算精度
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    return 1 - np.mean(np.abs(Y_pred - Y_label))


## 读取数据
with open(X_train_fpath) as f:
    next(f) # 不需要第一行的表头
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float) # 不要第一列的ID
    # print(X_train)
with open(Y_train_fpath) as f:
    next(f) # 不需要第一行的表头
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)# 不要第一列的ID，只取第二列
    # print(Y_train)
with open(X_test_fpath) as f:
    next(f) # 不需要第一行的表头
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
    # print(X_test)


## 数据集处理
# 训练集和测试集normalization
X_train, X_mean, X_std = _normalize(X_train, train=True)
X_test, _, _ = _normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)
data_dim = X_train.shape[1]


## 计算每个类别的样本的平均值和协方差
# 区分类别
X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y==0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y==1])
# 计算每个类别的样本的平均值
mean_0 = np.mean(X_train_0, axis=0) # 计算每个维度特征的平均值
mean_1 = np.mean(X_train_1, axis=0)
# 计算每个类别的样本的协方差矩阵（可以研究下协方差矩阵是如何计算的以及为什么）
cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))
for x in X_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0] # transpose没有参数的话，就是转置，计算协方差矩阵时需要转置
for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]
# 计算共享协方差矩阵（Shared covariance is taken as a weighted average of individual in-class covariance）
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])


## 计算权重和偏置
# 计算协方差矩阵的逆矩阵
# Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
# Via SVD decomposition, one can get matrix inverse efficiently and accurately.
u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)
# 计算weight和bias
w = np.dot(inv, mean_0 - mean_1)
b = (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1)) + np.log(float(X_train_0.shape[0])) / X_train_1.shape[0]


## 计算在训练集上的准确率
Y_train_pred = 1 - _predict(X_train, w, b)
print('Training accuracy: {}'.format(_accuracy(Y_train_pred, Y_train)))


## 预测测试集结果
predictions = 1 - _predict(X_test, w, b)
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, label in enumerate(predictions):
        f.write('{},{}\n'.format(i, label))


## 寻找最重要的10个维度的特征
index = np.argsort(np.abs(w))[::-1] # 将w按绝对值从大到小排序
with open(X_test_fpath) as f:
    features = np.array(f.readline().strip('\n').split(','))
    for i in index[:10]:
        print(features[i], w[i])
