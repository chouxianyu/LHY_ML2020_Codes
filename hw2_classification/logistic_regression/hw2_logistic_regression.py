import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0) # 使每次随机生成的数字相同


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

# 数据打乱
def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

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

# 交叉熵损失函数
def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    return -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))

# 梯度计算
def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    Y_pred = _f(X, w, b)
    pred_error = Y_label - Y_pred
    w_grad = -np.sum(pred_error * X.T, axis=1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad


## 文件路径
X_train_fpath = '../data/X_train.csv'
Y_train_fpath = '../data/Y_train.csv'
X_test_fpath = '../data/X_test.csv'
output_fpath = 'output.csv'


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
# 训练集验证集划分
X_train, Y_train, X_valid, Y_valid = _train_valid_split(X_train,Y_train, valid_ratio=0.1)
train_size = X_train.shape[0]
valid_size = X_valid.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of validation set: {}'.format(valid_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))

## 训练（使用小批次梯度下降法，Mini-batch training）
# 参数初始化
w = np.zeros((data_dim, ))
b = np.zeros((1, ))
# 训练参数
max_iter = 10
batch_size = 8
learning_rate = 0.2
# 保存每个epoch的loss以作图
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []
step = 1
# 迭代
for epoch in range(max_iter):
    # 打乱训练集
    X_train, Y_train = _shuffle(X_train, Y_train)
    # Mini-batch training
    for idx in range(int(np.floor(X_train.shape[0] / batch_size))):
        # 取batch
        X = X_train[idx * batch_size : idx * batch_size + batch_size]
        Y = Y_train[idx * batch_size : idx * batch_size + batch_size]
        # 计算梯度
        w_grad, b_grad = _gradient(X, Y, w, b)
        # 梯度下降（learning rate decay with time）
        w = w - learning_rate / np.sqrt(step) * w_grad
        b = b - learning_rate / np.sqrt(step) * b_grad
        step = step + 1
    # 计算训练集和验证集的loss和精度
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)
    y_valid_pred = _f(X_valid, w, b)
    Y_valid_pred = np.round(y_valid_pred)
    valid_acc.append(_accuracy(Y_valid_pred, Y_valid))
    valid_loss.append(_cross_entropy_loss(y_valid_pred, Y_valid) / valid_size)
print('Training loss: {}'.format(train_loss[-1]))
print('Validation loss: {}'.format(valid_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Validation accuracy: {}'.format(valid_acc[-1]))


## 训练过程可视化
# loss可视化
plt.plot(train_loss)
plt.plot(valid_loss)
plt.title('Loss')
plt.legend(['train', 'valid'])
plt.savefig('Loss.png')
plt.show()
# accuracy可视化
plt.plot(train_acc)
plt.plot(valid_acc)
plt.title('Accuracy')
plt.legend(['train', 'valid'])
plt.savefig('Accuracy.png')
plt.show()


## 预测测试集
predictions = _predict(X_test, w, b)
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
