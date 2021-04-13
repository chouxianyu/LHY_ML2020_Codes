import os
import torch
import utils
from preprocess import Preprocess
from model import LSTM_Net
from data import TwitterDataset
from train import training

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3份数据的路径
train_with_label = './data/training_label.txt'
train_no_label = './data/training_nolabel.txt'
testing_data = './data/testing_data.txt'
# word2vec模型的路径
w2v_path = os.path.join('./data/w2v_all.model')

sen_len = 20 # 统一句子长度
fix_embedding = True # 训练时固定embedding模型
batch_size = 128
epoch = 5
lr = 0.001

print("loading data ...") # 读取'training_label.txt'和'training_nolabel.txt'
train_x, y = utils.load_training_data(train_with_label)
train_x_no_label = utils.load_training_data(train_no_label)

# 对input和labels做预处理
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

# 定义model
model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device)

# 将一部分训练集当做验证集
X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]

# 加载数据
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)

# 开始训练
training(batch_size, epoch, lr, './data', train_loader, val_loader, model, device)
