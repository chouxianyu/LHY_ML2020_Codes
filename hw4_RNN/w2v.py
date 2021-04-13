"""
这份代码用来训练实现word to vector的word embedding模型
注意！这里在训练word to vector时使用cpu，可能要花10分钟以上
如果出现AttributeError: module 'numpy.random' has no attribute 'default_rng'，根据https://blog.csdn.net/weixin_45195364/article/details/115386051可以更新numpy
"""
import os
from gensim.models import word2vec
from utils import load_training_data, load_testing_data

def train_word2vec(x):
    # 训练实现word to vector的word embedding模型
    # model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1) # 因为Word2Vec版本更新这行代码会报错，见https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
    model = word2vec.Word2Vec(x, vector_size=250, window=5, min_count=5, workers=12, epochs=10, sg=1)
    return model


if __name__ == "__main__":
    print("loading training data ...")
    train_x, y = load_training_data('./data/training_label.txt')
    train_x_no_label = load_training_data('./data/training_nolabel.txt')

    print("loading testing data ...")
    test_x = load_testing_data('./data/testing_data.txt')

    print("training model ...")
    model = train_word2vec(train_x + train_x_no_label + test_x)
    # model = train_word2vec(train_x + test_x)
    
    print("saving model ...")
    model.save('./data/w2v_all.model')
