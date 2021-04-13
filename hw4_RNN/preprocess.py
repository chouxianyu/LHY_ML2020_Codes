"""
这份代码用來做数据预处理
"""


import torch
import torch.nn as nn
from gensim.models import Word2Vec


class Preprocess:
    def __init__(self, sentences, sen_len, w2v_path='./data/w2v_all.model'):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len  # 将所有sentence都变成这个长度
        self.idx2word = []  # idindex转word
        self.word2idx = {} # word转index
        self.embedding_matrix = [] # embedding的结果

    def get_w2v_model(self):
        # 加载 w2v.py 训练好的 word2vec 模型
        self.embedding = Word2Vec.load(self.w2v_path)  # model
        self.embedding_dim = self.embedding.vector_size  # embedding得到的vector的维度数量
    
    def add_embedding(self, word):
        # 对一个word进行embedding
        # 把 word 添加到 embedding，并赋予它一个随机生成的representation vector
        # word 只会是 "<PAD>" 或 "<UNK>"
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    
    def make_embedding(self, load=True):
        print('Getting embedding ...')
        # 加载 w2v.py 训练好的 word2vec 模型
        if load:
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # for i, word in enumerate(self.embedding.wv.vocab):
        for i, word in enumerate(self.embedding.wv.key_to_index): # https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
            print('Getting word #{}'.format(i+1), end='\r')
            #e.g. self.word2index['he'] = 1
            #e.g. self.index2word[1] = 'he'
            #e.g. self.vectors[1] = 'he' vector
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            # self.embedding_matrix.append(self.embedding[word]) # TypeError: 'Word2Vec' object is not subscriptable
            self.embedding_matrix.append(self.embedding.wv[word]) # https://radimrehurek.com/gensim/models/word2vec.html Usage部分(get numpy vector of a word)
        print()
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 对 "<PAD>" 跟 "<UNK>" 进行embedding
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("Total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # 将每个sentence变成一样的长度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        # 把多个sentence中的word都转成对应的index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 將每個句子變成一樣的長度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        # 把 labels 转成 tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)

