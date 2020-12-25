import numpy as np
import pandas as pd
from sklearn.externals import joblib
import gensim
import sys
import pickle
from sklearn.externals.joblib import Parallel, delayed
import multiprocessing


class Dataset(object):

    def __init__(self, **kwargs):
        self.dict = kwargs['dict']
        self.data = kwargs['line']
        self.features_size = 70
        self.context_length = 10
        self.features_vec = np.zeros((self.features_size,))
        self.left_context = []
        self.right_context = []
        self.sentence = []
        self.context = []
        self.mention = []
        self.label = []
        self.doc_2_vec = kwargs['Doc2Vec']
        self.type_length = len(self.dict['label2id'].keys())
        self.types = self.dict['label2id']
        self.doc_vec = np.zeros((50, ))
        # self.word2id = self.dict['word2id']

    def call_line(self):
        start, end, sentence, labels, features = self.data.strip().split('\t')
        start = int(start)
        end = int(end)
        labels, words, features = labels.split(), sentence.split(), features.split()
        tokens = gensim.utils.simple_preprocess(sentence)
        self.doc_vec = self.doc_2_vec.infer_vector(tokens)
        self.left_context = self.get_left(words, start, end)
        self.right_context = self.get_right(words, start, end)
        self.context = self.get_context(words, start, end)
        sent_words = ['<BOS>'] + words + ['<EOS>']
        # print(sent_words)
        self.sentence = self.convert_to_id(sent_words)
        self.label = self.type_idx(labels)
        self.mention = self.get_mention(words, start, end)
        self.features_vec[:len(features)] = self.get_feature_ids(features)

    def get_context(self, line, start, end):
        context = (line[max(0, start - self.context_length): start] +
                   ["PAD"] +
                   line[end: min(len(line), end + self.context_length)])
        return self.convert_to_id(context)

    def get_left(self, line, start, end):
        """Debug"""
        """debugging done adding the mention in left context """
        left_context = line[max(0, start - self.context_length): end+1]
        return self.convert_to_id(left_context)

    def get_right(self, line, start, end):
        """Debug"""
        """ Adding mention in the right context to see if I am able to see some change"""
        right_context = line[start: min(len(line), end + self.context_length)]
        return self.convert_to_id(right_context)

    def convert_to_id(self, token):
        token_id = [self.dict['word2id'][x] for x in token]
        return token_id

    def get_mention(self, line, start, end):
        mention = line[start:end]
        return self.convert_to_id(mention)

    def get_feature_ids(self, feature_in):
        # print(len(feature_in))
        # print(feature_in)
        token_id = [self.dict['feature2id'][x] for x in feature_in]
        if len(token_id) > self.features_size:
            token_id = token_id[:self.features_size]
        return np.array(token_id)

    def type_idx(self, label):
        type_vec = np.zeros((self.type_length,))
        for type_ in label:
            type_ = type_.strip()
            type_idx = self.types[type_]
            type_vec[type_idx] = 1
        return type_vec



def gather(**kwargs):
    train_file = open(kwargs['file_path'], "r")
    dict_data = kwargs['dict']
    doc2vec = kwargs['doc_path']
    train_file = train_file.readlines()
    train_in = []
    for line in train_file:
        data_ = Dataset(line=line, Doc2Vec=doc2vec, dict=dict_data)
        data_.call_line()
        train_in.append(data_)
    return train_in


# if __name__ == '__main__':
#     dict_data_ = joblib.load(sys.argv[2])
#     doc2vec_ = gensim.models.doc2vec.Doc2Vec.load(sys.argv[3])
#     train_dataset = gather(file_path=sys.argv[1], dict=dict_data_, doc_path=doc2vec_)
#     joblib.dump(train_dataset, sys.argv[4])
#
#
#
#     print(train_dataset[0].label)
#     print(train_dataset[0].features_vec)
#     print(train_dataset[0].left_context)
#     print(train_dataset[0].right_context)
#     print(train_dataset[0].context)
#     print(train_dataset[0].sentence)
#     print(train_dataset[0].mention)

