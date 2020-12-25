# -*- coding: utf-8 -*-
"""
preprocessing file for Fb13 & WN11 for LSTM model
"""

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from collections import Counter, defaultdict
import pickle
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
import pickle as pkl
import random
import itertools
import sys
import random


def get_train_entities(data_in, dataset="WN11"):
    """
    :param data_set: type of dataset
    :param data_in:file read line by
    :return: token list
    """
    token_list = []
    column_name = data_in.columns.tolist()
    for k in range(1, len(column_name)):
        temp = data_in[column_name[k]]
        for entity in temp:
            if data_set == "WN11":
                entity = entity.strip("__")
            x = entity.split("_")
            token_list += x
    list_1 = Counter(token_list)
    token_list = list(list_1.keys())
    return token_list


def word_2_id(token_list):
    """
    :param token_list: list of words
    :return: id_to_word and word_to_id list
    """
    word2ID = defaultdict()
    ID2word = defaultdict()
    i = 1
    for ent in token_list:
        word2ID[ent] = i
        ID2word[i] = ent
        i += 1
    word2ID['unk'] = i
    ID2word[i] = 'unk'
    return word2ID, ID2word


def build_embedding_dict(Model, size, word_to_id):
    """

    :param Model: Glove word2vec model
    :param size: embedding vector dimension size
    :param threshold: number of occurence
    :param word_to_id: word to id dictionary
    :return: embedding matrix, out of vocabulary words
    """
    # unk_count = np.random.normal(0, 1, (size,))
    OOV_count = []
    embedding_matrix = np.zeros((len(word_to_id.keys()) + 1, size))
    for key, value in word_to_id.items():
        if value in Model.wv.vocab.keys():
            embedding_matrix[key] = Model.wv.get_vector(value)
        else:
            embedding_matrix[key] = Model.wv.get_vector("unk")
            OOV_count.append(value)
    return embedding_matrix, OOV_count


def get_rel_dict(head_list, tail_list, rel_list):
    """
    :param head_list: entity list with head, tail, relations
    :param tail_list: tail list
    :param rel_list : relations list
    :return: dictionary of relations mapped to entities
    """
    rel_head = defaultdict(list)
    rel_tail = defaultdict(list)
    for index, entity in enumerate(rel_list):
        print(index, entity)
        print(head_list[index], tail_list[index])
        if head_list[index] not in rel_head[entity]:
            rel_head[entity].append(head_list[index])
        if tail_list[index] not in rel_tail[entity]:
            rel_tail[entity].append(tail_list[index])
    return rel_head, rel_tail


def build_test_dev_vec(data_list, word_to_id, train=False, dataset="WN11"):
    """
    :param data_list: data frame of relation, head and tail
    :param word_to_id: word to id dictionary
    :return: test data build with negative example and positive examples with id values
    """
    # columns = data_list.columns.tolist()
    columns = ["Head", "relation", "tail"]
    build_dict = defaultdict(list)
    for k in range(len(columns)):
        temp_list = []
        a = 0
        for entity in data_list[columns[k]]:
            if dataset == "WN11":
                entity = entity.strip('__')
            temp = []
            ent = entity.split("_")
            for y in ent:
                if y in word_to_id.keys():
                    temp.append(word_to_id[y])
                else:
                    a+=1
                    temp.append(word_to_id['unk'])
            # print(temp, ent, train)
            temp_list.append(temp)
        build_dict[columns[k]] = temp_list
        if not train:
            build_dict['score'] = data_list['score'].tolist()
    print(a)
    return build_dict


def main(*args, **kwargs):
    # store_location = "/home/rpatel12/ferraro_user/NAM_Modified_data/data_sets/WN_11/"
    # import pandas as pd

    # DATA_LOCATION = "/home/rpatel12/ferraro_user/WN11_1/"
    print(args)
    print(type(args))
    DATA_LOCATION = args[0]
    # model_location = "/home/rpatel12/ferraro_user/glove_data/"
    model_location = args[1]
    store_location = DATA_LOCATION
    data_set_type = args[2]
    data_list = pd.read_csv(DATA_LOCATION + "data_list_train.csv")
    print(data_list.head())
    token_list = get_train_entities(data_list, dataset=data_set_type)
    print(len(token_list))
    data_dev = pd.read_csv(DATA_LOCATION + "data_list_dev.csv")
    data_test = pd.read_csv(DATA_LOCATION + "data_list_test.csv")

    glove_file = datapath(model_location + "glove.840B.300d.txt")
    tmp_file = get_tmpfile(model_location + "test_word2vec.txt")
    #
    _ = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)

    # converting word id and id to word
    word_2_id_dict, id_2_word_dict = word_2_id(token_list)

    # building the embedding matrix
    embedding_matrix, OOV_word = build_embedding_dict(model, 300, id_2_word_dict)
    print(OOV_word)
    # build relation to head and relation to tail dictionaries
    head = data_list["Head"].tolist()
    tail = data_list['tail'].tolist()
    rel = data_list['relation'].tolist()

    relation_2_head, relation_2_tail = get_rel_dict(head, tail, rel)
    train_vec_dict = build_test_dev_vec(data_list, word_2_id_dict, train=True, dataset=data_set_type)
    dev_vec_dict = build_test_dev_vec(data_dev, word_2_id_dict, dataset=data_set_type)
    test_vec_dict = build_test_dev_vec(data_test, word_2_id_dict, dataset=data_set_type)

    # saving embedding matrix
    output = open(store_location + "embedding_matrix.pkl", 'wb')
    pickle.dump(embedding_matrix, output, protocol=2)
    output.close()

    # saving train data
    output = open(store_location + "train_vec_dict.pkl", 'wb')
    pickle.dump(train_vec_dict, output, protocol=2)
    output.close()
    # saving dev data
    output = open(store_location + "dev_vec_dict.pkl", 'wb')
    pickle.dump(dev_vec_dict, output, protocol=2)
    output.close()

    # saving test data
    output = open(store_location + "test_vec_dict.pkl", 'wb')
    pickle.dump(test_vec_dict, output, protocol=2)
    output.close()

    # saving word to id dict
    output = open(store_location + "word_2_id.pkl", 'wb')
    pickle.dump(word_2_id_dict, output, protocol=2)
    output.close()
    # saving id to work dict
    output = open(store_location + "id_2_word.pkl", 'wb')
    pickle.dump(id_2_word_dict, output, protocol=2)
    output.close()
    # saving rel to tail
    output = open(store_location + "relation_2_tail.pkl", 'wb')
    pickle.dump(relation_2_tail, output, protocol=2)
    output.close()
    # saving rel to head
    output = open(store_location + "relation_2_head.pkl", 'wb')
    pickle.dump(relation_2_head, output, protocol=2)
    output.close()


if __name__ == "__main__":
    data_location = sys.argv[1]
    glove_model_location = sys.argv[2]
    data_set = sys.argv[3]
    main(data_location, glove_model_location, data_set)
