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
from sklearn.externals import joblib


def get_train_entities(data_in, data_set="WN11"):
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
            if column_name[k] == 'relation':
                x = entity.split("_")
                # print(x)
            else:
                x = entity.split('.')
            x = [i for i in x if i != '']
            # print(x)
            final_list = []
            for val in x:
                # print(val)
                final_list += val.split("_")
            token_list += final_list
    # print(x)
    list_1 = Counter(token_list)
    token_list = list(list_1.keys())
    print(token_list)
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


def get_rel_dict(data_in):
    """
    :param head_list: entity list with head, tail, relations
    :param tail_list: tail list
    :param rel_list : relations list
    :return: dictionary of relations mapped to entities
    """
    head_list = data_in["Head"].tolist()
    tail_list = data_in['tail'].tolist()
    rel_list = data_in['relation'].tolist()

    rel_head = defaultdict(list)
    rel_tail = defaultdict(list)
    for index, entity in enumerate(rel_list):
        # print(index, entity)
        # print(head_list[index], tail_list[index])
        if head_list[index] not in rel_head[entity]:
            rel_head[entity].append(head_list[index])
        if tail_list[index] not in rel_tail[entity]:
            rel_tail[entity].append(tail_list[index])
    return rel_head, rel_tail


def build_test_dev_vec(data_list, word_to_id, train=False, dataset="WN18RR"):
    """
    :param data_list: data frame of relation, head and tail
    :param word_to_id: word to id dictionary
    :return: test data build with negative example and positive examples with id values
    """
    # columns = data_list.columns.tolist()
    columns = ["Head", "relation", "tail"]
    build_dict = defaultdict(list)
    a = 0
    for k in range(len(columns)):
        temp_list = []
        for entity in data_list[columns[k]]:
            if dataset == "WN11":
                entity = entity.strip('__')
            temp = []
            if columns[k] == 'relation':
                ent = entity.split("_")
            else:
                ent = entity.split('.')
            ent = [i for i in ent if i != '']
            for y in ent:
                if y in word_to_id.keys():
                    temp.append(word_to_id[y])
                else:
                    a+=1
                    temp.append(word_to_id['unk'])
            temp_list.append(temp)
        build_dict[columns[k]] = temp_list
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
    store_location = DATA_LOCATION + "link_pred"
    data_set_type = args[2]
    data_list = pd.read_csv(DATA_LOCATION + "train.txt", sep='\t', names=["Head", "relation", "tail"])
    print(data_list.head())
    token_list = get_train_entities(data_list, data_set=data_set_type)
    print(len(token_list))
    data_dev = pd.read_csv(DATA_LOCATION + "valid.txt", sep='\t', names=["Head", "relation", "tail"])
    data_test = pd.read_csv(DATA_LOCATION + "test.txt", sep='\t', names=["Head", "relation", "tail"])

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
    print(len(OOV_word))
    # build relation to head and relation to tail dictionaries
    relation_2_head, relation_2_tail = get_rel_dict(data_list)
    rel_2_head_dev, rel_2_tail_dev = get_rel_dict(data_dev)
    rel_2_head_test, rel_2_tail_test = get_rel_dict(data_test)
    train_vec_dict = build_test_dev_vec(data_list, word_2_id_dict,dataset=data_set_type)
    dev_vec_dict = build_test_dev_vec(data_dev, word_2_id_dict, dataset=data_set_type)
    test_vec_dict = build_test_dev_vec(data_test, word_2_id_dict, dataset=data_set_type)

    # saving embedding matrix
    output = open(store_location + "embedding_matrix.pkl", 'wb')
    pickle.dump(embedding_matrix, output, protocol=2)
    output.close()

    joblib.dump(OOV_word, store_location + "OOV_words_list.pkl")
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

    output = open(store_location + "relation_2_tail_dev.pkl", 'wb')
    pickle.dump(rel_2_tail_dev, output, protocol=2)
    output.close()
    # saving rel to head
    output = open(store_location + "relation_2_head_dev.pkl", 'wb')
    pickle.dump(rel_2_head_dev, output, protocol=2)
    output.close()

    output = open(store_location + "relation_2_tail_test.pkl", 'wb')
    pickle.dump(rel_2_tail_test, output, protocol=2)
    output.close()
    # saving rel to head
    output = open(store_location + "relation_2_head_test.pkl", 'wb')
    pickle.dump(rel_2_head_test, output, protocol=2)
    output.close()


if __name__ == "__main__":
    data_location = sys.argv[1]
    glove_model_location = sys.argv[2]
    dataset = sys.argv[3]
    main(data_location, glove_model_location, dataset)
