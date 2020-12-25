import pandas as pd
import numpy as np
from sklearn.externals import joblib
from collections import defaultdict, Counter
import sys
import string
import os
import random
import itertools

def get_tokens(val):
    val = val.lower()
    token_list = []
    if "_" in val:
        val = val.lower()
        val = val.split('_')
        for word in val:
            token_list.append(word.translate(str.maketrans('', '', string.punctuation)))
    else:
        token_list.append(val.translate(str.maketrans('', '', string.punctuation)))
    return token_list


def get_ids(word, dict_out):
    tokens = get_tokens(word)
    return [dict_out[x] if x in dict_out.keys() else dict_out['unk'] for x in tokens]


def get_neg_samples(data_in, epoch, word_2_id, relation_dict, lower, upper):
    head_list, tail_list, relation_list = data_in['Head'].tolist(), \
                                          data_in['tail'].tolist(), \
                                          data_in['relation'].tolist()
    head_vec_1 = []
    tail_vec_1 = []
    relation_vec = []
    y_output = []
    for _ in range(epoch):
        for k in (range(len(head_list))):
            head_en = get_ids(head_list[k], word_2_id)
            tail_en = get_ids(tail_list[k], word_2_id)
            relation1 = relation_list[k]
            print(relation1)
            # print(head_list[k], relation1, tail_list[k])
            if relation1 == "gender":
                relation_en = get_ids(relation_list[k], word_2_id)
                head_ = [q for q in itertools.repeat(head_en, times=2)]
                relation = [q for q in itertools.repeat(relation_en, times=2)]
                negative_sample = random.sample(relation_dict[relation1], 1)
            else:
                relation_en = get_ids(relation_list[k], word_2_id)
                k1 = random.randint(lower, upper)
                head_ = [q for q in itertools.repeat(head_en, times=k1)]
                relation = [q for q in itertools.repeat(relation_en, times=k1)]
                # negative_sample = random.sample(tail_list, k1-1)
                key_list = list(relation_dict.keys())
                try:
                    key_list = list(relation_dict.keys())
                    key_list.remove(relation1)
                except ValueError:
                    print(relation1)
                rel_sample = random.sample(key_list, k1 - 1)
                negative_sample = []
                for val in rel_sample:
                    negative_sample += random.sample(relation_dict[val], 1)
                    # print(head_list[k], relation1, tail_list[k], val, negative_sample[-1])
            tail_in = []
            for t in negative_sample:
                tail_in.append(get_ids(t, word_2_id))
            tail_in.append(tail_en)
            head_vec_1 += head_
            tail_vec_1 += tail_in
            relation_vec += relation
            y_true = []
            for u in range(len(head_)):
                if tail_in[u] == tail_en:
                    y_true.append(1)
                else:
                    y_true.append(0)
            y_output += y_true
    print(len(head_list))
    # print(change)
    sampled_data = build_vector(head_vec_1, tail_vec_1, relation_vec, y_output, False)
    return sampled_data


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
        if head_list[index] not in rel_head[entity]:
            rel_head[entity].append(head_list[index])
        if tail_list[index] not in rel_tail[entity]:
            rel_tail[entity].append(tail_list[index])
    return rel_head, rel_tail


def build_vector(head_list, tail_list, relation_list, y_true, interchange=False):
    df = pd.DataFrame(columns=["Head", "relation", "tail", "score"])
    if interchange:
        df["Head"] = tail_list
        df["relation"] = relation_list
        df["tail"] = head_list
        df["score"] = y_true
    else:
        df["Head"] = head_list
        df["relation"] = relation_list
        df["tail"] = tail_list
        df["score"] = y_true
    return df


def get_tokens_train(data):
    columns = ["Head", "relation", "tail"]
    token_list = []
    for col in columns:
        for val in data[col]:
            val = val.lower()
            if "_" in val:
                val = val.lower()
                val = val.split('_')
                for word in val:
                    token_list.append(word.translate(str.maketrans('', '', string.punctuation)))
            else:
                token_list.append(val.translate(str.maketrans('', '', string.punctuation)))
    return token_list


def load_word2vec(file_path):
    word2vec = {}
    with open(file_path) as lines:
        for line in lines:
            split = line.split(" ")
            word = split[0]
            vector_strings = split[1:]
            vector = [float(num) for num in vector_strings]
            word2vec[word] = np.array(vector)
    return word2vec


def create_id2vec(id2word, word2vec):
    unk_vec = word2vec["unk"]
    dim_of_vector = len(unk_vec)
    num_of_tokens = max(list(id2word.keys()))
    id2vec = np.zeros((num_of_tokens + 1, dim_of_vector), dtype=np.float16)
    for id, word in id2word.items():
        if word != 'PAD':
            id2vec[id, :] = word2vec[word] if word in word2vec else unk_vec
    return id2vec


def CreatXid(token_dict, limit):
    word2id, id2word = {}, {}
    word2id['PAD'] = 0
    id2word[0] = 'PAD'
    word2id['<BOS>'] = 1
    word2id['<EOS>'] = 2
    id2word[1] = '<BOS>'
    id2word[2] = '<EOS>'
    word2id['unk'] = 3
    id2word[3] = 'unk'
    i = 4
    for key, value in token_dict.items():
        if value > limit:
            word2id[key] = i
            id2word[i] = key
            i += 1
    return word2id, id2word


def main():
    PATH = sys.argv[1]
    limit = int(sys.argv[2])
    glove_path = sys.argv[3]
    lower = int(sys.argv[4])
    upper = int(sys.argv[5])
    train_data = pd.read_csv(PATH + "train_kge.csv")
    head_list = train_data['Head'].tolist()
    tail_list = train_data['tail'].tolist()
    rel_list = train_data['relation'].tolist()
    dev_data = pd.read_csv(PATH + "dev_kge.csv")
    test_data = pd.read_csv(PATH + "test_kge.csv")
    token_list = get_tokens_train(train_data)
    print(len(token_list))
    print(token_list[:100])
    token_counter = Counter(token_list)
    print(len(token_counter))
    rel_head, rel_tail = get_rel_dict(head_list, tail_list, rel_list)
    print(rel_head.keys(), rel_tail.keys())
    word2id_, id2word_ = CreatXid(token_counter, limit)
    word2vec_ = load_word2vec(glove_path)
    id2vec = create_id2vec(id2word_, word2vec_)
    print(id2vec.shape)
    sample_data = get_neg_samples(train_data, 1, word2id_, rel_tail, lower, upper)
    dev_out = get_neg_samples(dev_data, 1, word2id_, rel_tail, 2, 2)
    test_out = get_neg_samples(test_data, 1, word2id_, rel_tail, 2, 2)
    sample_data.to_pickle('data/random_dataset/Sample_data_head_24.pkl')
    dev_out.to_pickle('data/random_dataset/dev_vec_dict.pkl')
    test_out.to_pickle('data/random_dataset/test_vec_dict.pkl')
    data_dict = {'id2vec': id2vec, 'word2id': word2id_, 'id2word': id2word_}
    joblib.dump(data_dict, 'kge_dict.pkl')

if __name__ == '__main__':
    main()
