# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:26:52 2019

@author: rpsworker
"""
# %%

import pickle
import itertools
import random
from tqdm import tqdm
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import sys


def get_ids(head, word_2_id, dataset="WN11"):
    temp_list = []
    if dataset == "WN11":
        head = head.strip("__")
    en = head.split("_")
    for k in en:
        if k in word_2_id.keys():
            temp_list.append(word_2_id[k])
        else:
            temp_list.append(word_2_id['unk'])
    return temp_list


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


def main():
    DATA_LOCATION = sys.argv[1]
    STORE_LOCATION = DATA_LOCATION
    data_location = DATA_LOCATION
    change = False
    load_data = pd.read_csv(data_location + "data_list_train.csv")
    data_type = sys.argv[2]
    if data_type == 'train':
        load_data = pd.read_csv(data_location + "data_list_train.csv")
    if data_type == 'dev':
        load_data = pd.read_csv(data_location + "data_list_dev.csv")
    if data_type == 'test':
        load_data = pd.read_csv(data_location + "data_list_test.csv")
    file_name = sys.argv[3]
    lower = int(sys.argv[4])
    upper = int(sys.argv[5])
    epoch = int(sys.argv[6])
    head_list = load_data["Head"].tolist()
    relation_list = load_data["relation"].tolist()
    tail_list = load_data["tail"].tolist()
    if change == "False":
        relation_dict = pickle.load(open(DATA_LOCATION + "relation_2_tail.pkl", "rb"))
    else:
        relation_dict = pickle.load(open(DATA_LOCATION + "relation_2_head.pkl", "rb"))
    word_2_id = pickle.load(open(DATA_LOCATION + "word_2_id.pkl", "rb"))
    g = 512
    head_vec_1 = []
    tail_vec_1 = []
    relation_vec = []
    y_output = []
    for _ in range(epoch):
        for k in (range(len(head_list))):
            head_en = get_ids(head_list[k], word_2_id)
            tail_en = get_ids(tail_list[k], word_2_id)
            relation1 = relation_list[k]
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
                key_list.remove(relation1)
                rel_sample = random.sample(key_list, k1 - 1)
                negative_sample = []
                for val in rel_sample:
                    negative_sample += random.sample(relation_dict[val], 1)
                    print(head_list[k], relation1, tail_list[k], val, negative_sample[-1])
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
    print(change)
    sampled_data = build_vector(head_vec_1, tail_vec_1, relation_vec, y_output, False)
    sampled_data.to_pickle(STORE_LOCATION + file_name, protocol=2)


main()
