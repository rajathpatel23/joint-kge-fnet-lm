# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:26:52 2019

@author: rpsworker
"""
# %%
import json
import pickle
import itertools
import random
from tqdm import tqdm
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import sys


def get_ids(entity, word_2_id, json_in, column_name="relation"):
    temp_list = []
    # if dataset == "WN11":
    #     head = head.strip("__")
    if column_name == 'relation':
        x = entity.split("/")[1:]
        ent = x
    else:
        print("This is entity: ", entity)
        x = entity.split()

        try:
            ent= json_in[entity.strip()]['label']
            x = ent.split()
            print("We are getting id: == > ", entity, ent, x)
        except KeyError:
            ent = 'unk'
            x = ent.split()
            print("We are getting id: == > ", entity, ent, x)

    x = [i for i in x if i != '']
    for k in x:
        if k in word_2_id.keys():
            temp_list.append(word_2_id[k])
        else:
            temp_list.append(word_2_id['unk'])
    print("We are got id: == > ", temp_list, entity, ent, x)
    return temp_list

def get_ids_1(entity, word_2_id):
    temp_list = []

    print("This is entity: ", entity)
    x = entity.split()
    x = [i for i in x if i != '']
    for k in x:
        if k in word_2_id.keys():
            temp_list.append(word_2_id[k])
        else:
            temp_list.append(word_2_id['unk'])
    print("We are got id: == > ", temp_list, entity, x)
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

    # change = sys.argv[2]
    file_name = sys.argv[2]
    lower = int(sys.argv[3])
    upper = int(sys.argv[4])
    epoch = int(sys.argv[5])
    relation_file = sys.argv[6]
    input_file = sys.argv[7]
    type_ = sys.argv[8]
    json_file = sys.argv[9]
    with open(DATA_LOCATION + json_file) as file:
        json_data = json.load(file)
    load_data = pd.read_csv(data_location + input_file, sep='\t',names=["Head", "relation", "tail"])
    head_list = load_data["Head"].tolist()
    relation_list = load_data["relation"].tolist()
    tail_list = load_data["tail"].tolist()
    # if change == "False":
    relation_dict = pickle.load(open(DATA_LOCATION + relation_file, "rb"))
    # else:
    #     relation_dict = pickle.load(open(DATA_LOCATION + relation_file, "rb"))
    word_2_id = pickle.load(open(DATA_LOCATION + "word_2_id.pkl", "rb"))
    g = 512
    head_vec_1 = []
    tail_vec_1 = []
    relation_vec = []
    y_output = []
    for _ in range(epoch):
        for k in (range(len(head_list))):
            head_en = get_ids(head_list[k], word_2_id, json_data, column_name="Head", )
            print("Head")
            tail_en = get_ids(tail_list[k], word_2_id, json_data, column_name="tail")
            print("Tail")
            relation1 = relation_list[k]
            # print(head_list[k], relation1, tail_list[k])
            if relation1 == "gender":
                relation_en = get_ids(relation_list[j], word_2_id, json_data, column_name='relation')
                head_ = [q for q in itertools.repeat(head_en, times=2)]
                relation = [q for q in itertools.repeat(relation_en, times=2)]
                negative_sample = random.sample(relation_dict[relation1], 1)
            else:
                relation_en = get_ids(relation_list[k], word_2_id, json_data, column_name='relation')
                k1 = random.randint(lower, upper)
                head_ = [q for q in itertools.repeat(head_en, times=k1)]
                relation = [q for q in itertools.repeat(relation_en, times=k1)]
                # negative_sample = random.sample(tail_list, k1-1)
                key_list = list(relation_dict.keys())
                key_list.remove(relation1)
                random.shuffle(key_list)
                rel_sample = random.sample(key_list, len(key_list))
                negative_sample = []
                val = 0
                a = 0
                while a < k1-1 and val < len(key_list):
                    list_rel = relation_dict[key_list[val]]
                    if 'unk' in list_rel:
                        list_rel.remove('unk')
                    if list_rel:
                        negative_sample += random.sample(list_rel, 1)
                        print(head_list[k], relation1, tail_list[k], val, negative_sample[-1])
                        a+=1
                    if a == k1-1:
                        break
                    val+=1


            tail_in = []
            for t in negative_sample:
                tail_in.append(get_ids_1(t, word_2_id))
            tail_in.append(tail_en)
            head_vec_1 += head_
            tail_vec_1 += tail_in
            relation_vec += relation
            y_true = []
            for u in range(len(head_)):
                if tail_in[u] == tail_en:
                    y_true.append(1)
                else:
                    if type_ == "train":
                        y_true.append(0)
                    else:
                        y_true.append(-1)
            y_output += y_true
    print(len(head_list))
    # print(change)
    sampled_data = build_vector(head_vec_1, tail_vec_1, relation_vec, y_output, False)
    sampled_data.to_pickle(STORE_LOCATION + file_name, protocol=2)


main()
