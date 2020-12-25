from sklearn.externals import joblib
from collections import defaultdict, Counter
import gzip
import sys
import string
import numpy as np


def CreateXid(dict_in, limit):
    word2id, id2word = {}, {}
    word2id["PAD"] = 0
    id2word[0] = "PAD"
    word2id["<BOS>"] = 1
    id2word[1] = "<BOS>"
    word2id["<EOS>"] = 2
    id2word[2] = "<EOS>"
    word2id["unk"] = 3
    id2word[3] = "unk"
    i = 4
    for key, value in dict_in.items():
        if value > limit:
            word2id[key] = i
            id2word[i] = key
            i += 1
    return word2id, id2word


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


def collect_tokens(filename):
    token_list = []
    with gzip.open(filename, 'rt') as file:
        for line in file:
            a = line.split()
            for word in a:
                if "_" in word:
                    token_list += word.split("_")
                else:
                    token_list.append(word.translate(str.maketrans('', '', string.punctuation)))
    return token_list


if __name__ == '__main__':
    PATH = sys.argv[1]
    glove_path = sys.argv[2]
    save_location_path = sys.argv[3]
    limit = eval(sys.argv[4])
    token_list = []
    token_list += collect_tokens(PATH)
    token_count = Counter(token_list)
    word2id_, id2word_ = CreateXid(token_count, limit=limit)
    word2vec_ = load_word2vec(glove_path)
    id2vec = create_id2vec(id2word_, word2vec_)
    print(id2vec.shape)
    dict_data = {"id2vec": id2vec, "word2id": word2id_, "id2word": id2word_}
    joblib.dump(dict_data, save_location_path)
