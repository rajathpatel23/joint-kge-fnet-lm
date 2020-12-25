from sklearn.externals import joblib
import gzip
import numpy as np
import string


class Batcher(object):
    def __init__(self, data, dict_data, batch_size, limit=False):
        with gzip.open(data, "rt") as file:
            self.data = file.readlines()
        self.batch_size = batch_size
        self.num_of_samples = len(self.data)
        self.dict_data = dict_data
        self.batch_num = 0
        self.max_batch_num = int(self.num_of_samples / self.batch_size)
        self.limit = limit

    def tokenize(self, line):
        token_list = []
        line = line.split()
        for word in line:
            # print(word)
            word = word.lower()
            if "_" in word:
                token_list += word.split("_")
            else:
                w_ = word.translate(str.maketrans('', '', string.punctuation))
                if w_.strip():
                    token_list.append(w_)
        # print(token_list)
        if len(token_list) > 800 and self.limit:
            token_list = token_list[:800]
        return token_list

    def create_output(self, line):
        token = self.tokenize(line)
        token = ["<BOS>"] + token + ["<EOS>"]
        return [self.dict_data[x] if x in self.dict_data else self.dict_data["unk"] for x in token]

    def next(self):
        sent_in = [0 for _ in range(self.batch_size)]
        for j in range(self.batch_size):
            sent_in[j] = self.create_output(self.data[self.batch_num * self.batch_size + j])
        self.batch_num = (self.batch_num + 1) % self.max_batch_num
        return sent_in

    def shuffle(self):
        shuffle(self.data)


# if __name__ == '__main__':
#     dict_data = joblib.load('../data/random_dataset/dict_wikifact_selected.pkl')
#     test_file = Batcher("../data/random_dataset/dev_selected.txt.gz", dict_data['word2id'], 1)
#     for i in range(len(test_file.data)):
#         print(test_file.next())
#         if i == 100:
#             break
#         i += 1
