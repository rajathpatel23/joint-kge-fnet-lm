import numpy as np
from sklearn.externals import joblib
import random


class Batcher:
    def __init__(self, storage, data, batch_size, context_length, id2vec, doc2vec, sentence):
        self.context_length = context_length
        self.storage = storage
        self.data = data
        self.num_of_samples = int(data.shape[0])
        self.dim = 300  # len(id2vec[0])
        self.num_of_labels = data.shape[1] - 4 - 70
        self.batch_size = batch_size
        self.batch_num = 0
        self.max_batch_num = int(self.num_of_samples / self.batch_size)
        self.id2vec = id2vec
        self.pad = np.zeros(self.dim)
        self.pad[0] = 1.0
        self.doc_2_vec = doc2vec
        # self.sentence_in =np.asanyarray(sentence)

    def create_input_output(self, row, doc_2_vec):
        s_start = row[0]
        s_end = row[1]
        e_start = row[2]
        e_end = row[3]
        # print(e_end)
        labels = row[74:]
        # print(labels)
        features = row[4:74]
        # print(features)
        seq_context = np.zeros((self.context_length * 2 + 1, self.dim))
        context_ = [0 for _ in range(self.context_length * 2 + 1)]
        mention = self.storage[e_start:e_end]
        mention = mention.tolist()
        # print("mention: ", self.storage[e_start:e_end])
        temp = [self.id2vec[self.storage[i]][:self.dim] for i in range(e_start, e_end)]
        mean_target = np.mean(temp, axis=0)
        temp_1 = []
        j = max(0, self.context_length - (e_start - s_start))
        for i in range(max(s_start, e_start - self.context_length), e_start):
            temp_1.append(self.storage[i])
            context_[j] = self.storage[i]
            # print(self.storage[i])
            seq_context[j, :] = self.id2vec[self.storage[i]][:self.dim]
            j += 1
        seq_context[j, :] = np.ones_like(self.pad)
        context_[j] = 0
        j += 1
        temp_2 = []
        for i in range(e_end, min(e_end + self.context_length, s_end)):
            temp_2.append(self.storage[i])
            context_[j] = self.storage[i]
            seq_context[j, :] = self.id2vec[self.storage[i]][:self.dim]
            j += 1
        # print("this is left context", temp_1, "this is right context", temp_2)
        # print(temp_1, type(temp_1))
        # print(temp_2, type(temp_2))
        sentence = temp_1 + mention + temp_2
        left_ = context_[:self.context_length]
        right_ = context_[self.context_length + 1:]
        # print(sentence)
        return seq_context, mean_target, labels, features, doc_2_vec, sentence, mention, left_, right_

    def next(self):
        X_context = np.zeros((self.batch_size, self.context_length * 2 + 1, self.dim))
        X_target_mean = np.zeros((self.batch_size, self.dim))
        Y = np.zeros((self.batch_size, self.num_of_labels))
        F = np.zeros((self.batch_size, 70), np.int32)
        doc_vec = np.zeros((self.batch_size, 50), np.float64)
        sent_in = [0 for _ in range(self.batch_size)]
        mention_id = [0 for _ in range(self.batch_size)]
        left_id = [0 for _ in range(self.batch_size)]
        right_id = [0 for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            X_context[i, :, :], X_target_mean[i, :], Y[i, :], F[i, :], doc_vec[i:], sent_in[i], mention_id[i], left_id[i], right_id[i] = self.create_input_output(
                self.data[self.batch_num * self.batch_size + i, :],
                self.doc_2_vec[self.batch_num * self.batch_size + i, :])
        self.batch_num = (self.batch_num + 1) % self.max_batch_num
        return [X_context, X_target_mean, Y, F, doc_vec, sent_in, mention_id, left_id, right_id]

    # def shuffle(self):
    def shuffle(self):
        assert len(self.data) == len(self.doc_2_vec)
        p = np.random.permutation(len(self.data))
        self.data = self.data[p]
        self.doc_2_vec = self.doc_2_vec[p]
        # self.sentence_in = self.sentence_in[p]
#
# if __name__ == '__main__':
#     train = joblib.load('/p/data/NAM_NER_rajat/NAM_NER_Multi_task/preprocess_new/resource_1/OntoNotes/train_dataset_11072019.pkl')
#     dict_ = joblib.load('/p/data/NAM_NER_rajat/NAM_NER_Multi_task/preprocess_new/resource_1/OntoNotes/dict_gillick.pkl')
#     batcher = Batcher(train['storage'], train['data'], 3, 10, dict_['id2vec'], train['Doc2Vec'], train['sentence_in'])
#     batcher.shuffle()
#     x, y, z, a, b, s = batcher.next()
#     # print(x[0])
#     # print(x.shape)
#     # print(s)
#     # print(len(s))