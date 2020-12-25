# -*- coding: utf-8 -*-

import sys


def main():
    word2freq = {}
    feature2freq = {}
    label2freq = {}
    with open(sys.argv[1]) as f:
        for line in f:
            temp = line.strip().split("\t")
            labels, features, words = temp[3], temp[4], temp[2]
            for label in labels.split():
                if label not in label2freq:
                    label2freq[label] = 1
                else:
                    label2freq[label] += 1
            for word in words.split():
                if word not in word2freq:
                    word2freq[word] = 1
                else:
                    word2freq[word] += 1
                    # print(word2freq[word])
            for feature in features.split():
                if feature not in feature2freq:
                    feature2freq[feature] = 1
                else:
                    feature2freq[feature] += 1

    def _local(file_path, X2freq, start_idx=0, word2id=False):
        with open(file_path, "w") as f:
            if word2id:
                f.write(str(0) + "\t" + "PAD" + "\t" + str(1) + "\n")
                f.write(str(1) + "\t" + "<BOS>" + "\t" + str(1) + "\n")
                f.write(str(2) + "\t" + "<EOS>" + "\t" + str(1) + "\n")
                f.write(str(3) + "\t" + "UNK" + "\t" + str(1) + "\n")
            for i, (X, freq) in enumerate(sorted(X2freq.items(), key=lambda t: -t[1]), start_idx):
                f.write(str(i) + "\t" + X + "\t" + str(freq) + "\n")



    _local(sys.argv[2], word2freq, start_idx=4, word2id=True)
    _local(sys.argv[3], feature2freq, start_idx=1)
    _local(sys.argv[4], label2freq)


if (__name__ == '__main__'):
    main()

