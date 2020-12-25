import numpy as np
import sys
import os
from random import shuffle
from sklearn.externals import joblib


if __name__ == '__main__':
    directory = sys.argv[1]

    file_list = os.listdir(directory)
    extract_list = []
    for file in file_list:
        if file.endswith(".sm"):
            extract_list.append(file)
    shuffle(extract_list)
    train_limit = int(0.8*len(extract_list))
    train_files = extract_list[:train_limit]
    dev_limit = train_limit + int(0.1*len(extract_list))
    dev_files = extract_list[train_limit:dev_limit]
    test_files = extract_list[dev_limit:]
    joblib.dump(train_files, "train_files_freebase.pkl")
    joblib.dump(dev_files, "dev_files_freebase.pkl")
    joblib.dump(test_files, "test_files_freebase.pkl")