import numpy as np
import sys
import os
from random import shuffle
from sklearn.externals import joblib
import gzip

def build(filename, list_file):
    with gzip.open(filename, "wt") as file_zip:
        for line in list_file:
            file_zip.write(line)
        file_zip.close()


if __name__ == '__main__':
    main_file = sys.argv[1]
    with gzip.open(main_file, "rt") as zipfile:
        file = zipfile.readlines()
        shuffle(file)
        s = int(0.8 * len(file))
        e = int(0.8 * len(file)) + int(0.1 * len(file))
        train_lines = file[:s]
        build("train_selected.txt.gz", train_lines)
        dev_lines = file[s:e]
        build("dev_selected.txt.gz", dev_lines)
        test_lines = file[e:]
        build("test_selected.txt.gz", test_lines)

