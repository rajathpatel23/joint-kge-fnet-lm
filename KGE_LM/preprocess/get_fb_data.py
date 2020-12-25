import numpy as np
import gzip
import sys
import os
from collections import defaultdict
from sklearn.externals import joblib


if __name__ == '__main__':
    directory = sys.argv[1]
    train_file = joblib.load(sys.argv[2])
    triple = defaultdict(list)
    extension = sys.argv[3]
    output_file = sys.argv[4]
    assert extension in [".en"]
    for file in train_file:
        if extension != ".sm":
            file = file.replace(".sm", extension)
            with open(directory + file, "r") as fb_file:
                for line in fb_file:
                    a = line.split()
                    if len(a) == 3:
                        print(a)
                        triple["Head"].append(a[0])
                        triple["relation"].append(a[1])
                        triple["tail"].append(a[2])
    joblib.dump(triple, output_file)