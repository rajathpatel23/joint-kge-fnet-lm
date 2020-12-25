import numpy as np
import gzip
import sys
import os
from sklearn.externals import joblib

def get_text(line):
    temp_list = []
    for word in line:
        if word.startswith("@"):
            a = word.split('/')
            temp_list.append(a[0].strip('@'))
        else:
            temp_list.append(word)
    print(temp_list)
    return temp_list


if __name__ == '__main__':
    directory = sys.argv[1]
    train_files = joblib.load(sys.argv[2])
    extension = sys.argv[3]
    output_file = sys.argv[4]
    assert extension in [".sm", ".bd"]
    with gzip.open(output_file, "wt", encoding='utf-8') as zipfile:
        for file in train_files:
            if extension != ".sm":
                file = file.replace(".sm", extension)
            with open(directory + file, "r") as file:
                for l in file:
                    l = l.split()
                    text_line = get_text(l)
                    text_line = " ".join(text_line)
                    print(text_line)
                    zipfile.write(str(text_line) + "\n")
        zipfile.close()
