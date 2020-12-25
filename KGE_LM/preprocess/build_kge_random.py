import gzip
from sklearn.utils import shuffle
import sys
import pandas as pd
from sklearn.externals import  joblib


if __name__ == '__main__':
    PATH = joblib.load(sys.argv[1])
    whole_df = pd.DataFrame.from_dict(PATH)
    whole_df = shuffle(whole_df)
    a = int(0.8 * len(whole_df))
    s = int(0.8 * len(whole_df)) + int(0.1 *len(whole_df))
    train_df = whole_df[:a]
    dev_df = whole_df[a:s]
    test_df = whole_df[s:]
    train_df.to_csv("data/train_kge.csv")
    dev_df.to_csv("data/dev_kge.csv")
    test_df.to_csv("data/test_kge.csv")
