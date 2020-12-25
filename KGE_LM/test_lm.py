import argparse
from sklearn.externals import joblib
from src.model.lm_model_1 import Language_Model
from src.batcher import Batcher
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import pandas as pd
from collections import defaultdict
import itertools
import random
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import logging
import os
import sys
import logging


def pad_sequence_(sequence, max_length):
    sequence = pad_sequences(sequence, maxlen=max_length, dtype='int64', padding='post',
                             truncating='post',
                             value=0)
    return sequence


def eval_lm(data_set, set_type='Dev', dict_in=None):
    total_loss_, iters = 0, 0
    word_count = 0
    M_ = len(data_set.data)
    logging.info("Dev sentence number: {}".format(M_))
    for _ in range(0, M_, 1):
        sent_in = data_set.next()
        a_ = [len(x) for x in sent_in]
        word_count += ((sum(a_)) -2)
        max_sent_ = max(a_)
        if max_sent_ > 2:
            sent_pad_ = pad_sequence_(sent_in, max_length=max_sent_)
            loss_dev, score_dev = Model.predict_lang(sent_pad_)
            total_loss_ += loss_dev
            iters += 1
    avg_dev_loss = total_loss_ / iters
    logging.info(set_type + " validation loss LM : {}".format(avg_dev_loss))
    logging.info(set_type + " word count: {}".format(word_count))
    perplexity_dev = np.exp(total_loss_ / word_count)
    logging.info(set_type + " validatoin results: ==== ")
    logging.info(set_type + " validation perplexity: {}".format(perplexity_dev))
    return perplexity_dev


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', help='data location of the LM data', type=str)
    parser.add_argument('--data_location_lm', help='data location of the LM data', type=str)
    parser.add_argument('--model_name', help='Model name', type=str,
                        dest='model_name', default='NAM_NER')
    parser.add_argument('--exp_num', help='experiment number', type=str, default=99)
    parser.add_argument('--exp_info', help='experiment information for logging', type=str,
                        default='NAM NER experiments')
    parser.add_argument('--learning_rate_lm', help='task LM learning rate',
                        type=float, default=1e-3)
    parser.add_argument('--option', help='Training of the FNER data', type=int,
                        default=5)
    parser.add_argument('--lstm_layer', help='lstm layer size',
                        default='[256]', type=str)
    parser.add_argument('--layer_type', help='type of LSTM layer', type=str, default='BiLSTM')
    parser.add_argument('--steps_per_epoch_lm', help='steps per epoch for LM task',
                        default=100, type=int)
    parser.add_argument('--epochs', help='Number of Epochs', type=int, default=30)
    parser.add_argument('--keep_probs', help='LSTM dropout', type=float, default=0.6)
    args = parser.parse_args()
    model_name = args.model_name
    DATA_LOCATION = args.data_location
    if not os.path.exists(DATA_LOCATION + "run_logs_LM/"):
        os.makedirs(DATA_LOCATION + "run_logs_LM/")
    Log_location = DATA_LOCATION + "run_logs_LM/"
    if not os.path.exists(DATA_LOCATION + "model_LM/"):
        os.makedirs(DATA_LOCATION + "model_LM/")
    sess_save_location = DATA_LOCATION + "model_LM/"
    experiment_number = args.exp_num
    experiment_info = args.exp_info
    epochs = args.epochs
    keep_prob_ = args.keep_probs
    lstm_size = eval(args.lstm_layer)
    log_file_name = model_name + "_test" + "_" + experiment_number + ".log"
    logging.basicConfig(filename=Log_location + log_file_name, filemode='w', level=logging.DEBUG)
    logging.info("This is the experiment number: %s", experiment_number)
    logging.info("This experiment has changes: %s", experiment_info)
    logging.info("LSTM dropout: {}".format(keep_prob_))
    logging.info("number of training epochs: {}".format(epochs))
    logging.info("steps per epoch LM: {}".format(args.steps_per_epoch_lm))
    logging.info("learning_rate LM: {}".format(args.learning_rate_lm))
    logging.info("Loading the dictionaries")
    data_dicts = joblib.load(args.data_location_lm + "dict_wikifact_selected.pkl")
    test_data_ = args.data_location + 'test_selected.txt.gz'
    print("data_dict loaded")
    test_batcher = Batcher(test_data_, data_dicts['word2id'], 1)
    print("Data set loaded")
    Model = Language_Model(embedding_matrix=data_dicts['id2vec'], lstm_layer=lstm_size,
                           dropout=keep_prob_, learning_rate=args.learning_rate_lm, layer_type=args.layer_type)
    logging.info("Using the final model: ")
    Model.load(sess_save_location, model_name, experiment_number)
    eval_lm(test_batcher, set_type="Test", dict_in=data_dicts['id2word'])
    logging.info("Using the best model: ")
    Model.load(sess_save_location, model_name + "_best_LM", experiment_number)
    eval_lm(test_batcher, set_type="Test", dict_in=data_dicts['id2word'])
    print("Model Loaded !!!!!")
    id2word = data_dicts['id2word']
    word2id = data_dicts['word2id']
    input = "<BOS> issa serge coelo born 1967"
    input_sentence = input.split()
    input_sent = [[word2id[x] if x in word2id.keys() else word2id['unk'] for x in input_sentence]]
    print(input_sent)
    i = 0
    while i < 20:
        next_output = Model.generate_lang(input_sent)
        a = np.argmax(next_output, axis=-1)
        a = a[0].tolist()
        word_list = [id2word[x] for x in a]
        word = id2word[a[0]]
        input_sentence.append(word)
        print("Predicted sentence: ", input_sentence)
        input_sent[0] += a
        print("This is next input: ", input_sent)
        i+=1
    sentence = " ".join(input_sentence)
    print(sentence)
