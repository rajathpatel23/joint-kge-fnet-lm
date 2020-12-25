import argparse
from sklearn.externals import joblib
from src.model.nn_model_fner import Model
from src.batcher import Batcher
from src.hook import acc_hook, save_predictions
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import pandas as pd
from collections import defaultdict
import pickle
import itertools
import random
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import logging
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, \
    precision_recall_curve, auc
import sys


def pad_sequence_(sequence, max_length):
    sequence = pad_sequences(sequence, maxlen=max_length, dtype='int64', padding='post',
                             truncating='post',
                             value=0)
    return sequence


def pad_single(sent_in):
    max_sent = max([len(x) for x in sent_in])
    sent_in_pad = pad_sequence_(sent_in, max_sent)
    return sent_in_pad


def valid_eval(data_in, task, eval_type=None, final=False):
    # m1 = data_in['mention']
    #     # l1 = data_in['left_context']
    #     # r1 = data_in['right_context']
    #     # la1 = data_in['label']
    #     # m1, l1, r1 = pad_method(m1, l1, r1)

    if task == 'FNER':
        collector = []
        true = []
        iters = 0
        total_loss = []
        c_, m_, lab, f, d, s_in, m_id, l_id, r_id = data_in.next()
        lf_id = pad_single(l_id)
        rt_id = pad_single(r_id)
        rt_id = np.flip(rt_id, axis=-1)
        p1 = 100
        for k in range(0, len(c_), p1):
            s = Model.predict(lf_id[k:k + p1], rt_id[k:k + p1],
                              context_data=None,
                              mention_representation_data=m_[k:k + p1],
                              feature_data=f[k:k+p1],
                              doc_vector=None)
            loss_val = Model.error(lf_id[k:k + p1], rt_id[k:k + p1], lab[k:k + p1],
                                   context_data=None,
                                   mention_representation_data=m_[k:k + p1],
                                   feature_data=f[k:k+p1],
                                   doc_vector=None)

            r = lab[k:k + p1]
            collector.append(s)
            true.append(r)
            total_loss.append(loss_val)
            iters += 1
        average_eval_loss = sum(total_loss) / iters
        print(task + " Loss: ", average_eval_loss)
        collector = np.array(collector)
        collector = np.vstack(collector)
        collector = np.squeeze(collector)
        true = np.array(true)
        true = np.vstack(true)
        print(collector.shape, true.shape)
        print(collector)
        print(true)
        strict_f1 = acc_hook(collector, true)
        logging.info(str(eval_type) + " FNER loss: {}".format(average_eval_loss))
        if final:
            fname = args.dataset + "_" + args.encoder + "_" + str(args.feature) + "_" + str(args.hier) + ".txt"
            save_predictions(collector, true, dicts["id2label"], fname)
        return strict_f1


if __name__ == "__main__":
    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset to train model",
                        choices=["figer", "gillick"])
    parser.add_argument("encoder", help="context encoder to use in model",
                        choices=['hier-attention'])
    parser.add_argument('--feature', dest='feature', action='store_true')
    parser.add_argument('--no-feature', dest='feature', action='store_false')
    parser.set_defaults(feature=False)
    parser.add_argument('--hier', dest='hier', action='store_true')
    parser.add_argument('--no-hier', dest='hier', action='store_false')
    parser.set_defaults(hier=False)
    parser.add_argument('--model_name', help='Model name', type=str,
                        dest='model_name', default='NAM_NER')
    parser.add_argument('--data_location', help='data location of the KGE data', type=str,
                        dest='data_location')
    parser.add_argument('--exp_num', help='experiment number', type=str, default=99,
                        dest='exp_num')
    parser.add_argument('--exp_info', help='experiment information for logging', type=str,
                        default='NAM NER experiments', dest='exp_info')
    # parser.add_argument('--epochs', help='Number of Epochs', type=int, default=30,
    #                     dest='epochs')
    parser.add_argument('--keep_probs', help='LSTM dropout', type=float, default=0.6,
                        dest='keep_probs')
    parser.add_argument('--learning_rate_fner', help='task kge learning rate',
                        type=float, default=1e-4, dest="learning_rate_fner")
    parser.add_argument('--lstm_layer', help='lstm layer size',
                        default='[256]', type=str, dest="lstm_layer")
    # parser.add_argument('--steps_per_epoch_fner', help='steps per epoch for FNER task',
    #                     default=2000, dest='steps_per_epoch_fner', type=int)
    parser.add_argument('--doc_2_vec', help='enable doc_2_vec', default="True",
                        dest='doc_2_vec', type=str)
    parser.add_argument('--option', help='Training of the FNER data', type=int,
                        default=5, dest='option')
    args = parser.parse_args()
    current_view = 1
    model_name = args.model_name
    DATA_LOCATION = args.data_location
    if not os.path.exists(DATA_LOCATION + "run_logs_bilstm_fner"):
        os.makedirs(DATA_LOCATION + "run_logs_bilstm_fner/")
    Log_location = DATA_LOCATION + "run_logs_bilstm_fner/"
    if not os.path.exists(DATA_LOCATION + "model_BiLSTM_fner/"):
        os.makedirs(DATA_LOCATION + "model_BiLSTM_fner/")
    sess_save_location = DATA_LOCATION + "model_BiLSTM_fner/"
    experiment_number = args.exp_num
    sess_graph_name = args.model_name + "_train_" + args.exp_num
    if not os.path.exists(DATA_LOCATION + "model_BiLSTM_fner/" + sess_graph_name + "/"):
        os.makedirs(DATA_LOCATION + "model_BiLSTM_fner/" + sess_graph_name + "/")
    session_graph = DATA_LOCATION + "model_BiLSTM_fner/" + sess_graph_name + "/"
    experiment_number = args.exp_num
    # experiment_number = sys.argv[3]
    log_file_name = model_name + "_test" + "_" + experiment_number + ".log"
    experiment_info = args.exp_info
    epochs = args.epochs
    keep_prob_ = args.keep_probs
    lr = args.learning_rate_fner
    lstm_size = eval(args.lstm_layer)
    logging.basicConfig(filename=Log_location + log_file_name, filemode='w', level=logging.DEBUG)
    logging.info("This is the experiment number: %s", experiment_number)
    logging.info("This experiment has changes: %s", experiment_info)
    logging.info("LSTM dropout: {}".format(keep_prob_))
    logging.info("number of training epochs: {}".format(epochs))
    logging.info("learning rate: {}".format(lr))
    logging.info("steps per epoch: {}".format(args.steps_per_epoch_fner))
    logging.info("fner dataset: {}".format(args.dataset))
    logging.info("option: {}".format(args.option))
    logging.info("LSTM size: {}".format(args.lstm_layer))
    logging.info("Doc2vec: {}".format(args.doc_2_vec))
    logging.info("Feature: {}".format(args.feature))

    logging.info("Creating the model")

    logging.info("Loading the dictionaries")
    d = "Wiki" if args.dataset == "figer" else "OntoNotes"
    dicts = joblib.load(DATA_LOCATION + "dict_" + args.dataset + ".pkl")

    logging.info("Loading the datasets")
    train_dataset = joblib.load(DATA_LOCATION + "train_dataset" + "_11072019.pkl")
    dev_dataset = joblib.load(DATA_LOCATION + "dev_dataset" + "_11072019.pkl")
    test_dataset = joblib.load(DATA_LOCATION + "test_dataset" + "_11072019.pkl")
    print("fetching the FNER dataset")
    logging.info("train_size:{}".format(train_dataset["data"].shape[0]))
    logging.info("dev_size: {}".format(dev_dataset["data"].shape[0]))
    logging.info("test_size: {}".format(test_dataset["data"].shape[0]))

    # batch_size : 1000, context_length : 10
    train_batcher = Batcher(train_dataset["storage"], train_dataset["data"], 1000, 10, dicts["id2vec"],
                            train_dataset['Doc2Vec'], train_dataset['sentence_in'])
    dev_batcher = Batcher(dev_dataset["storage"], dev_dataset["data"], dev_dataset["data"].shape[0], 10,
                          dicts["id2vec"], dev_dataset['Doc2Vec'], dev_dataset['sentence_in'])
    test_batcher = Batcher(test_dataset["storage"], test_dataset["data"], test_dataset["data"].shape[0], 10,
                           dicts["id2vec"], test_dataset['Doc2Vec'], test_dataset['sentence_in'])
    # step_par_epoch = args.steps_per_epoch

    Model = Model(type=args.dataset, encoder=args.encoder, hier=args.hier, feature=args.feature,
                  dropout=keep_prob_, decay=True, session_graph=session_graph,
                  learning_rate_fner=args.learning_rate_fner,
                  embedding_matrix=dicts['id2vec'], lstm_layer=lstm_size, doc_vec=eval(args.doc_2_vec))

    previous_f1 = 0.0
    logging.info("test evaluation FNER: ======>")
    Model.load(sess_save_location, model_name, experiment_number)
    type_embed = Model.get_type_embed()
    joblib.dump(type_embed, DATA_LOCATION + "run_logs_bilstm_fner/" + model_name + ".pkl")
    print("Model Loaded")
    valid_eval(test_batcher, task='FNER', eval_type="Test", final=True)
