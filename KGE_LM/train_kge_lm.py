import tensorflow as tf
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
import argparse
from sklearn.externals import joblib
from src.model.joint_model import NAM_Modified
from src.batcher import Batcher
import numpy as np
import pandas as pd


def pad_sequence_(sequence, max_length, padding='post', truncating='post'):
    sequence = pad_sequences(sequence, maxlen=max_length, dtype='int64', padding=padding,
                             truncating=truncating,
                             value=0)
    return sequence


def get_aucpr(y_out, y_true):
    precision, recall, threshold = precision_recall_curve(y_out, y_true)
    aucpr = auc(recall, precision)
    return aucpr


def tuning(prediction):
    tune_prediction = []
    for i in range(len(prediction)):
        if prediction[i] >= 0.5:
            tune_prediction.append(1)
        else:
            tune_prediction.append(-1)
    return tune_prediction


def tuning_1(prediction):
    tune_prediction = []
    for i in range(len(prediction)):
        if prediction[i] >= 0.5:
            tune_prediction.append(1)
        else:
            tune_prediction.append(0)
    return tune_prediction


def evaluation(y_out, pred_out, data_type=False):
    if data_type:
        pred_1 = tuning_1(pred_out)
    else:
        pred_1 = tuning(pred_out)
    # for val in pred_out[0]:
    #     logging.info("val => {}".format(val))

    acc = 0
    assert len(pred_out) == len(y_out)
    for i in range(len(pred_1)):
        if pred_1[i] == y_out[i]:
            acc += 1
    b = (acc / len(pred_1)) * 100
    logging.info("The test accuracy: {}".format(b))
    roc_score_test = roc_auc_score(y_out, pred_out)
    aucpr_test = get_aucpr(y_out, pred_out)
    f1_test = f1_score(y_out, pred_1)
    precision_test = precision_score(y_out, pred_1)
    recall_test = recall_score(y_out, pred_1)
    logging.info("This is the AUCPR score: {}".format(aucpr_test))
    logging.info("This is the roc_score: {}".format(roc_score_test))
    logging.info("This is the F1 score: {}".format(f1_test))
    logging.info("This is precision: {}".format(precision_test))
    logging.info("This is recall: {}".format(recall_test))
    logging.info("Classification report: {}".format(classification_report(y_out, pred_1)))
    return b


def eval_data(model, head_vec, tail_vec, rel_vec, label_ids, data_type=True):
    N = len(head_vec)
    g1 = 1024
    output_dev_collector = []
    for k1 in (range(0, N, g1)):
        if k1 + g1 > N:
            d1 = N
        else:
            d1 = k1 + g1
        h, r, t, embed = model.get_embed(head_vec[k1:d1], tail_vec[k1:d1], rel_vec[k1:d1])
        # embed = np.squeeze(embed)
        output_ = model.predict(embed)
        # print(output_[0].shape)
        output_dev_collector += output_[0].tolist()
    acc = evaluation(label_ids, output_dev_collector, data_type)
    return acc


def eval_lm(data_set, set_type='Dev', dict_in=None):
    total_loss_, iters = 0, 0
    word_count = 0
    M_ = len(data_set.data)
    logging.info("Dev sentence number: {}".format(M_))
    for _ in range(0, M_, 1):
        sent_in = data_set.next()
        a_ = [len(x) for x in sent_in]
        word_count += (sum(a_)) - 2
        max_sent_ = max(a_)
        if max_sent_ > 2:
            loss_dev, score_dev = Model.predict_lang(sent_in)
            total_loss_ += loss_dev
            iters += 1
    avg_dev_loss = total_loss_ / iters
    logging.info(set_type + " validation loss LM : {}".format(avg_dev_loss))
    logging.info(set_type + " word count: {}".format(word_count))
    perplexity_dev = np.exp(total_loss_ / word_count)
    secondary_ppl = np.exp(total_loss_ / iters)
    logging.info(set_type + " validatoin results: ==== ")
    logging.info(set_type + " validation perplexity: {}".format(perplexity_dev))
    logging.info(set_type + " second ppl: {}".format(secondary_ppl))
    return perplexity_dev


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_kge', help='KGE dataset', type=str, default='FB13')
    parser.add_argument('--model_name', help='Model name', type=str,
                        dest='model_name', default='NAM_NER')
    parser.add_argument('--data_location_kge', help='data location of the KGE data', type=str)
    parser.add_argument('--data_location_lm', help='data location of the LM data', type=str)
    parser.add_argument('--exp_num', help='experiment number', type=str, default=99,
                        dest='exp_num')
    parser.add_argument('--exp_info', help='experiment information for logging', type=str,
                        default='NAM NER experiments', dest='exp_info')
    parser.add_argument('--epochs', help='Number of Epochs', type=int, default=30,
                        dest='epochs')
    parser.add_argument('--learning_rate_lm', help='task LM learning rate',
                        type=float, default=1e-3)
    parser.add_argument('--learning_rate_kge', help='task KGE learning rate',
                        type=float, default=1e-4)
    parser.add_argument('--lstm_layer', help='lstm layer size',
                        default='[256]', type=str)
    parser.add_argument('--option', help='evaluation of data', type=int,
                        default=1)
    parser.add_argument('--steps_per_epoch_kge', help='steps per epoch for KGE task',
                        default=1000, type=int)
    parser.add_argument('--steps_per_epoch_lm', help='steps per epoch for LM task',
                        default=1000, type=int)
    parser.add_argument('--joint_opt', help='joint optimization',
                        type=str, default=str(False))
    parser.add_argument('--balance', help='KGE training weights', type=float, default=3.0)
    parser.add_argument('--hidden_units_1', help='KGE hidden unit 1', type=int, default=1024)
    parser.add_argument('--hidden_units_2', help='KGE hidden unit 2', type=int, default=512)
    parser.add_argument('--hidden_units_3', help='KGE hidden unit 3', type=int, default=512)
    parser.add_argument('--hidden_units_4', help='KGE hidden unit 4', type=int, default=1)
    parser.add_argument('--keep_prob', help='LSTM dropout', type=float, default=0.6)
    parser.add_argument('--layer_type', help='type of LSTM', type=str, default='BiLSTM')
    parser.add_argument('--task', help="Task to run - INIT, KGE, LM", default='INIT', type=str)
    prev_acc = 0.0
    args = parser.parse_args()
    model_name = args.model_name
    DATA_LOCATION = args.data_location_kge
    if not os.path.exists(DATA_LOCATION + "run_log_kge_lm/"):
        os.makedirs(DATA_LOCATION + "run_log_kge_lm/")
    Log_location = DATA_LOCATION + "run_log_kge_lm/"
    if not os.path.exists(DATA_LOCATION + "model_kge_lm/"):
        os.makedirs(DATA_LOCATION + "model_kge_lm/")
    sess_save_location = DATA_LOCATION + "model_kge_lm/"
    experiment_number = args.exp_num
    log_file_name = model_name + "_train" + "_" + experiment_number + ".log"
    experiment_info = args.exp_info
    epochs = args.epochs
    keep_prob_ = args.keep_prob
    lstm_size = eval(args.lstm_layer)
    logging.basicConfig(filename=Log_location + log_file_name, filemode='w', level=logging.DEBUG)
    logging.info("This is the experiment number: %s", experiment_number)
    logging.info("This experiment has changes: %s", experiment_info)
    logging.info("LSTM dropout: {}".format(keep_prob_))
    logging.info("number of training epochs: {}".format(epochs))
    logging.info("steps per epoch LM: {}".format(args.steps_per_epoch_lm))
    logging.info("steps per epoch KGE: {}".format(args.steps_per_epoch_kge))
    logging.info("learning_rate KGE: {}".format(args.learning_rate_kge))
    logging.info("learning_rate LM: {}".format(args.learning_rate_lm))
    logging.info("Options: {}".format(args.option))
    logging.info("Loading the dictionaries")
    train_data_ = args.data_location_lm + 'train_selected.txt.gz'
    dev_data_ = args.data_location_lm + 'dev_selected.txt.gz'
    test_data_ = args.data_location_lm + 'test_selected.txt.gz'
    data_dict = joblib.load(args.data_location_kge + 'dict_wikifact_selected.pkl')
    print("data_dict loaded")
    train_batch = 5
    test_batch = 1
    dev_batch = 1
    train_batcher = Batcher(train_data_, data_dict['word2id'], 5)
    print("train object formed")
    dev_batcher = Batcher(dev_data_, data_dict['word2id'], dev_batch)
    print("dev object formed")
    test_batcher = Batcher(test_data_, data_dict['word2id'], train_batch)
    print("Data set loaded")

    load_data = joblib.load(DATA_LOCATION + "Sample_data_lm_24.pkl")
    head_list = load_data["Head"]
    relation_list = load_data["relation"]
    tail_list = load_data["tail"]
    label = np.array(load_data["score"])
    dev_data = joblib.load(DATA_LOCATION + "dev_vec_lm_dict.pkl")
    test_data = joblib.load(DATA_LOCATION + "test_vec_lm_dict.pkl")
    head_vec_dev = dev_data["Head"]
    tail_vec_dev = dev_data["tail"]
    relation_vec_dev = dev_data["relation"]
    y_output_dev = dev_data["score"]
    head_vec_test = test_data["Head"]
    tail_vec_test = test_data["tail"]
    relation_vec_test = test_data["relation"]
    y_output_test = test_data["score"]

    pad_head = max([len(x) for x in head_vec_dev])
    pad_tail = max([len(x) for x in tail_vec_dev])
    pad_rel = max([len(x) for x in relation_vec_dev])
    pad_h_test = max([len(x) for x in head_vec_test])
    pad_t_test = max([len(x) for x in tail_vec_test])
    pad_r_test = max([len(x) for x in relation_vec_test])

    head_vec_dev = pad_sequence_(head_vec_dev, pad_head, padding="post", truncating='post')
    tail_vec_dev = pad_sequence_(tail_vec_dev, pad_tail, padding="post", truncating='post')
    relation_vec_dev = pad_sequence_(relation_vec_dev, pad_rel, padding="post", truncating='post')
    head_vec_test = pad_sequence_(head_vec_test, pad_h_test, padding="post", truncating='post')
    tail_vec_test = pad_sequence_(tail_vec_test, pad_t_test, padding="post", truncating='post')
    relation_vec_test = pad_sequence_(relation_vec_test, pad_r_test, padding="post", truncating='post')
    head_vec_train = np.array(head_list)
    tail_vec_train = np.array(tail_list)
    relation_vec_train = np.array(relation_list)

    Model = NAM_Modified(embedding_matrix=data_dict['id2vec'], lstm_layer=lstm_size, balance=args.balance,
                         dropout=args.keep_prob, hidden_units_1=args.hidden_units_1,
                         learning_rate_kge=args.learning_rate_kge, task=args.task,
                         learning_rate_lm=args.learning_rate_lm,
                         hidden_units_2=args.hidden_units_2, hidden_units_3=args.hidden_units_3,
                         hidden_units_4=args.hidden_units_4, final=True, layer_type=args.layer_type)
    print("Model Loaded")
    prev_perlexity = 1e+10
    for epoch in range(epochs):
        batch_loss_2 = []
        logging.info("Epoch ==> {}".format(epoch))
        logging.info('epoch= {}'.format(epoch))
        batch_loss = []
        M = len(head_vec_train)
        for i in range(args.steps_per_epoch_kge):
            index_range = np.arange(0, M)
            val_ = np.random.choice(index_range, 256, replace=False)
            head_vec_1 = head_vec_train[val_]
            p_h_max = max([len(x) for x in head_vec_1])
            head_vec_1 = pad_sequence_(head_vec_1, p_h_max)
            tail_vec_1 = tail_vec_train[val_]
            p_t_max = max([len(x) for x in tail_vec_1])
            tail_vec_1 = pad_sequence_(tail_vec_1, p_t_max)
            relation_vec = relation_vec_train[val_]
            p_r_max = max([len(x) for x in relation_vec])
            relation_vec = pad_sequence_(relation_vec, p_r_max)
            y_output = label[val_]
            head_o, rel_o, tail_o, embed_total = Model.get_embed(head_vec_1, tail_vec_1, relation_vec)
            cost, opt = Model.fit(embed_total, y_output)
            batch_loss.append(cost)
        epoch_loss = sum(batch_loss) / len(batch_loss)
        logging.info('Average epoch loss KGE = {}'.format(epoch_loss))
        print('Average epoch loss = {}'.format(epoch_loss))
        logging.info("Dev evaluation: ======>")
        curr_acc = eval_data(Model, head_vec_dev, tail_vec_dev, relation_vec_dev, y_output_dev)
        if curr_acc > prev_acc:
            Model.save(sess_save_location, model_name + "_best_KGE_", experiment_number)
            prev_acc = curr_acc
        Model.save(sess_save_location, model_name, experiment_number)
        if epoch % args.option == 0 and args.task == "INIT":
            for i in range(args.steps_per_epoch_lm):
                sent_ = train_batcher.next()
                max_sent = max([len(x) for x in sent_])
                # print(max_sent)
                sent_pad = pad_sequence_(sent_, max_length=max_sent)
                loss, _ = Model.train_lang(sent_pad)
                batch_loss_2.append(loss)
            epoch_loss_lm = sum(batch_loss_2) / len(batch_loss_2)
            print("Average epoch loss: ", epoch_loss_lm)
            logging.info('Average epoch loss LM = {}'.format(epoch_loss_lm))
            if epoch % 1 == 0:
                logging.info("Dev LM evaluation ================")
                curr_perplexity = eval_lm(dev_batcher, set_type="Dev", dict_in=data_dict['id2word'])
                if curr_perplexity < prev_perlexity:
                    Model.save(sess_save_location, model_name + "_best_LM", experiment_number)
                    prev_perlexity = curr_perplexity
            Model.save(sess_save_location, model_name, experiment_number)

    logging.info("Training Complete")
    if args.task == 'INIT':
        eval_lm(test_batcher, set_type="Test", dict_in=data_dict['id2word'])
    logging.info("KGE Test evaluation")
    eval_data(Model, head_vec_test, tail_vec_test, relation_vec_test, y_output_test)
