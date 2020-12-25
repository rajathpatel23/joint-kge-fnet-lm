import argparse
from sklearn.externals import joblib
from src.model.nn_model_auto_kge import NAM_Modified
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
        pred_1 = tuning_1(pred_out[0])
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
    # print(pred_out, y_out)
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


def eval_data(model, head_vec, tail_vec, rel_vec, label_ids, data_type=False):
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
        output_ = model.predict_kge(embed)
        # print(output_[0].shape)
        output_dev_collector += output_[0].tolist()
    evaluation(label_ids, output_dev_collector, data_type)


def pad_single(sent_in):
    max_sent = max([len(x) for x in sent_in])
    sent_in_pad = pad_sequence_(sent_in, max_sent)
    return sent_in_pad


def valid_eval(data_in, task='FNER', eval_type=None, final=False):
    m1 = data_in['mention']
    l1 = data_in['left_context']
    r1 = data_in['right_context']
    lab = data_in['label']
    lf_id = pad_single(l1)
    rt_id = pad_single(r1)
    m_ = pad_single(m1)
    # m_, lf_id, rt_id = pad_method(m1, l1, r1)
    collector = []
    true = []
    eval_loss = []
    iters = 0
    p1 = 100
    total_loss = []
    iters = 0
    for k in range(0, len(m_), p1):
        s = Model.predict(lf_id[k:k + p1], rt_id[k:k + p1],
                          context_data=None,
                          mention_representation_data=m_[k:k + p1],
                          feature_data=None,
                          doc_vector=None)
        loss_val = Model.error(lf_id[k:k + p1], rt_id[k:k + p1], lab[k:k + p1],
                               context_data=None,
                               mention_representation_data=m_[k:k + p1],
                               feature_data=None,
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
    strict_f1 = acc_hook(collector, true)
    logging.info(str(eval_type) + " FNER loss: {}".format(average_eval_loss))
    if final:
        fname = args.dataset + "_" + args.encoder + "_" + str(args.feature) + "_" + str(args.hier) + "_" + str(
            args.dataset_kge) + ".txt"
        save_predictions(collector, true, dicts["id2label"], fname)
    return strict_f1


if __name__ == "__main__":
    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset to train model",
                        choices=["figer", "gillick", "wikiauto"])
    parser.add_argument("encoder", help="context encoder to use in model",
                        choices=["averaging", "lstm",
                                 "attentive",
                                 "sentence-attentive",
                                 'attentive_figet',
                                 'hier-attention'])
    parser.add_argument('--feature', dest='feature', action='store_true')
    parser.add_argument('--no-feature', dest='feature', action='store_false')
    parser.set_defaults(feature=False)
    parser.add_argument('--hier', dest='hier', action='store_true')
    parser.add_argument('--no-hier', dest='hier', action='store_false')
    parser.set_defaults(hier=False)
    parser.add_argument('--model_name', help='Model name', type=str,
                        dest='model_name', default='NAM_NER')
    parser.add_argument('--dataset_kge', help='KGE dataset name', type=str, default='FB13')
    parser.add_argument('--data_location_kge', help='data location of the KGE data', type=str)
    parser.add_argument('--data_location_fner', help='data location of the FNER data', type=str)
    parser.add_argument('--exp_num', help='experiment number', type=str, default=99,
                        dest='exp_num')
    parser.add_argument('--exp_info', help='experiment information for logging', type=str,
                        default='NAM NER experiments', dest='exp_info')
    parser.add_argument('--epochs', help='Number of Epochs', type=int, default=30,
                        dest='epochs')
    parser.add_argument('--keep_probs', help='LSTM dropout', type=float, default=0.6,
                        dest='keep_probs')
    parser.add_argument('--learning_rate_fner', help='task fner learning rate',
                        type=float, default=1e-3)
    parser.add_argument('--learning_rate_kge', help='task kge learning rate',
                        type=float, default=1e-3)
    parser.add_argument('--lstm_layer', help='lstm layer size',
                        default='[256]', type=str)
    parser.add_argument('--steps_per_epoch_fner', help='steps per epoch for FNER task',
                        default=100, dest='steps_per_epoch_fner', type=int)
    parser.add_argument('--steps_per_epoch_kge', help='steps per epoch for KGE task',
                        default=100, type=int)
    parser.add_argument('--doc_2_vec', help='enable doc_2_vec', default="False", type=str)
    parser.add_argument('--option', help='Training of the FNER data', type=int,
                        default=1)
    parser.add_argument('--balance', help='KGE training weights', type=float, default=3.0)
    parser.add_argument('--hidden_units_1', help='KGE hidden unit 1', type=int, default=1024)
    parser.add_argument('--hidden_units_2', help='KGE hidden unit 2', type=int, default=512)
    parser.add_argument('--hidden_units_3', help='KGE hidden unit 3', type=int, default=512)
    parser.add_argument('--hidden_units_4', help='KGE hidden unit 4', type=int, default=1)
    parser.add_argument('--keep_prob', help='LSTM dropout', type=float, default=0.6)
    parser.add_argument('--split', help='split for KGE', type=int, default=9)
    parser.add_argument('--pad_head', help='Head max len', type=int, default=3)
    parser.add_argument('--pad_tail', help='Tail max len', type=int, default=3)
    parser.add_argument('--pad_rel', help='Relation max len', type=int, default=3)
    parser.add_argument('--task', help="Task to run - INIT, KGE, FNER", default='INIT', type=str)
    parser.add_argument('--joint_opt', help="joint optimization", default='False', type=str)

    args = parser.parse_args()
    model_name = args.model_name
    DATA_LOCATION = args.data_location_kge
    DATA_LOCATION_2 = args.data_location_fner
    if not os.path.exists(DATA_LOCATION + "run_log_fner_kge_wiki/"):
        os.makedirs(DATA_LOCATION + "run_log_fner_kge_wiki/")
    Log_location = DATA_LOCATION + "run_log_fner_kge_wiki/"
    if not os.path.exists(DATA_LOCATION + "model_fner_kge_wiki/"):
        os.makedirs(DATA_LOCATION + "model_fner_kge_wiki/")
    sess_save_location = DATA_LOCATION + 'model_fner_kge_wiki/'
    experiment_number = args.exp_num
    experiment_info = args.exp_info
    epochs = args.epochs
    sess_graph_name = args.model_name + "_train_" + args.exp_num
    if not os.path.exists(DATA_LOCATION + "model_fner_kge/" + sess_graph_name + "/"):
        os.makedirs(DATA_LOCATION + "model_fner_kge/" + sess_graph_name + "/")
    session_graph = DATA_LOCATION + "model_fner_kge/" + sess_graph_name + "/"
    log_file_name = model_name + "_train" + "_" + experiment_number + ".log"
    pad_head = args.pad_head
    pad_tail = args.pad_tail
    pad_rel = args.pad_rel
    lstm_size = eval(args.lstm_layer)
    keep_prob_ = args.keep_prob
    logging.basicConfig(filename=Log_location + log_file_name, filemode='w', level=logging.DEBUG)
    logging.info("This is the experiment number: %s", experiment_number)
    logging.info("This experiment has changes: %s", experiment_info)
    logging.info("LSTM dropout: {}".format(keep_prob_))
    logging.info("number of training epochs: {}".format(epochs))
    logging.info("steps per epoch FNER: {}".format(args.steps_per_epoch_fner))
    logging.info("steps per epoch KGE: {}".format(args.steps_per_epoch_kge))
    logging.info("KGE dataset: {}".format(args.dataset_kge))
    logging.info("fner dataset: {}".format(args.dataset))
    logging.info("learning_rate FNER: {}".format(args.learning_rate_fner))
    logging.info("learning_rate KGE: {}".format(args.learning_rate_kge))
    logging.info("Options: {}".format(args.option))
    logging.info("Joint Optimization: {}".format(args.joint_opt))
    logging.info("Loading the dictionaries")

    dicts = joblib.load(DATA_LOCATION_2 + "dict_" + args.dataset + ".pkl")

    logging.info("Loading the datasets")

    print("fetching the FNER dataset")
    train_data = joblib.load(DATA_LOCATION_2 + 'train_dataset_id.pkl')
    dev_batcher = joblib.load(DATA_LOCATION_2 + 'valid_dataset_id.pkl')
    test_batcher = joblib.load(DATA_LOCATION_2 + 'test_dataset_id.pkl')
    men_train = np.array(train_data['mention'])
    left_train = np.array(train_data['left_context'])
    right_train = np.array(train_data['right_context'])
    label_train = train_data['label']
    embedding_matrix_1 = joblib.load(DATA_LOCATION_2 + 'embedding_matrix.pkl')
    print("loading KGE dataset")
    embedding_matrix = pickle.load(open(DATA_LOCATION + "embedding_matrix.pkl", "rb"))
    load_data = joblib.load(DATA_LOCATION + "Sample_data_head_24.pkl")
    head_list = load_data["Head"].tolist()
    relation_list = load_data["relation"].tolist()
    tail_list = load_data["tail"].tolist()
    label = np.array(load_data["score"])
    dev_data = joblib.load(DATA_LOCATION + "dev_vec_dict.pkl")
    test_data = joblib.load(DATA_LOCATION + "test_vec_dict.pkl")
    head_vec_dev = dev_data["Head"]
    tail_vec_dev = dev_data["tail"]
    relation_vec_dev = dev_data["relation"]
    y_output_dev = dev_data["score"]
    head_vec_test = test_data["Head"]
    tail_vec_test = test_data["tail"]
    relation_vec_test = test_data["relation"]
    y_output_test = test_data["score"]

    Model = NAM_Modified(embedding_matrix_fner=embedding_matrix_1, lstm_layer=lstm_size, balance=args.balance,
                         type=args.dataset, encoder=args.encoder, feature=args.feature,
                         dropout=args.keep_prob, hidden_units_1=args.hidden_units_1,
                         embedding_matrix_kge=embedding_matrix, learning_rate_joint=1e-3,
                         session_graph=session_graph, split=args.split, joint_opt=eval(args.joint_opt),
                         learning_rate_kge=args.learning_rate_kge, task=args.task,
                         learning_rate_fner=args.learning_rate_fner, doc_vec=eval(args.doc_2_vec),
                         hidden_units_2=args.hidden_units_2, hidden_units_3=args.hidden_units_3,
                         hidden_units_4=args.hidden_units_4, final=True)
    print("Model loaded")
    head_vec_dev = pad_sequence_(head_vec_dev, pad_head, padding="pre", truncating='post')
    tail_vec_dev = pad_sequence_(tail_vec_dev, pad_tail, padding="pre", truncating='post')
    relation_vec_dev = pad_sequence_(relation_vec_dev, pad_rel, padding="pre", truncating='post')
    head_vec_test = pad_sequence_(head_vec_test, pad_head, padding="pre", truncating='post')
    tail_vec_test = pad_sequence_(tail_vec_test, pad_tail, padding="pre", truncating='post')
    relation_vec_test = pad_sequence_(relation_vec_test, pad_rel, padding="pre", truncating='post')
    head_vec_train = pad_sequence_(head_list, pad_head, padding="pre", truncating='post')
    tail_vec_train = pad_sequence_(tail_list, pad_tail, padding="pre", truncating='post')
    relation_vec_train = pad_sequence_(relation_list, pad_rel, padding="pre", truncating='post')
    previous_f1 = 0.0
    for epoch in range(args.epochs):
        logging.info('epoch= {}'.format(epoch))
        batch_loss = []
        if epoch % args.option == 0:
            batch_loss = []
            logging.info("Creating batchers")
            # train_batcher.shuffle()
            M  = len(men_train)
            for i in range(args.steps_per_epoch_fner):
                index_range = np.arange(0, M)
                val_ = np.random.choice(index_range, 1024, replace=False)
                men = men_train[val_]
                left_ = left_train[val_]
                # print(left_)
                right_ = right_train[val_]
                # print(right_)
                target_data = label_train[val_]
                mention_id = pad_single(men)
                left_id = pad_single(left_)
                right_id = pad_single(right_)
                right_id = np.flip(right_id, axis=-1)
                loss_1, optim_, summary, step = Model.train_FNER(left_id, right_id, target_data, context_data=None,
                                                                 mention_representation_data=mention_id,
                                                                 feature_data=None, doc_vector=None)
                Model.summary_writer.add_summary(summary, step)
                batch_loss.append(loss_1)
            epoch_loss_fner = sum(batch_loss) / len(batch_loss)
            logging.info('Average epoch loss FNER = {}'.format(epoch_loss_fner))
            print('Average epoch loss FNER = {}'.format(epoch_loss_fner))
        if epoch % args.option == 0:
            # logging.info("Train evaluation: ======>")
            # train_f1 = valid_eval(train_batcher, task='FNER', eval_type="Train")
            logging.info("Dev evaluation: ======>")
            strict_f1 = valid_eval(dev_batcher, task='FNER', eval_type="Dev")
            strict_f1 = round(strict_f1, 4)
            if strict_f1 > previous_f1:
                logging.info("New best f1: {}".format(strict_f1))
                Model.save(sess_save_location, model_name + "_FNER_best", experiment_number)
                previous_f1 = strict_f1
                logging.info("test evaluation FNER: ======>")
            s_f1 = valid_eval(test_batcher, task='FNER', eval_type="Test")
        # for epoch in range(epochs):
        batch_loss_2 = []
        logging.info("Epoch ==> {}".format(epoch))
        logging.info('epoch= {}'.format(epoch))
        batch_loss = []
        M = len(head_vec_train)
        for i in range(args.steps_per_epoch_kge):
            index_range = np.arange(0, M)
            val_ = np.random.choice(index_range, 1024, replace=False)
            head_vec_1 = head_vec_train[val_]
            tail_vec_1 = tail_vec_train[val_]
            relation_vec = relation_vec_train[val_]
            y_output = label[val_]
            head_o, rel_o, tail_o, embed_total = Model.get_embed(head_vec_1, tail_vec_1, relation_vec)
            # embed_total = np.squeeze(embed_total)
            cost, opt = Model.fit(embed_total, y_output)
            batch_loss.append(cost)
        epoch_loss = sum(batch_loss) / len(batch_loss)
        logging.info('Average epoch loss KGE = {}'.format(epoch_loss))
        print('Average epoch loss = {}'.format(epoch_loss))
        logging.info("Dev evaluation: ======>")
        eval_data(Model, head_vec_dev, tail_vec_dev, relation_vec_dev, y_output_dev)
        Model.save(sess_save_location, model_name, experiment_number)

    logging.info("Training completed.  Below are the final test scores: ")
    logging.info("-----test--------")
    logging.info("KGE Test evaluation")
    Model.load(sess_save_location, model_name, experiment_number)
    eval_data(Model, head_vec_test, tail_vec_test, relation_vec_test, y_output_test)
    logging.info("test evaluation FNER: ======>")
    Model.load(sess_save_location, model_name + "_FNER_best", experiment_number)
    valid_eval(test_batcher, task='FNER', eval_type="Test", final=True)
