from src.utils import get_aucpr, tuning, tuning_1, evaluation, eval_data, get_ids
from src.model_bilstm import NAM_Modified
import tensorflow as tf
import logging
import argparse
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import os
import sys
import pandas as pd
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="saved model filename", default="model_kge_lstm", type=str)
    parser.add_argument("--data_location", help="data location", type=str)
    parser.add_argument("--exp_num", help="training exp num", default=1, type=int)
    parser.add_argument("--exp_info", default="training KGE model with lstm", help="experiment info", type=str)
    parser.add_argument("--epochs", help="Number of epochs to train", default=40, type=int)
    parser.add_argument('--lstm_size', help="LSTM dimension and layer", type=list, default=[256])
    parser.add_argument('--keep_prob', help="Dropout probability", default=1.0, type=float)
    parser.add_argument('--weight', help="loss weight", default=3.0, type=float)
    parser.add_argument('--train_file_name', help="insert train file name", type=str)
    parser.add_argument('--batch_size', help="train batch size", default=1024, type=int)
    parser.add_argument('--pad_head', help="padding head length", default=3, type=int)
    parser.add_argument('--pad_rel', help='padding tail length', default=3, type=int)
    parser.add_argument('--pad_tail', help="padding rel length", default=3, type=int)
    parser.add_argument('--data_set_type', help="KGE dataset type", default="WN11", type=str)
    parser.add_argument('--final_output', help="final output", default="False", type=str)
    parser.add_argument('--average_input', help="average input", default="False", type=str)
    parser.add_argument('--decay', help="loss decay", default="False", type=str)
    parser.add_argument('--learning_rate', help='train learning rate', default=1e-4, type=float)
    parser.add_argument('--splits', help="LSTM embedding split", default=9, type=int)
    args = parser.parse_args()
    if not os.path.exists(args.data_location + "run_logs/"):
        os.makedirs(args.data_location + "run_logs/")
    Log_location = args.data_location + "run_logs/"
    if not os.path.exists(args.data_location + "model_LSTM/"):
        os.makedirs(args.data_location + "model_LSTM/")
    sess_save_location = args.data_location + "model_LSTM/"
    experiment_number = args.exp_num
    log_file_name = args.model_name + "_train" + "_" + str(args.exp_num) + ".log"
    logging.basicConfig(filename=Log_location + log_file_name, filemode='w', level=logging.DEBUG)
    logging.info("This is the experiment number: %s", args.exp_num)
    logging.info("This experiment has changes: %s", args.exp_info)
    logging.info("LSTM dropout: {}".format(args.keep_prob))
    logging.info("number of training epochs: {}".format(args.epochs))
    logging.info("train file name: {}".format(args.train_file_name))
    logging.info("loss weight: {}".format(args.weight))

    # loading the training data
    load_data = pd.read_pickle(args.data_location + args.train_file_name)
    head_list = load_data['Head'].tolist()
    relation_list = load_data["relation"].tolist()
    tail_list = load_data["tail"].tolist()
    label = load_data["score"].tolist()
    dev_data = pd.read_pickle(args.data_location + "dev_vec_dict.pkl")
    test_data = pd.read_pickle(args.data_location + "test_vec_dict.pkl")
    logging.info("batch_size: {}".format(args.batch_size))
    logging.info("learning rate: {}".format(args.learning_rate))
    logging.info("padding head max len: {}".format(args.pad_head))
    logging.info("padding tail max len: {}".format(args.pad_tail))
    logging.info("padding relation max len: {}".format(args.pad_rel))
    logging.info("Using final output: {}".format(args.final_output))
    logging.info("Averaging: {}".format(args.average_input))
    relation_dict = pickle.load(open(args.data_location + "relation_2_tail.pkl", "rb"))
    word_2_id = pickle.load(open(args.data_location +"word_2_id.pkl", "rb"))
    embedding_matrix = pickle.load(open(args.data_location + "embedding_matrix.pkl", 'rb'))
    head_vec_dev = dev_data["Head"]
    tail_vec_dev = dev_data["tail"]
    relation_vec_dev = dev_data["relation"]
    y_output_dev = dev_data["score"]

    head_vec_test = test_data["Head"]
    tail_vec_test = test_data["tail"]
    relation_vec_test = test_data["relation"]
    y_output_test = test_data["score"]

    head_vec_dev = pad_sequences(head_vec_dev, maxlen=args.pad_head, dtype='int64', padding='pre',
                                 truncating='post',
                                 value=0)
    tail_vec_dev = pad_sequences(tail_vec_dev, maxlen=args.pad_tail, dtype='int64', padding='pre',
                                 truncating='post',
                                 value=0)
    relation_vec_dev = pad_sequences(relation_vec_dev, maxlen=args.pad_rel, dtype='int64', padding='pre',
                                     truncating='post',
                                     value=0)

    head_vec_train = pad_sequences(head_list, maxlen=args.pad_head, dtype='int64', padding='pre',
                                   truncating='post',
                                   value=0)
    tail_vec_train = pad_sequences(tail_list, maxlen=args.pad_tail, dtype='int64', padding='pre',
                                   truncating='post',
                                   value=0)
    relation_vec_train = pad_sequences(relation_list, maxlen=args.pad_rel, dtype='int64', padding='pre',
                                       truncating='post',
                                       value=0)
    head_vec_test = pad_sequences(head_vec_test, maxlen=args.pad_head, dtype='int64', padding='pre',
                                  truncating='post',
                                  value=0)
    tail_vec_test = pad_sequences(tail_vec_test, maxlen=args.pad_tail, dtype='int64', padding='pre',
                                  truncating='post',
                                  value=0)
    relation_vec_test = pad_sequences(relation_vec_test, maxlen=args.pad_rel, dtype='int64', padding='pre',
                                      truncating='post',
                                      value=0)

    Model = NAM_Modified(embedding_matrix, lstm_layer=args.lstm_size, balance=args.weight, hidden_units_1=1024,
                         hidden_units_2=512, hidden_units_3=256, hidden_units_4=1, dropout=args.keep_prob,
                         learning_rate=args.learning_rate,
                         splits=args.splits, final=eval(args.final_output), averaging=eval(args.average_input), decay=eval(args.decay))

    for epoch in range(args.epochs):
        logging.info("epochs= {}".format(epoch))
        M = len(head_list)
        batch_loss = []
        for k in range(0, M, args.batch_size):
            if k + args.batch_size > M:
                d = M
            else:
                d = k + args.batch_size
            head_vec_1 = head_vec_train[k:d]
            tail_vec_1 = tail_vec_train[k:d]
            relation_vec = relation_vec_train[k:d]
            y_output = label[k:d]
            head_o , tail_o, rel_o, embed_total = Model.get_embed(head_vec_1, tail_vec_1, relation_vec)
            h1, r1, t1, z01 = Model.debug(embed_total)
            cost, opt = Model.fit(embed_total, y_output)
            batch_loss.append(cost)
        epoch_loss = sum(batch_loss) / len(batch_loss)
        logging.info('Average epoch loss = {}'.format(epoch_loss))
        if epoch % 5 == 0:
            logging.info("Dev evaluation: =================================================================>")
            eval_data(Model, head_vec_dev, tail_vec_dev, relation_vec_dev, y_output_dev, data_type=False)

    logging.info("Final test evaluation =============================================================>")
    eval_data(Model, head_vec_test, tail_vec_test, relation_vec_test, y_output_test, data_type=False)
    Model.save(sess_save_location, args.model_name, str(args.exp_num))
