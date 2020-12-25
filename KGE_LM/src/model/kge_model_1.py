import tensorflow as tf
import numpy as np
import pandas as pd
from collections import defaultdict
import logging


class NAM_Modified(object):

    def __init__(self, **kwargs):
        super(NAM_Modified, self).__init__()
        self.embedding_matrix = kwargs['embedding_matrix']
        self.lstm_layer = kwargs['lstm_layer']
        self.balance = kwargs['balance']
        self.drop_out = kwargs['dropout']
        self.layer_type = kwargs['layer_type']
        self.Input_dimension = self.lstm_layer[-1] * 2
        self.starter_learning_rate = kwargs['learning_rate_kge']
        self.hidden_units_1 = kwargs['hidden_units_1']
        self.hidden_units_2 = kwargs['hidden_units_2']
        self.hidden_units_3 = kwargs['hidden_units_3']
        self.hidden_units_4 = kwargs['hidden_units_4']
        self.final = kwargs['final']
        self.average = False
        self.decay = True
        print(self.final)
        if self.layer_type == 'BiLSTM':
            self.Input_dimension = self.lstm_layer[-1] * 2
        if self.layer_type == 'LSTM':
            self.Input_dimension = self.lstm_layer[-1]
        self.tail = tf.placeholder(dtype=tf.int64, name='y_true')
        self.head = tf.placeholder(dtype=tf.int64, name='data_in')
        self.rel = tf.placeholder(dtype=tf.int64, name='tail_in')
        self.y_true_1 = tf.placeholder(dtype=tf.float64, name="y_true_values")
        self.sequence_in = tf.placeholder(dtype=tf.float64, shape=(None, None, 300), name="sequence_in")
        self.embedding_mat = tf.get_variable(initializer=self.embedding_matrix, name="embedding_matrix", trainable=False)
        self.embed_head = tf.nn.embedding_lookup(self.embedding_mat, self.head)
        if self.average:
            self.embed_head = tf.reduce_mean(self.embed_head, axis=1)
        self.embed_tail = tf.nn.embedding_lookup(self.embedding_mat, self.tail)
        if self.average:
            self.embed_tail = tf.reduce_mean(self.embed_tail, axis=1)
        self.embed_relation = tf.nn.embedding_lookup(self.embedding_mat, self.rel)
        if self.average:
            self.embed_relation = tf.reduce_mean(self.embed_relation, axis=1)
        if self.average:
            self.embed_total = tf.stack([self.embed_head, self.embed_relation, self.embed_tail], axis=1)
        else:
            self.embed_total = tf.concat([self.embed_head, self.embed_relation, self.embed_tail], axis=1)
        self.session_type = "Initialize"
        self.current_view = 1

        # with tf.variable_scope('LSTM'):
        self.lstm_fw = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.lstm_layer]
        self.lstm_bw = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.lstm_layer]
        self.drops_fw = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.drop_out) for lstm in self.lstm_fw]
        self.drops_bw = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.drop_out) for lstm in self.lstm_bw]
        self.cell_fw = tf.contrib.rnn.MultiRNNCell(self.drops_fw)
        self.cell_bw = tf.contrib.rnn.MultiRNNCell(self.drops_bw)
        if self.layer_type == 'BiLSTM':
            self.lstm_output_1, self.final_state_1 = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw,
                                                                                     self.sequence_in, dtype=tf.float64)
            self.lstm_output_1 = tf.concat(self.lstm_output_1, -1)
            self.final_state_fw, self.final_state_bw = self.final_state_1
            self.final_state_c_fw, self.final_state_c_bw = self.final_state_fw[-1].c, self.final_state_bw[-1].c
            self.final_state_c = tf.concat([self.final_state_c_fw, self.final_state_c_bw], axis=1)
            logging.info("state 1 & 2 : {}{}".format(self.final_state_c_fw, self.final_state_c_bw))
        if self.layer_type == 'LSTM':
            self.lstm_output_1, self.final_state_1 = tf.nn.dynamic_rnn(self.cell_fw, self.sequence_in, dtype=tf.float64)
            self.final_state_c, self.final_state_h = self.final_state_1[-1].c, self.final_state_1[-1].h
        self.weights = {
            'l1': tf.get_variable("Wl1", shape=[self.Input_dimension, self.hidden_units_1], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
            'l2': tf.get_variable("Wl2", shape=[self.hidden_units_1, self.hidden_units_2], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
            'l3': tf.get_variable("Wl3", shape=[self.hidden_units_2, self.hidden_units_3], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
            'l4': tf.get_variable("Wl4", shape=[self.hidden_units_3, self.hidden_units_4], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
        }
        self.biases = {
            'l1': tf.get_variable("bl1", shape=[self.hidden_units_1], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
            'l2': tf.get_variable("bl2", shape=[self.hidden_units_2], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
            'l3': tf.get_variable("bl3", shape=[self.hidden_units_3], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
            'l4': tf.get_variable("bl4", shape=[self.hidden_units_4], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True),

        }

        self.z0 = self.final_state_c
        self.l1 = tf.matmul(self.z0, self.weights['l1']) + self.biases['l1']
        self.z1 = tf.nn.relu(self.l1)
        # self.z1 = tf.nn.dropout(self.z1, 0.1)
        self.l2 = tf.matmul(self.z1, self.weights['l2']) + self.biases['l2']
        self.z2 = tf.nn.relu(self.l2)
        self.l3 = tf.matmul(self.z2, self.weights['l3']) + self.biases['l3']
        self.z3 = tf.nn.relu(self.l3)
        self.z3 = tf.nn.dropout(self.z3, 0.2)
        self.l4 = tf.matmul(self.z3, self.weights['l4']) + self.biases['l4']
        self.output = self.l4
        self.output = tf.squeeze(self.output)
        self.output_2 = tf.nn.sigmoid(self.output)
        self.beta = 0.01
        self.regularizer = tf.nn.l2_loss(self.weights['l3']) \
                           + tf.nn.l2_loss(self.weights['l2']) + \
                           tf.nn.l2_loss(self.weights['l1']) + \
                           tf.nn.l2_loss(self.weights['l4'])
        self.regularizer = self.beta * self.regularizer
        self.cost = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(self.y_true_1, self.output, pos_weight=self.balance) +
            self.regularizer)
        if self.decay:
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                            decay_steps=100000,
                                                            decay_rate=0.5, staircase=True)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.cost, global_step=self.global_step)
        else:
            self.optimizer = tf.train.AdamOptimizer(self.starter_learning_rate)
            self.train_op = self.optimizer.minimize(self.cost, global_step=None)
        self.saver = tf.train.Saver()
        self.config = tf.ConfigProto(device_count={'GPU': 1})
        self.config.gpu_options.allow_growth = True
        self.init = tf.global_variables_initializer()
        self.session = tf.Session(config=self.config)
        self.trainable_var = tf.trainable_variables()
        self.total_var_num = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
        self.session.run(self.init)

    def get_embed(self, head_i, tail_i, rel_i):
        feed = {
            self.head: head_i, self.tail: tail_i, self.rel: rel_i
        }
        return self.session.run([self.embed_head, self.embed_tail, self.embed_relation, self.embed_total],
                                feed_dict=feed)

    def debug(self, sequence):
        feed = {
            self.sequence_in: sequence
        }
        return self.session.run([self.lstm_output_1, self.final_state_1], feed_dict=feed)

    def fit(self, sequence, y_out):
        self.session_type = "train"
        feed = {self.y_true_1: y_out,
                self.sequence_in: sequence
                }
        return self.session.run([self.cost, self.train_op], feed_dict=feed)

    def predict(self, sequence):
        self.session_type = "test"
        feed = {
            self.sequence_in: sequence
        }
        return self.session.run([self.output_2], feed_dict=feed)

    def save(self, location, model_name, experiment_number):
        save_path = self.saver.save(self.session, location + model_name + "_" + experiment_number + ".ckpt")

        logging.info("Model saved in path: %s" % save_path)

    def load(self, location, model_name, exp_num):
        self.saver.restore(self.session, location + model_name + '_' + exp_num + ".ckpt")
        logging.info("Model loaded from path: %s" % location)

    def get_train_var(self):
        return self.session.run([self.trainable_var])

    def get_total_var_num(self):
        return self.session.run([self.total_var_num])

    def close(self):
        self.session.close()
