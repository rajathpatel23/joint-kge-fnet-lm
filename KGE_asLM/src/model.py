import tensorflow as tf
import numpy as np
import pandas as pd
from collections import defaultdict
import logging


class NAM_Modified(object):

    def __init__(self, *args, **kwargs):
        super(NAM_Modified, self).__init__()
        self.lstm_layer = kwargs['lstm_layer']
        self.balance =  kwargs['balance']
        self.decay = kwargs['decay']
        self.starter_learning_rate = kwargs['learning_rate'] or 1e-4
        self.hidden_units_1 = kwargs['hidden_units_1']
        self.hidden_units_2 = kwargs['hidden_units_2']
        self.hidden_units_3 = kwargs['hidden_units_3']
        self.hidden_units_4 = kwargs['hidden_units_4']
        self.drop_out = kwargs['dropout']
        self.split = kwargs['splits']
        self.final = kwargs['final'] or False
        self.averaging = kwargs['averaging'] or False
        if self.final:
            self.Input_dimension = self.lstm_layer[-1]
        else:
            self.Input_dimension = self.lstm_layer[-1] * 2
        self.tail = tf.placeholder(dtype=tf.int64, name='y_true')
        self.head = tf.placeholder(dtype=tf.int64, name='data_in')
        self.rel = tf.placeholder(dtype=tf.int64, name='tail_in')
        self.y_true_1 = tf.placeholder(dtype=tf.float64, name='y_true_values')
        self.sequence_in = tf.placeholder(dtype=tf.float64, shape=(None, None, 300), name="sequence_in")
        self.embedding_mat = tf.get_variable(initializer=args[0], name="embedding_matrix", trainable=False)
        self.embed_head = tf.nn.embedding_lookup(self.embedding_mat, self.head)
        self.embed_tail = tf.nn.embedding_lookup(self.embedding_mat, self.tail)
        self.embed_relation = tf.nn.embedding_lookup(self.embedding_mat, self.rel)
        if self.averaging:
            self.embed_head = tf.reduce_mean(self.embed_head, axis=1)
            self.embed_tail = tf.reduce_mean(self.embed_tail, axis=1)
            self.embed_relation = tf.reduce_mean(self.embed_relation, axis=1)
            self.embed_total = tf.stack([self.embed_head, self.embed_relation, self.embed_tail], axis=1)
        else:
            self.embed_total = tf.concat([self.embed_head, self.embed_relation, self.embed_tail], axis=1)
        self.session_type = "Initialize"
        self.current_view = 1

        self.lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.lstm_layer]
        self.drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.drop_out) for lstm in self.lstms]
        self.cell = tf.contrib.rnn.MultiRNNCell(self.drops)
        self.lstm_output_1, self.final_state_1 = tf.nn.dynamic_rnn(self.cell, self.sequence_in, dtype=tf.float64)
        self.final_state_c, self.final_state_h = self.final_state_1[-1].c, self.final_state_1[-1].h
        print(self.final_state_c)
        self.weights = {
            'l1': tf.get_variable("Wl1", shape=[self.Input_dimension, self.hidden_units_1], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
            'l2': tf.get_variable("Wl2", shape=[self.hidden_units_1, self.hidden_units_2], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
            'l3': tf.get_variable("Wl3", shape=[self.hidden_units_2, self.hidden_units_3], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
            'l4': tf.get_variable("Wl4", shape=[self.hidden_units_3, self.hidden_units_4], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True)}
        self.biases = {
            'l1': tf.get_variable("bl1", shape=[self.hidden_units_1], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
            'l2': tf.get_variable("bl2", shape=[self.hidden_units_2], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
            'l3': tf.get_variable("bl3", shape=[self.hidden_units_3], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
            'l4': tf.get_variable("bl4", shape=[self.hidden_units_4], dtype=tf.float64,
                                  initializer=tf.contrib.layers.xavier_initializer(), trainable=True)}
        if self.averaging:
            self.head_, self.rel_, self.tail_ = tf.split(self.lstm_output_1, num_or_size_splits=3, axis=1)
        elif self.current_view == 1 and self.split == 6:
            _, self.head_, _, self.rel_, _, self.tail_ = tf.split(self.lstm_output_1, num_or_size_splits=6, axis=1)
        elif self.current_view == 1 and self.split == 9:
            _, _, self.head_, _, _, self.rel_, _, _, self.tail_ = tf.split(self.lstm_output_1, num_or_size_splits=9,
                                                                           axis=1)
        elif self.current_view == 1 and self.split == 12:
            _, _, _, self.head_, _, _, _, self.rel_, _, _, _, self.tail_ = tf.split(self.lstm_output_1,
                                                                                    num_or_size_splits=12, axis=1)
        elif self.current_view == 1 and self.split == 11:
            _, _, self.head_, _, _, _, _, self.rel_, _, _, self.tail_ = tf.split(self.lstm_output_1,
                                                                                 num_or_size_splits=11, axis=1)
        if self.final:
            print(self.final)
            self.z0 = self.final_state_c
            self.tail_ = tf.squeeze(self.tail_)
        else:
            self.tail_ = tf.squeeze(self.tail_)
            self.head_ = tf.squeeze(self.head_)
            self.rel_ = tf.squeeze(self.rel_)
            self.z0 = tf.concat([self.head_, self.rel_], axis=1)
            print(self.z0.shape)
        self.l1 = tf.matmul(self.z0, self.weights['l1'] + self.biases['l1'])
        self.z1 = tf.nn.relu(self.l1)
        self.z1 = tf.nn.dropout(self.z1, 0.1)
        self.l2 = tf.matmul(self.z1, self.weights['l2']) + self.biases['l2']
        self.z2 = tf.nn.relu(self.l2)
        self.l3 = tf.matmul(self.z2, self.weights['l3']) + self.biases['l3']
        self.z3 = tf.nn.relu(self.l3)
        self.z3 = tf.nn.dropout(self.z3, 0.2)
        if self.final:
            self.l4 = tf.matmul(self.z3, self.weights['l4']) + self.biases['l4']
            self.output = self.l4
        else:
            self.m = tf.multiply(self.z3, self.tail_)
            self.dot = tf.reduce_sum(self.m, axis=1)
            self.output = self.dot
        self.output = tf.squeeze(self.output)
        self.output_2 = tf.nn.sigmoid(self.output)
        self.beta = 0.01
        if self.final:
            self.regularizer = tf.nn.l2_loss(self.weights['l3']) \
                               + tf.nn.l2_loss(self.weights['l2']) + \
                               tf.nn.l2_loss(self.weights['l1']) + \
                               tf.nn.l2_loss(self.weights['l4'])
        else:
            self.regularizer = tf.nn.l2_loss(self.weights['l3']) \
                               + tf.nn.l2_loss(self.weights['l2']) + \
                               tf.nn.l2_loss(self.weights['l1'])
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
        return self.session.run([self.head_, self.rel_, self.tail_, self.z0], feed_dict=feed)

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
        self.session.close()
        logging.info("Model saved in path: %s" % save_path)

    def load(self, location, model_name, exp_num):
        self.saver.restore(self.session, location + model_name + '_' + exp_num + ".ckpt")
        logging.info("Model loaded from path: %s" % location)

    def get_train_var(self):
        return self.session.run([self.trainable_var])

    def get_total_var_num(self):
        return self.session.run([self.total_var_num])
        


