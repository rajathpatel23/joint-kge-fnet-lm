'''
'''

import tensorflow as tf
from sklearn.externals import joblib
import numpy as np
import logging


class Language_Model(object):
    def __init__(self, *args, **kwargs):
        self.embedding_matrix = kwargs['embedding_matrix'].astype(np.float32)
        self.lstm_layer = kwargs['lstm_layer']
        self.dropout = kwargs['dropout']
        self.learning_rate = kwargs['learning_rate']
        self.lstm_dim = self.lstm_layer[-1]
        self.max_norm = 5.0
        self.vocabulary_size = self.embedding_matrix.shape[0]
        self.a = self.lstm_dim
        with tf.device('/cpu:0'):
            self.embedding_mat = tf.get_variable(dtype=tf.float32, initializer=self.embedding_matrix,
                                                 name='embedding_matrix', trainable=False)
        # self.embedding_mat = tf.get_variable(dtype=tf.float32, shape=(self.vocabulary_size, 100),
        #                                         name='embedding_matrix', trainable=False)

        with tf.device('/gpu:0'):
            self.sentence_in = tf.placeholder(dtype=tf.int32, shape=(None, None), name='sentence_con')
            self.sent_input = self.sentence_in[:, :-1]
            self.sentence_out = self.sentence_in[:, 1:]
            self.sentence_embed = tf.nn.embedding_lookup(self.embedding_mat, self.sent_input)
            self.lstm_fw = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.lstm_layer]
            self.seq_len = tf.reduce_sum(tf.sign(self.sent_input), 1)
            self.drops_fw = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=1.0, output_keep_prob=self.dropout) for
                             lstm in self.lstm_fw]
            self.cell_fw = tf.contrib.rnn.MultiRNNCell(self.drops_fw)
            self.rnn_outputs, self.final_sentence = self.LSTM_Layer(input_=self.sentence_embed,
                                                                    sequence_len=self.seq_len,
                                                                    time_major=False)
            self.final_output = self.rnn_outputs[:, -1, :]
        with tf.device('/gpu:1'):
            self.weights = tf.get_variable(dtype=tf.float32, shape=(self.a, self.vocabulary_size),
                                           name="softmax_weights", trainable=True)
            self.bias = tf.get_variable(dtype=tf.float32, shape=self.vocabulary_size, name='softmax_bias', trainable=True)
            self.logits = tf.matmul(self.rnn_outputs, self.weights) + self.bias
            self.next_output = tf.nn.softmax(tf.matmul(self.final_output, self.weights) + self.bias)

            self.pred_logits = tf.nn.softmax(self.logits)
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                         targets=self.sentence_out,
                                                         weights=tf.sequence_mask(
                                                             self.seq_len,
                                                             tf.shape(self.sentence_in)[
                                                                 1] -1,
                                                             dtype=tf.float32),
                                                         average_across_timesteps=False,
                                                         average_across_batch=False)
            self.output_loss = tf.reduce_mean(tf.reduce_sum(self.loss, axis=1), axis=0)
            self.prediction_loss = tf.reduce_sum(tf.reduce_sum(self.loss, axis=1), axis=0)
        with tf.device('/gpu:0'):
            self.global_step_3 = tf.Variable(0, trainable=False)
            self.params_lm = tf.trainable_variables()
            self.gradients_lm = tf.gradients(self.output_loss, self.params_lm)
            self.clipped_gradients_lm, _ = tf.clip_by_global_norm(self.gradients_lm, self.max_norm)
            self.learning_rate_3 = tf.train.exponential_decay(self.learning_rate, self.global_step_3,
                                                              decay_steps=100000,
                                                              decay_rate=0.98, staircase=True)
            self.optimizer_lm = tf.train.AdamOptimizer(self.learning_rate_3)
            self.optim = self.optimizer_lm.apply_gradients(zip(self.clipped_gradients_lm, self.params_lm),
                                                           global_step=self.global_step_3)
            # self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.output_loss)
        self.saver = tf.train.Saver()
        self.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.config.gpu_options.allow_growth = True
        self.init = tf.global_variables_initializer()
        self.session = tf.Session(config=self.config)
        self.trainable_var = tf.trainable_variables()
        self.total_var_num = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
        self.session.run(self.init)

    def LSTM_Layer(self, input_, sequence_len, time_major=False):
        output_, final_sequence_ = tf.nn.dynamic_rnn(self.cell_fw,
                                                     input_,
                                                     dtype=tf.float32,
                                                     sequence_length=sequence_len,
                                                     time_major=time_major)
        return output_, final_sequence_

    def train_lang(self, sequence_in):
        feed = {self.sentence_in: sequence_in
                }
        return self.session.run([self.output_loss, self.optim], feed_dict=feed)

    def predict_lang(self, sequence_in):
        feed = {self.sentence_in: sequence_in}
        return self.session.run([self.prediction_loss, self.pred_logits], feed_dict=feed)

    def generate_lang(self, sequence_in):
        feed = {self.sentence_in: sequence_in}
        return self.session.run([self.next_output], feed_dict=feed)

    def load(self, location, model_name, exp_num):
        self.saver.restore(self.session, location + model_name + '_' + exp_num + ".ckpt")
        logging.info("Model loaded from path: %s" % location)

    def get_train_var(self):
        return self.session.run([self.trainable_var])

    def get_total_var_num(self):
        return self.session.run([self.total_var_num])

    def save(self, location, model_name, experiment_number):
        save_path = self.saver.save(self.session, location + model_name + "_" + experiment_number + ".ckpt")
        logging.info("Model saved in path: %s" % save_path)
