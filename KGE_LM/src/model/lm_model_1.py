
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
        self.layer_type = kwargs['layer_type']
        self.vocabulary_size = self.embedding_matrix.shape[0]
        self.a = self.lstm_dim
        self.sequence_sent = tf.placeholder(dtype=tf.int32, shape=(None, None), name='sentence_con')
        with tf.device('/cpu:0'):
            self.embedding_mat = tf.get_variable(dtype=tf.float32, initializer=self.embedding_matrix,
                                                 name='embedding_matrix', trainable=False)
        with tf.device('/gpu:0'):
            self.lstm_fw = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.lstm_layer]
            self.lstm_bw = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.lstm_layer]
            self.drops_fw = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=1.0, output_keep_prob=self.dropout) for
                             lstm in self.lstm_fw]
            self.drops_bw = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=1.0, output_keep_prob=self.dropout) for
                             lstm in self.lstm_bw]
            self.cell_fw = tf.contrib.rnn.MultiRNNCell(self.drops_fw)
            self.cell_bw = tf.contrib.rnn.MultiRNNCell(self.drops_bw)
            if self.layer_type == 'BiLSTM':
                self.sentence_out = self.sequence_sent[:, 1:-1]
                self.sentence_embed = tf.nn.embedding_lookup(self.embedding_mat, self.sequence_sent)
                self.seq_len = tf.reduce_sum(tf.sign(self.sequence_sent), 1)
                self.birnn_outputs, self.final_sentence = self.LSTM_Layer(input_=self.sentence_embed,
                                                                          time_major=False,
                                                                          sequence_len=self.seq_len)
                self.fw_outputs = self.birnn_outputs[0][:, :-2, :]
                self.bw_outputs = self.birnn_outputs[1][:, 2:, :]
                self.fw_outputs = self.birnn_outputs[0][:, :-2, :]
                self.final_output = self.fw_outputs[:, -1, :]
            if self.layer_type == 'LSTM':
                self.sentence_in = self.sequence_sent[:, :-1]
                self.sentence_out = self.sequence_sent[:, 1:]
                self.sentence_embed = tf.nn.embedding_lookup(self.embedding_mat, self.sentence_in)
                self.seq_len = tf.reduce_sum(tf.sign(self.sentence_in), 1)
                self.rnn_outputs, self.final_sentence = self.f_LSTM_Layer(input_=self.sentence_embed, time_major=False,
                                                                          sequence_length=self.seq_len)
                self.fw_outputs = self.rnn_outputs
                self.final_output = self.fw_outputs[:, -1, :]

        with tf.device('/gpu:1'):
            self.weights = tf.get_variable(dtype=tf.float32, shape=(self.a, self.vocabulary_size),
                                           name="softmax_weights")
            self.bias = tf.get_variable(dtype=tf.float32, shape=self.vocabulary_size, name='softmax_bias')
            self.logits = tf.matmul(self.fw_outputs, self.weights) + self.bias
            self.next_output = tf.nn.softmax(tf.matmul(self.final_output, self.weights) + self.bias)

            self.pred_logits = tf.nn.softmax(self.logits)
            if self.layer_type == 'BiLSTM':
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                             targets=self.sentence_out,
                                                             weights=tf.sequence_mask(
                                                                 self.seq_len - 2,
                                                                 tf.shape(self.sequence_sent)[
                                                                     1] - 2,
                                                                 dtype=tf.float32),
                                                             average_across_timesteps=False,
                                                             average_across_batch=False)
            if self.layer_type == 'LSTM':
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                             targets=self.sentence_out,
                                                             weights=tf.sequence_mask(
                                                                 self.seq_len,
                                                                 tf.shape(self.sequence_sent)[
                                                                     1] - 1,
                                                                 dtype=tf.float32),
                                                             average_across_timesteps=False,
                                                             average_across_batch=False)
            self.output_loss = tf.reduce_mean(tf.reduce_mean(self.loss, axis=1), axis=0)
            self.prediction_loss = tf.reduce_sum(tf.reduce_sum(self.loss, axis=1), axis=0)

        with tf.device('/gpu:2'):
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
        self.saver = tf.train.Saver()
        self.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.config.gpu_options.allow_growth = True
        self.init = tf.global_variables_initializer()
        self.session = tf.Session(config=self.config)
        self.trainable_var = tf.trainable_variables()
        self.total_var_num = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
        self.session.run(self.init)

    def LSTM_Layer(self, input_, sequence_len, time_major=False):
        output_, final_sequence_ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw,
                                                                   input_,
                                                                   dtype=tf.float32,
                                                                   sequence_length=sequence_len,
                                                                   time_major=time_major)
        return output_, final_sequence_

    def f_LSTM_Layer(self, input_, time_major=False, sequence_length=None):
        output_, final_sequence_ = tf.nn.dynamic_rnn(self.cell_fw,
                                                     input_,
                                                     dtype=tf.float32,
                                                     sequence_length=sequence_length,
                                                     time_major=time_major)
        return output_, final_sequence_

    def train_lang(self, sequence_in):
        feed = {self.sequence_sent: sequence_in
                }
        return self.session.run([self.output_loss, self.optim], feed_dict=feed)

    def predict_lang(self, sequence_in):
        feed = {self.sequence_sent: sequence_in}
        return self.session.run([self.prediction_loss, self.pred_logits], feed_dict=feed)

    def generate_lang(self, sequence_in):
        feed = {self.sequence_sent: sequence_in}
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
