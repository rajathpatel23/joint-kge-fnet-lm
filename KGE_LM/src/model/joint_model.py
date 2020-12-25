import tensorflow as tf
import numpy as np
import sys
import io
import logging


class NAM_Modified(object):

    def __init__(self, **kwargs):
        super(NAM_Modified, self).__init__()
        self.lstm_layer = kwargs['lstm_layer']
        self.lstm_dim = self.lstm_layer[-1]
        self.balance = kwargs['balance']
        self.layer_type = kwargs['layer_type']
        self.decay = True
        self.starter_learning_rate = kwargs['learning_rate_kge'] or 1e-4
        self.learning_rate_lm = kwargs['learning_rate_lm'] or 1e-3
        self.hidden_units_1 = kwargs['hidden_units_1'] or 1
        self.hidden_units_2 = kwargs['hidden_units_2'] or None
        self.hidden_units_3 = kwargs['hidden_units_3'] or None
        self.hidden_units_4 = kwargs['hidden_units_4'] or None
        self.drop_out = kwargs['dropout']
        self.final = kwargs['final']
        self.embedding_matrix = kwargs['embedding_matrix'].astype(np.float32)
        self.task = kwargs['task']
        self.vocabulary_size = self.embedding_matrix.shape[0]
        # self.embedding_matrix =
        self.max_norm = 5.0
        if self.final and self.layer_type == 'BiLSTM':
            self.Input_dimension = self.lstm_layer[-1] * 2
        elif self.layer_type == 'BiLSTM' and not self.final:
            self.Input_dimension = self.lstm_layer[-1] * 4
        else:
            self.Input_dimension = self.lstm_layer[-1]
        with tf.device('/cpu:0'):
            self.embedding_mat = tf.get_variable(initializer=self.embedding_matrix, name="embedding_matrix",
                                                 trainable=True, dtype=tf.float32)
        with tf.device('/gpu:0'):
            self.tail = tf.placeholder(dtype=tf.int32, name='y_true')
            self.head = tf.placeholder(dtype=tf.int32, name='head_in')
            self.rel = tf.placeholder(dtype=tf.int32, name='tail_in')
            self.y_true_1 = tf.placeholder(dtype=tf.float32, name="y_true_values")
            self.sequence_in = tf.placeholder(dtype=tf.float32, shape=(None, None, 300), name="sequence_in")
            self.embed_head = tf.nn.embedding_lookup(self.embedding_mat, self.head)
            self.embed_tail = tf.nn.embedding_lookup(self.embedding_mat, self.tail)
            self.embed_relation = tf.nn.embedding_lookup(self.embedding_mat, self.rel)
            self.embed_total = tf.concat([self.embed_head, self.embed_relation, self.embed_tail], axis=1)
            self.sequence_sent = tf.placeholder(dtype=tf.int32, shape=(None, None), name='sentence_in')
            self.task = "INIT"
            self.current_view = 1
            self.lstm_fw = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.lstm_layer]
            self.lstm_bw = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.lstm_layer]
            self.drops_fw = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.drop_out) for lstm in
                             self.lstm_fw]
            self.drops_bw = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.drop_out) for lstm in
                             self.lstm_bw]
            self.cell_fw = tf.contrib.rnn.MultiRNNCell(self.drops_fw)
            self.cell_bw = tf.contrib.rnn.MultiRNNCell(self.drops_bw)
            self.weights = {
                'l1': tf.get_variable("Wl1", shape=[self.Input_dimension, self.hidden_units_1], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
                'l2': tf.get_variable("Wl2", shape=[self.hidden_units_1, self.hidden_units_2], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
                'l3': tf.get_variable("Wl3", shape=[self.hidden_units_2, self.hidden_units_3], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
                'l4': tf.get_variable("Wl4", shape=[self.hidden_units_3, self.hidden_units_4], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
            }
            self.biases = {
                'l1': tf.get_variable("bl1", shape=[self.hidden_units_1], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
                'l2': tf.get_variable("bl2", shape=[self.hidden_units_2], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
                'l3': tf.get_variable("bl3", shape=[self.hidden_units_3], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
                'l4': tf.get_variable("bl4", shape=[self.hidden_units_4], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(), trainable=True),
            }
        if self.task == 'KGE' or self.task == 'INIT':
            with tf.device('/gpu:0'):
                if self.layer_type == 'BiLSTM':
                    self.lstm_output_1, self.final_state_1 = self.LSTM_Layer(self.sequence_in, time_major=False)
                    self.lstm_output_1 = tf.concat(self.lstm_output_1, 2)
                    self.final_state_fw, self.final_state_bw = self.final_state_1
                    self.final_state_c_fw, self.final_state_c_bw = self.final_state_fw[-1].c, self.final_state_bw[-1].c
                    self.final_state_c = tf.concat([self.final_state_c_fw, self.final_state_c_bw], axis=1)
                    logging.info("state 1 & 2 : {}{}".format(self.final_state_c_fw, self.final_state_c_bw))
                if self.layer_type == 'LSTM':
                    self.lstm_output_1, self.final_state_1 = self.f_LSTM_Layer(self.sequence_in, time_major=False)
                    self.final_state_c, self.final_state_h = self.final_state_1[-1].c, self.final_state_1[-1].h
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

        if self.task == 'LM' or self.task == 'INIT':
            with tf.device('/gpu:1'):
                if self.layer_type == 'BiLSTM':
                    self.sentence_out = self.sequence_sent[:, 1:-1]
                    self.sentence_embed = tf.nn.embedding_lookup(self.embedding_mat, self.sequence_sent)
                    self.seq_len = tf.reduce_sum(tf.sign(self.sequence_sent), 1)
                    self.birnn_outputs, self.final_sentence = self.LSTM_Layer(input_=self.sentence_embed,
                                                                          time_major=False,
                                                                          sequence_length=self.seq_len)
                    self.fw_outputs = self.birnn_outputs[0][:, :-2, :]
                    self.bw_outputs = self.birnn_outputs[1][:, 2:, :]
                    self.final_output = self.fw_outputs[:, -1, :]
                if self.layer_type == 'LSTM':
                    self.sentence_in = self.sequence_sent[:, :-1]
                    self.sentence_out = self.sequence_sent[:, 1:]
                    self.sentence_embed = tf.nn.embedding_lookup(self.embedding_mat, self.sentence_in)
                    self.seq_len = tf.reduce_sum(tf.sign(self.sentence_in), 1)
                    self.rnn_outputs, self.final_sentence = self.f_LSTM_Layer(input_=self.sentence_embed, time_major=False, sequence_length=self.seq_len)
                    self.fw_outputs = self.rnn_outputs
                    self.final_output = self.fw_outputs[:, -1, :]
            with tf.device('/gpu:2'):
                self.a = self.lstm_dim
                self.weights = tf.get_variable(dtype=tf.float32, shape=(self.a, self.vocabulary_size),
                                               name="softmax_weights")
                self.bias = tf.get_variable(dtype=tf.float32, shape=self.vocabulary_size, name='softmax_bias')
                self.logits = tf.matmul(self.fw_outputs, self.weights) + self.bias
                self.next_output = tf.nn.softmax(tf.matmul(self.final_output, self.weights) + self.bias)
                print(self.next_output.shape)

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

        if self.task == 'KGE' or self.task == 'INIT':
            with tf.device('/gpu:0'):
                self.global_step_2 = tf.Variable(0, trainable=False)
                self.params_kge = tf.trainable_variables()
                self.gradients_kge = tf.gradients(self.cost, self.params_kge)
                self.clipped_gradients_kge, _ = tf.clip_by_global_norm(self.gradients_kge, self.max_norm)
                self.learning_rate_2 = tf.train.exponential_decay(self.starter_learning_rate, self.global_step_2,
                                                                  decay_steps=100000,
                                                                  decay_rate=0.05, staircase=True)
                self.optimizer_kge = tf.train.AdamOptimizer(self.learning_rate_2)
                self.train_op = self.optimizer_kge.apply_gradients(zip(self.clipped_gradients_kge, self.params_kge),
                                                                   global_step=self.global_step_2)
        if self.task == 'LM' or self.task == "INIT":
            with tf.device('/gpu:1'):
                self.global_step_3 = tf.Variable(0, trainable=False)
                self.params_lm = tf.trainable_variables()
                self.gradients_lm = tf.gradients(self.output_loss, self.params_lm)
                self.clipped_gradients_lm, _ = tf.clip_by_global_norm(self.gradients_lm, self.max_norm)
                self.learning_rate_3 = tf.train.exponential_decay(self.learning_rate_lm, self.global_step_3,
                                                                  decay_steps=100000,
                                                                  decay_rate=0.5, staircase=True)
                self.optimizer_lm = tf.train.AdamOptimizer(self.learning_rate_3)
                self.optim_lm = self.optimizer_lm.apply_gradients(zip(self.clipped_gradients_lm, self.params_lm),
                                                                  global_step=self.global_step_3)

        self.saver = tf.train.Saver(tf.trainable_variables())
        self.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.config.gpu_options.allow_growth = True
        self.init = tf.global_variables_initializer()
        self.session = tf.Session(config=self.config)
        self.total_var_num = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
        self.session.run(self.init)

    def get_embed(self, head_i, tail_i, rel_i):
        feed = {
            self.head: head_i, self.tail: tail_i, self.rel: rel_i
        }
        return self.session.run([self.embed_head, self.embed_relation, self.embed_tail, self.embed_total],
                                feed_dict=feed)

    def LSTM_Layer(self, input_, time_major=False, sequence_length=None):
        lstm_out, final_out = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw,
                                                              input_, time_major=time_major,
                                                              sequence_length=sequence_length,
                                                              dtype=tf.float32)
        return lstm_out, final_out

    def f_LSTM_Layer(self, input_, time_major=False, sequence_length=None):
        output_, final_sequence_ = tf.nn.dynamic_rnn(self.cell_fw,
                                                     input_,
                                                     dtype=tf.float32,
                                                     sequence_length=sequence_length,
                                                     time_major=time_major)
        return output_, final_sequence_


    def train_lang(self, sequence_in):
        self.task = 'LM'
        feed = {self.sequence_sent: sequence_in
                }
        return self.session.run([self.output_loss, self.optim_lm], feed_dict=feed)

    def predict_lang(self, sequence_in):
        self.task = 'LM'
        feed = {self.sequence_sent: sequence_in}
        return self.session.run([self.prediction_loss, self.pred_logits], feed_dict=feed)

    def generate_lang(self, sequence_in):
        feed = {self.sequence_sent: sequence_in}
        return self.session.run([self.next_output], feed_dict=feed)

    def debug(self, sequence):
        self.task = 'KGE'
        feed = {
            self.sequence_in: sequence
        }
        return self.session.run([self.lstm_output_1, self.final_state_1], feed_dict=feed)

    def fit(self, sequence, y_out):
        self.task = 'KGE'
        feed = {self.y_true_1: y_out,
                self.sequence_in: sequence
                }
        return self.session.run([self.cost, self.train_op], feed_dict=feed)

    def predict(self, sequence):
        self.task = 'KGE'
        feed = {
            self.sequence_in: sequence
        }
        return self.session.run([self.output_2], feed_dict=feed)

    # def get_train_var(self):
    #     return self.session.run([self.trainable_var])
    #
    def get_total_var_num(self):
        return self.session.run([self.total_var_num])

    def save(self, location, model_name, experiment_number):
        save_path = self.saver.save(self.session, location + model_name + "_" + experiment_number + ".ckpt")
        logging.info("Model saved in path: %s" % save_path)

    def close(self):
        self.session.close()
        logging.info("Model session closed")

    def load(self, location, model_name, exp_num):
        self.saver.restore(self.session, location + model_name + '_' + exp_num + ".ckpt")
        logging.info("Model loaded from path: %s" % location)
