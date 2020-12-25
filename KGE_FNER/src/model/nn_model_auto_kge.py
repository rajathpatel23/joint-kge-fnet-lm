import tensorflow as tf
import logging
import numpy as np
import pandas as pd
import sys
# sys.path.append('../../')
# sys.path.append('../')


def weight_variable(name, shape, pad=False):
    initial = np.random.uniform(-0.01, 0.01, size=shape)
    if pad:
        initial[0] = np.zeros(shape[1])
    initial = tf.constant_initializer(initial)
    return tf.get_variable(name=name, shape=shape, initializer=initial, dtype=tf.float32, trainable=True)


class NAM_Modified(object):
    def __init__(self, *args, **kwargs):
        super(NAM_Modified, self).__init__()
        self.lstm_layer = kwargs['lstm_layer']
        self.balance = kwargs['balance']
        encoder = kwargs['encoder']
        type_ = kwargs['type']
        self.feature = kwargs['feature']
        feature = self.feature
        assert (encoder in ['hier-attention', 'attentive_figet'])
        assert (type_ in ["figer", "gillick", "BBN", "wikiauto"])
        self.max_norm = -1.0
        self.type = type_
        self.encoder = encoder
        self.decay = True
        self.doc_vec = False
        self.att_dim = 100  # dim of attention module
        self.feature_dim = 50  # dim of feature representation
        self.feature_input_dim = 70
        self.emb_dim = 300
        self.context_length = 10
        self.lstm_dim = self.lstm_layer[-1]
        self.feature_size = 600000 if type_ == "figer" else 100000
        self.coeff = 0.1
        if encoder == "averaging":
            self.rep_dim = self.emb_dim * 3
        else:
            self.rep_dim = self.lstm_dim * 2 + self.emb_dim
        if feature:
            self.rep_dim += self.feature_dim
        if self.doc_vec:
            self.rep_dim += self.doc_dim
        if type_ == 'figer':
            self.target_dim = 113
        if type_ == 'gillick':
            self.target_dim = 89
        if type_ == 'BBN':
            self.target_dim = 47
        if type_ == 'wikiauto':
            self.target_dim = 74
        self.starter_learning_rate = kwargs['learning_rate_kge'] or 1e-4
        self.learning_rate_fner= kwargs['learning_rate_fner']
        self.hidden_units_1 = kwargs['hidden_units_1'] or 1
        self.hidden_units_2 = kwargs['hidden_units_2'] or None
        self.hidden_units_3 = kwargs['hidden_units_3'] or None
        self.hidden_units_4 = kwargs['hidden_units_4'] or None
        self.drop_out = kwargs['dropout']
        self.split = kwargs['split']
        self.final = kwargs['final']
        self.embedding_matrix = kwargs['embedding_matrix_kge'].astype(np.float32)
        self.embedding_matrix_1 = kwargs['embedding_matrix_fner'].astype(np.float32)
        self.learning_rate_joint = kwargs['learning_rate_joint'] or 1e-3
        self.task = kwargs['task']
        self.session_graph = kwargs['session_graph']
        self.joint_opt = kwargs['joint_opt']
        self.vocabulary_size = self.embedding_matrix.shape[0]
        if self.final:
            self.Input_dimension = self.lstm_layer[-1] * 2
        else:
            self.Input_dimension = self.lstm_layer[-1] * 4
        # with tf.device('/gpu:0'):
        self.embedding_mat = tf.get_variable(initializer=self.embedding_matrix, name="embedding_matrix_kge",
                                             trainable=True, dtype=tf.float32)
        self.embedding_mat_1 = tf.get_variable(initializer=self.embedding_matrix_1, name="embedding_matrix_fner",
                                               trainable=False)

        # with tf.device('/gpu:0'):
        self.tail = tf.placeholder(dtype=tf.int32, name='y_true')
        self.head = tf.placeholder(dtype=tf.int32, name='head_in')
        self.rel = tf.placeholder(dtype=tf.int32, name='tail_in')
        self.keep_prob = tf.placeholder(tf.float32)
        self.mention_representation = tf.placeholder(tf.int32, shape=(None, None))
        self.y_true_1 = tf.placeholder(dtype=tf.float32, name="y_true_values")
        self.sequence_in = tf.placeholder(dtype=tf.float32, shape=(None, None, 300), name="sequence_in")
        self.embed_head = tf.nn.embedding_lookup(self.embedding_mat, self.head)
        self.embed_tail = tf.nn.embedding_lookup(self.embedding_mat, self.tail)
        self.embed_relation = tf.nn.embedding_lookup(self.embedding_mat, self.rel)
        self.embed_total = tf.concat([self.embed_head, self.embed_relation, self.embed_tail], axis=1)
        self.target = tf.placeholder(tf.float32, [None, self.target_dim])
        self.sequence_sent = tf.placeholder(dtype=tf.int32, shape=(None, None), name='sentence_in')
        self.sentence_in = tf.placeholder(tf.float32, shape=[None, None, self.emb_dim])
        self.mention_rep = tf.nn.embedding_lookup(self.embedding_matrix_1, self.mention_representation)
        self.mention_representation_dropout = tf.reduce_mean(tf.nn.dropout(self.mention_rep,
                                                                           self.keep_prob), axis=1)
        self.context = [tf.placeholder(tf.float32, [None, self.emb_dim]) for _ in
                        range(self.context_length * 2 + 1)]
        self.left_in = tf.placeholder(dtype=tf.int32, shape=(None, None), name='left_context')
        self.right_in = tf.placeholder(dtype=tf.int32, shape=(None, None), name='context_right')

        print(self.embed_total.shape)
        print(self.embed_total.shape)
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
            # with tf.device('/gpu:0'):
            self.lstm_output_1, self.final_state_1 = self.LSTM_Layer(self.sequence_in, time_major=False)
            self.lstm_output_1 = tf.concat(self.lstm_output_1, 2)
            print(self.lstm_output_1.shape)
            self.final_state_fw, self.final_state_bw = self.final_state_1
            self.final_state_c_fw, self.final_state_c_bw = self.final_state_fw[-1].c, self.final_state_bw[-1].c
            self.final_state_c = tf.concat([self.final_state_c_fw, self.final_state_c_bw], axis=1)
            logging.info("state 1 & 2 : {}{}".format(self.final_state_c_fw, self.final_state_c_bw))

            if self.current_view == 1 and self.split == 9:
                _, _, self.head_, _, _, self.rel_, _, _, self.tail_ = tf.split(self.lstm_output_1,
                                                                               num_or_size_splits=9,
                                                                               axis=1)
            elif self.current_view == 1 and self.split == 12:
                _, _, _, self.head_, _, _, _, self.rel_, _, _, _, self.tail_ = tf.split(self.lstm_output_1,
                                                                                        num_or_size_splits=12,
                                                                                        axis=1)
            elif self.current_view == 1 and self.split == 11:
                _, _, self.head_, _, _, _, _, self.rel_, _, _, self.tail_ = tf.split(self.lstm_output_1,
                                                                                     num_or_size_splits=11, axis=1)
            if self.final:
                print(self.final)
                self.z0 = self.final_state_c
                print("This is tail_: ", self.tail_.shape)
                self.tail_ = tf.squeeze(self.tail_)
            else:
                self.tail_ = tf.squeeze(self.tail_)
                print(self.tail_.shape)
                self.head_ = tf.squeeze(self.head_)
                print(self.head_.shape)
                self.rel_ = tf.squeeze(self.rel_)
                print(self.rel_)
                self.z0 = tf.concat([self.head_, self.rel_], axis=1)
                print(self.z0.shape)
            self.l1 = tf.matmul(self.z0, self.weights['l1']) + self.biases['l1']
            self.z1 = tf.nn.relu(self.l1)
            self.l2 = tf.matmul(self.z1, self.weights['l2']) + self.biases['l2']
            self.z2 = tf.nn.relu(self.l2)
            self.l3 = tf.matmul(self.z2, self.weights['l3']) + self.biases['l3']
            self.z3 = tf.nn.relu(self.l3)
            self.m = tf.multiply(self.z3, self.tail_)
            self.dot = tf.reduce_sum(self.m, axis=1)
            self.output = self.dot
            self.output_2 = tf.nn.sigmoid(self.output)
            self.beta = 0.01
            self.regularizer = tf.nn.l2_loss(self.weights['l3']) \
                               + tf.nn.l2_loss(self.weights['l2']) + tf.nn.l2_loss(self.weights['l1'])
            self.regularizer = self.beta * self.regularizer
            self.cost = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(self.y_true_1, self.output, pos_weight=self.balance) +
                self.regularizer)

        if self.task == 'FNER' or self.task == 'INIT':
            # with tf.device('/gpu:0'):
            if self.encoder == 'hier-attention':
                self.left_context = tf.nn.embedding_lookup(self.embedding_mat_1, self.left_in)
                print(self.left_context)
                self.right_context = tf.nn.embedding_lookup(self.embedding_mat_1, self.right_in)
                print(self.right_context)
                self.out_context_left, self.final_left = self.LSTM_Layer(self.left_context, time_major=False)
                self.out_context_right, self.final_right = self.LSTM_Layer(self.right_context, time_major=False)
                self.out_context_left = tf.concat(self.out_context_left, 2)
                self.out_context_right = tf.concat(self.out_context_right, 2)
                self.total_context = tf.concat([self.out_context_left, self.out_context_right], axis=1)
                print("This is total context", self.total_context.shape)
                # self.total_context = tf.transpose(self.total_context, [1, 0, 2])
                # print("This is total context: ", self.total_context.shape)
                if self.encoder == 'hier-attention':
                    self.w_omega = tf.get_variable("w_omega", shape=[self.lstm_dim * 2, self.att_dim],
                                                   dtype=tf.float32,
                                                   trainable=True)
                    self.b_omega = tf.get_variable("b_omega", shape=[self.att_dim], dtype=tf.float32,
                                                   trainable=True)
                    self.u_omega = tf.get_variable("u_omega", shape=[self.att_dim], dtype=tf.float32,
                                                   trainable=True)
                    self.context_representation, self.att_1 = self.attention_m(self.total_context)
                    print("This is context rep: ", self.context_representation.shape)
            if self.feature:
                self.features = tf.placeholder(tf.int32, [None, self.feature_input_dim])
                self.feature_embeddings = weight_variable('feat_embds', (self.feature_size, self.feature_dim), True)
                self.feature_representation = tf.nn.dropout(
                    tf.reduce_mean(tf.nn.embedding_lookup(self.feature_embeddings, self.features), 1),
                    self.keep_prob)
                self.representation = tf.concat([self.mention_representation_dropout, self.feature_representation,
                                                 self.context_representation], axis=1)
                print(self.representation)
            else:
                self.representation = tf.concat(
                    [self.mention_representation_dropout, self.context_representation],
                    axis=1)
            self.V_1 = weight_variable('n_W1', (self.rep_dim, 1024))
            self.V_2 = weight_variable('n_W2', (1024, 512))
            self.V_3 = weight_variable('n_W3', (512, self.target_dim))
            self.y_1 = tf.matmul(self.representation, self.V_1)
            self.y1 = tf.contrib.layers.batch_norm(self.y_1, center=True, scale=True)
            self.y_1 = tf.nn.relu(self.y_1)
            self.y_2 = tf.matmul(self.y_1, self.V_2)
            self.y_2 = tf.nn.relu(self.y_2)
            self.logit = tf.matmul(self.y_2, self.V_3)
            self.distribution = self.logit
            self.loss = tf.reduce_mean(tf.losses.hinge_loss(logits=self.logit, labels=self.target))

        if self.joint_opt:
            # with tf.device('/gpu:0'):
            self.lambda_ = 1
            # if self.joint_optimizer:
            self.total_loss = self.cost + self.lambda_ * self.loss
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate_1 = tf.train.exponential_decay(self.learning_rate_joint, self.global_step,
                                                              decay_steps=100000,
                                                              decay_rate=0.5, staircase=True)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate_joint)
            self.optim_total = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

        if self.task == 'FNER' or self.task == 'INIT':
            # with tf.device('/gpu:0'):
            self.global_step_2 = tf.Variable(0, trainable=False)
            self.params_fner = tf.trainable_variables()
            self.gradients_fner = tf.gradients(self.loss, self.params_fner)
            self.clipped_gradients_fner, _ = tf.clip_by_global_norm(self.gradients_fner, self.max_norm)
            self.learning_rate_2 = tf.train.exponential_decay(self.learning_rate_fner, self.global_step_2,
                                                              decay_steps=4000,
                                                              decay_rate=0.25, staircase=True)
            self.optimizer_fner = tf.train.AdamOptimizer(self.learning_rate_2)
            self.optim_fner = self.optimizer_fner.apply_gradients(zip(self.clipped_gradients_fner, self.params_fner),
                                                                  global_step=self.global_step_2)
        if self.task == 'KGE' or self.task == 'INIT':
        # with tf.device('/gpu:0'):
            self.global_step_1 = tf.Variable(0, trainable=False)
            self.params_kge = tf.trainable_variables()
            self.gradients_kge = tf.gradients(self.cost, self.params_kge)
            self.clipped_gradients_kge, _ = tf.clip_by_global_norm(self.gradients_kge, self.max_norm)
            self.learning_rate_3 = tf.train.exponential_decay(self.starter_learning_rate, self.global_step_1,
                                                              decay_steps=10000,
                                                              decay_rate=0.05, staircase=True)
            self.optimizer_kge = tf.train.AdamOptimizer(self.learning_rate_3)
            self.train_op = self.optimizer_kge.apply_gradients(zip(self.clipped_gradients_kge, self.params_kge),
                                                               global_step=self.global_step_1)

        self.fner_loss_summary = tf.summary.scalar("FNER_loss", self.loss)
        self.attn_sent_fner = tf.summary.histogram("FNER_attn_histogram", self.att_1)
        self.summary_op_fner = tf.summary.merge([self.attn_sent_fner, self.fner_loss_summary])

        self.saver = tf.train.Saver(tf.trainable_variables())
        self.config = tf.ConfigProto(device_count={'GPU': 1})
        self.config.gpu_options.allow_growth = True
        self.init = tf.global_variables_initializer()
        self.session = tf.Session(config=self.config)
        self.summary_writer = tf.summary.FileWriter(self.session_graph + "summary_log", self.session.graph)
        self.session = tf.Session(config=self.config)
        self.trainable_var = tf.trainable_variables()
        self.total_var_num = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
        self.session.run(self.init)

    def attention_m(self, inputs, sentence=False):
        v_in = tf.nn.tanh(tf.tensordot(inputs, self.w_omega, axes=1) + self.b_omega)
        v_out = tf.tensordot(v_in, self.u_omega, axes=1, name='vu')
        attn = tf.nn.softmax(v_out, name='alphas')
        attn_weight = inputs * tf.expand_dims(attn, -1)
        print("This is attn_weight: ", attn_weight.shape)
        if sentence:
            return attn_weight, attn
        cont_rep = tf.reduce_sum(attn_weight, axis=1)
        return cont_rep, attn

    def LSTM_Layer(self, input_, time_major=False):
        output_, final_sequence_ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw,
                                                                   input_,
                                                                   dtype=tf.float32,
                                                                   time_major=time_major)
        return output_, final_sequence_

    def get_embed(self, head_i, tail_i, rel_i):
        feed = {
            self.head: head_i, self.tail: tail_i, self.rel: rel_i
        }
        return self.session.run([self.embed_head, self.embed_relation, self.embed_tail, self.embed_total],
                                feed_dict=feed)

    def fit(self, sequence, y_out):
        self.task = 'KGE'
        feed = {self.y_true_1: y_out,
                self.sequence_in: sequence
                }
        return self.session.run([self.cost, self.train_op], feed_dict=feed)

    def predict_kge(self, sequence):
        self.task = 'KGE'
        feed = {
            self.sequence_in: sequence
        }
        return self.session.run([self.output_2], feed_dict=feed)

    def train_FNER(self, left_in, right_in, target_data, context_data=None,
                   mention_representation_data=None, feature_data=None,
                   doc_vector=None):
        self.task = 'FNER'
        feed = {self.target: target_data,
                self.keep_prob: [0.5],
                self.left_in: left_in,
                self.right_in: right_in}
        if self.feature == True and feature_data is not None:
            feed[self.features] = feature_data
        if context_data is not None:
            for i in range(self.context_length * 2 + 1):
                feed[self.context[i]] = context_data[:, i, :]
        if mention_representation_data is not None:
            feed[self.mention_representation] = mention_representation_data
        return self.session.run([self.loss, self.optim_fner, self.summary_op_fner, self.global_step_2], feed_dict=feed)

    def error(self, left_in, right_in, target_data, context_data=None,
              mention_representation_data=None, feature_data=None,
              doc_vector=None):
        self.task = 'FNER'
        feed = {self.target: target_data,
                self.keep_prob: [1.0],
                self.left_in: left_in,
                self.right_in: right_in}
        if self.feature == True and feature_data is not None:
            feed[self.features] = feature_data
        if self.doc_vec == True and doc_vector is not None:
            feed[self.doc_sequence] = doc_vector
        if context_data is not None:
            for i in range(self.context_length * 2 + 1):
                feed[self.context[i]] = context_data[:, i, :]
        if mention_representation_data is not None:
            feed[self.mention_representation] = mention_representation_data
        return self.session.run(self.loss, feed_dict=feed)

    def predict(self, left_in, right_in, context_data=None,
                mention_representation_data=None, feature_data=None,
                doc_vector=None):
        self.task = 'FNER'
        feed = {self.keep_prob: [1.0],
                self.left_in: left_in,
                self.right_in: right_in}
        if self.feature == True and feature_data is not None:
            feed[self.features] = feature_data
        if self.doc_vec == True and doc_vector is not None:
            feed[self.doc_sequence] = doc_vector
        if context_data is not None:
            for i in range(self.context_length * 2 + 1):
                feed[self.context[i]] = context_data[:, i, :]
        if mention_representation_data is not None:
            feed[self.mention_representation] = mention_representation_data
        return self.session.run(self.distribution, feed_dict=feed)

    def get_type_embed(self):
        return self.session.run([self.V_3])

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

    def close(self):
        self.session.close()
        logging.info("Model session closed")
