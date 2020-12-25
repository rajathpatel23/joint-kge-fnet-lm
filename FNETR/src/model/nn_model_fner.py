# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
import logging

sys.path.append('../../')
sys.path.append('../')


# from create_prior_knowledge import create_prior
# import numpy as np

def create_prior(label2id_file):
    nodes = []
    num_label = 0
    with open(label2id_file) as f:
        for line in f:
            num_label += 1
            (id, label, freq) = line.strip().split()
            nodes += [label]

    prior = np.zeros((num_label, len(nodes)))
    with open(label2id_file) as f:
        for line in f:
            (id, label, freq) = line.strip().split()
            temp_ = label.split("/")[1:]
            temp = ["/" + "/".join(temp_[:q + 1]) for q in range(len(temp_))]
            code = []
            for i, node in enumerate(nodes):
                if node in temp:
                    code.append(1)
                else:
                    code.append(0)
            prior[int(id), :] = np.array(code)
    return prior


def weight_variable(name, shape, pad=False):
    initial = np.random.uniform(-0.01, 0.01, size=shape)
    if pad:
        initial[0] = np.zeros(shape[1])
    initial = tf.constant_initializer(initial)
    return tf.get_variable(name=name, shape=shape, initializer=initial, dtype=tf.float32, trainable=True)


class Model:
    def __init__(self, *args, **kwargs):
        encoder = kwargs['encoder']
        initializer = tf.contrib.layers.xavier_initializer()
        hier = kwargs['hier']
        feature = kwargs['feature']
        type_ = kwargs['type']
        self.max_norm = -1.0
        assert (encoder in ['hier-attention', 'attentive_figet'])
        assert (type_ in ["figer", "gillick", "BBN"])
        self.type = type
        self.encoder = encoder
        self.hier = hier
        self.feature = feature
        self.task = 'INIT'
        self.current_view = 1
        self.doc_vec = kwargs['doc_vec']
        # Hyperparameters
        self.context_length = 10
        self.emb_dim = 300
        self.target_dim = 113 if type_ == "figer" else 89
        self.feature_size = 600000 if type_ == "figer" else 100000
        self.learning_rate_fner = kwargs['learning_rate_fner']
        self.lstm_layer = kwargs['lstm_layer']
        self.session_graph = kwargs['session_graph']
        self.lstm_dim = self.lstm_layer[-1]
        self.att_dim = 300  # dim of attention module
        self.feature_dim = 50  # dim of feature representation
        self.feature_input_dim = 70
        self.coeff = 0.1
        if encoder == "averaging":
            self.rep_dim = self.emb_dim * 3
        else:
            # self.rep_dim = self.lstm_dim * 2 + self.emb_dim
            self.rep_dim = self.lstm_dim * 2 + self.emb_dim
        if feature:
            self.rep_dim += self.feature_dim
        if self.doc_vec:
            self.rep_dim += self.doc_dim
        self.doc_weight = 70
        # self.lstm_layer = kwargs['lstm_layer']
        self.Input_dimension = self.lstm_dim * 2
        self.drop_out = kwargs['dropout']
        self.embedding_matrix = kwargs['embedding_matrix']
        self.embedding_matrix = self.embedding_matrix.astype(np.float32)
        # Place Holders
        self.keep_prob = tf.placeholder(tf.float32)
        self.mention_representation = tf.placeholder(tf.float32, [None, self.emb_dim])
        # self.context = [tf.placeholder(tf.float32, [None, self.emb_dim]) for _ in range(self.context_length * 2 + 1)]
        self.context = [tf.placeholder(tf.float32, [None, self.emb_dim]) for _ in range(self.context_length * 2 + 1)]
        self.left_in = tf.placeholder(dtype=tf.int32, shape=(None, None), name='left_context')
        self.right_in = tf.placeholder(dtype=tf.int32, shape=(None, None), name='context_right')
        self.embedding_mat = tf.get_variable(initializer=self.embedding_matrix, name="embedding_matrix",
                                             trainable=False)

        self.lstm_fw = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.lstm_layer]
        self.lstm_bw = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.lstm_layer]
        self.drops_fw = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=1.0, output_keep_prob=self.keep_prob) for
                         lstm in self.lstm_fw]
        self.drops_bw = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=1.0, output_keep_prob=self.keep_prob) for
                         lstm in self.lstm_bw]
        self.cell_fw = tf.contrib.rnn.MultiRNNCell(self.drops_fw)
        self.cell_bw = tf.contrib.rnn.MultiRNNCell(self.drops_bw)
        self.target = tf.placeholder(tf.float32, [None, self.target_dim])
        self.sentence_in = tf.placeholder(tf.float32, shape=[None, None, self.emb_dim])
        self.mention_representation_dropout = tf.nn.dropout(self.mention_representation, self.keep_prob)

        if self.encoder == 'hier-attention':
            self.left_context = tf.nn.embedding_lookup(self.embedding_mat, self.left_in)
            print(self.left_context)
            self.right_context = tf.nn.embedding_lookup(self.embedding_mat, self.right_in)
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
                self.w_omega = tf.get_variable("w_omega", shape=[self.lstm_dim * 2, self.att_dim], dtype=tf.float32,
                                               trainable=True)
                self.b_omega = tf.get_variable("b_omega", shape=[self.att_dim], dtype=tf.float32, trainable=True)
                self.u_omega = tf.get_variable("u_omega", shape=[self.att_dim], dtype=tf.float32, trainable=True)
                self.context_representation, self.att_1 = self.attention_m(self.total_context)
                print("This is context rep: ", self.context_representation.shape)
        if feature:
            self.features = tf.placeholder(tf.int32, [None, self.feature_input_dim])
            self.feature_embeddings = weight_variable('feat_embds', (self.feature_size, self.feature_dim), True)
            self.feature_representation = tf.nn.dropout(
                tf.reduce_mean(tf.nn.embedding_lookup(self.feature_embeddings, self.features), 1), self.keep_prob)
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
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit,
        #                                                                    labels=self.target))
        self.loss = tf.reduce_mean(tf.losses.hinge_loss(logits=self.logit, labels=self.target))

        self.global_step_2 = tf.Variable(0, trainable=False)
        self.params_fner = tf.trainable_variables()
        self.gradients_fner = tf.gradients(self.loss, self.params_fner)

        self.clipped_gradients_fner, _ = tf.clip_by_global_norm(self.gradients_fner, self.max_norm)
        self.learning_rate_2 = tf.train.exponential_decay(self.learning_rate_fner, self.global_step_2,
                                                          decay_steps=1000,
                                                          decay_rate=0.25, staircase=True)
        self.optimizer_fner = tf.train.AdamOptimizer(self.learning_rate_fner)
        self.optim_fner = self.optimizer_fner.apply_gradients(zip(self.clipped_gradients_fner, self.params_fner),
                                                              global_step=self.global_step_2)

        # if self.task == 'FNER' or self.task == 'INIT':
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
        if self.doc_vec == True and doc_vector is not None:
            feed[self.doc_sequence] = doc_vector
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
