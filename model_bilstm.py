#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from common import config
import tensorflow.contrib as tf_contrib


class BiLSTM:
    def __init__(self):
        # set the initializer of conv_weight and conv_bias
        self.weight_init = tf_contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                          mode='FAN_IN', uniform=False)
        self.bias_init = tf.zeros_initializer()
        self.reg = tf_contrib.layers.l2_regularizer(config.weight_decay)

    def _embed_layer(self, name, inp):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable(initializer=tf.random_uniform([config.vocab_size, config.embeddingsize], -1.0, 1.0), name='W')
            embedded_chars = tf.nn.embedding_lookup(W, tf.cast(inp, tf.int32))
        return tf.cast(embedded_chars, tf.float32)

    def _lstm_layer(self, name, inp, rnn_size):
        with tf.variable_scope(name) as scope:
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_size, forget_bias=1.0, name="fw")
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_size, forget_bias=1.0, name="bw")
            hiddens, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                                  cell_bw=lstm_bw_cell,
                                                                                  inputs=inp,
                                                                                  dtype=tf.float32)
        return tf.concat(hiddens, 2)

    def _attention(self, name, inp, attention_size):
        with tf.variable_scope(name) as scope:
            hidden_size = inp.shape[2].value  # D value - hidden size of the RNN layer

            # Trainable parameters
            w_omega = tf.get_variable(name="w", initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
            b_omega = tf.get_variable(name="b", initializer=tf.random_normal([attention_size], stddev=0.1))
            u_omega = tf.get_variable(name="u", initializer=tf.random_normal([attention_size], stddev=0.1))

            with tf.name_scope('v'):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
                v = tf.tanh(tf.tensordot(inp, w_omega, axes=1) + b_omega)

            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
            vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
            alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

            # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
            output = tf.reduce_sum(inp * tf.expand_dims(alphas, -1), 1)

            return output

    def _fc_layer(self, name, inp, units, dropout=0.5):
        with tf.variable_scope(name) as scope:
            shape = inp.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(inp, [-1, dim]) # flatten
            if dropout > 0:
                x = tf.nn.dropout(x, keep_prob=dropout, name='dropout')
            x = tf.layers.dense(x, units, kernel_initializer=self.weight_init,
                                bias_initializer=self.bias_init, kernel_regularizer=self.reg)
        return x

    def build(self):
        data = tf.placeholder(tf.float32, shape=(None, config.padding_size), name='data')
        label = tf.placeholder(tf.int32, shape=(None,), name='label')
        label_onehot = tf.one_hot(label, config.nr_class, dtype=tf.int32)
        is_training = tf.placeholder(tf.bool, name='is_training')

        # embeding
        x = self._embed_layer(name='embed_input', inp=data)  # [batch_size, padding_length, embeding_length]

        # bilstm
        x = self._lstm_layer(name="lstm1", inp=x, rnn_size=16)

        # attention
        x = self._attention(name="attention", inp=x, attention_size=16)

        # fc1
        logits = self._fc_layer(name='fc1', inp=x, units=config.nr_class, dropout=0)

        placeholders = {
            'data': data,
            'label': label,
            'is_training': is_training,
        }
        return placeholders, label_onehot, logits



