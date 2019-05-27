#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from common import config
import tensorflow.contrib as tf_contrib


class TextCNN:
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
        return embedded_chars[:, :, :, tf.newaxis]

    def _conv_layer(self, name, inp, kernel_shape, stride, padding='VALID', is_training=False):
        with tf.variable_scope(name) as scope:
            conv_filter = tf.get_variable(name='filter', shape=kernel_shape,
                                          initializer=self.weight_init, regularizer=self.reg)
            conv_bias = tf.get_variable(name='bias', shape=kernel_shape[-1],
                                        initializer=self.bias_init)
            x = tf.nn.conv2d(inp, conv_filter, strides=[1, stride, stride, 1],
                             padding=padding, data_format='NHWC')
            x = tf.nn.bias_add(x, conv_bias, data_format='NHWC')
            x = tf.layers.batch_normalization(x, axis=3, training=is_training)
            x = tf.nn.relu(x)
        return x

    def _pool_layer(self, name, inp, ksize, stride, padding='VALID', mode='MAX'):
        assert mode in ['MAX', 'AVG'], 'the mode of pool must be MAX or AVG'
        if mode == 'MAX':
            x = tf.nn.max_pool(inp, ksize=[1, ksize, 1, 1], strides=[1, stride, 1, 1],
                               padding=padding, name=name, data_format='NHWC')
        elif mode == 'AVG':
            x = tf.nn.avg_pool(inp, ksize=[1, ksize, 1, 1], strides=[1, stride, 1, 1],
                               padding=padding, name=name, data_format='NHWC')
        return x

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
        label = tf.placeholder(tf.int32, shape=(None, ), name='label')
        label_onehot = tf.one_hot(label, config.nr_class, dtype=tf.int32)
        is_training = tf.placeholder(tf.bool, name='is_training')

        # embeding
        x = self._embed_layer(name='embed_input', inp=data) # [batch_size, padding_length, embeding_length]

        # conv1
        x_1 = self._conv_layer(name='conv1', inp=x,
                             kernel_shape=[1, config.embeddingsize, 1,  32], stride=1,
                             is_training=is_training)  # N*padding_length*1*32
        x_1 = self._pool_layer(name='pool1', inp=x_1, ksize=config.padding_size, stride=1, mode='MAX')  # Nx1x1x32

        # conv2
        x_2 = self._conv_layer(name='conv2', inp=x,
                               kernel_shape=[2, config.embeddingsize, 1, 32], stride=1,
                               is_training=is_training)  # N*padding_length*1*32
        x_2 = self._pool_layer(name='pool2', inp=x_2, ksize=config.padding_size - 1, stride=1, mode='MAX')  # Nx1x1x32

        # conv2
        x_3 = self._conv_layer(name='conv3', inp=x,
                               kernel_shape=[4, config.embeddingsize, 1, 32], stride=1,
                               is_training=is_training)  # N*padding_length*1*32
        x_3 = self._pool_layer(name='pool3', inp=x_3, ksize=config.padding_size - 3, stride=1, mode='MAX')  # Nx1x1x32

        # fc1
        logits = self._fc_layer(name='fc1', inp=tf.concat([x_1, x_2, x_3], 3), units=config.nr_class, dropout=0)

        placeholders = {
            'data': data,
            'label': label,
            'is_training': is_training,
        }
        return placeholders, label_onehot, logits
