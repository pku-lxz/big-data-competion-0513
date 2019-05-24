#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from common import config
import tensorflow.contrib as tf_contrib
import os
import argparse
from dataset import Dataset


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

    def _conv_layer(self, name, inp, kernel_shape, stride, padding='SAME',is_training=False):
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

    def _pool_layer(self, name, inp, ksize, stride, padding='SAME', mode='MAX'):
        assert mode in ['MAX', 'AVG'], 'the mode of pool must be MAX or AVG'
        if mode == 'MAX':
            x = tf.nn.max_pool(inp, ksize=[1, ksize, 1, 1], strides=[1, stride, stride, 1],
                               padding=padding, name=name, data_format='NHWC')
        elif mode == 'AVG':
            x = tf.nn.avg_pool(inp, ksize=[1, ksize, 1, 1], strides=[1, stride, stride, 1],
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
        x = self._embed_layer(name='embed_input', inp=data) # [batch_size, padding_length, embeding_length, channel=1]

        # conv
        x = self._conv_layer(name='conv1', inp=x,
                             kernel_shape=[1, config.embeddingsize, 1,  32], stride=1,
                             is_training=is_training)  # N*padding_length*1*32
        x = self._pool_layer(name='pool1', inp=x, ksize=config.padding_size, stride=1, mode='MAX')  # Nx1x1x16

        # fc1
        logits = self._fc_layer(name='fc1', inp=x, units=config.nr_class, dropout=0)

        placeholders = {
            'data': data,
            'label': label,
            'is_training': is_training,
        }
        return placeholders, label_onehot, logits


def main():
    data = Dataset()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue', dest='continue_path', required=False)
    parser.add_argument('-l', '--loss', default='softmax')
    args = parser.parse_args()

    network = TextCNN()
    placeholders, label_onehot, logits = network.build()

    correct_pred = tf.equal(tf.cast(tf.argmax(tf.nn.softmax(logits), 1), dtype=tf.int32),
                            tf.cast(tf.argmax(label_onehot, 1), dtype=tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = tf.losses.softmax_cross_entropy(label_onehot, logits) + loss_reg

    ## train config
    global_steps = tf.Variable(0, trainable=False)
    opt = tf.train.AdamOptimizer(config.lr)
    # in order to update BN in every iter
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train = opt.minimize(loss)

    ## init tensorboard
    tf.summary.scalar('loss_regularization', loss_reg)
    tf.summary.scalar('loss_crossEntropy', loss - loss_reg)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('learning_rate', config.lr)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'tf_log', 'train'),
                                             tf.get_default_graph())
    test_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'tf_log', 'test'),
                                            tf.get_default_graph())

    ## create a session
    tf.set_random_seed(12345) # ensure consistent results
    global_cnt = 0
    epoch_start = 0
    g_list = tf.global_variables()
    saver = tf.train.Saver(var_list=g_list)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # init all variables
        if args.continue_path:  # load a model snapshot
            ckpt = tf.train.get_checkpoint_state(args.continue_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            epoch_start = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[1])
            global_cnt = epoch_start * config.train_size//config.batch_size

        ## training
        for epoch in range(epoch_start + 1, config.nr_epoch + 1):
            for _ in range(config.train_size//config.batch_size):
                images, labels = data.one_batch_train()
                global_cnt += 1
                feed_dict = {
                    placeholders['data']: images,
                    placeholders['label']: labels,
                    global_steps: global_cnt,
                    placeholders['is_training']: True,
                }
                _, loss_v, loss_reg_v, acc_v, lr_v, summary = sess.run([train, loss, loss_reg,
                                                                        accuracy, config.lr, merged],
                                                                       feed_dict=feed_dict)
                if global_cnt % config.show_interval == 0:
                    train_writer.add_summary(summary, global_cnt)
                    print(
                        "e:{},{}/{}".format(epoch, global_cnt % config.train_size//config.batch_size,
                                            config.train_size//config.batch_size),
                        'loss: {:.3f}'.format(loss_v),
                        'loss_reg: {:.3f}'.format(loss_reg_v),
                        'acc: {:.3f}'.format(acc_v),
                        'lr: {:.3f}'.format(lr_v),
                    )

            ## save model
            if epoch % config.snapshot_interval == 0:
                saver.save(sess, os.path.join(config.log_model_dir, 'epoch-{}'.format(epoch)),
                           global_step=global_cnt)

        print('Training is done, exit.')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os._exit(1)
