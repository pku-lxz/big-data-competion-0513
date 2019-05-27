import tensorflow as tf
from common import config
from dataset_modern import Dataset
import argparse
from model_cnn import TextCNN
from model_bilstm import BiLSTM
import os
import pandas as pd
import numpy as np


def main():
    d = Dataset()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue', dest='continue_path', required=False)
    args = parser.parse_args()

    ## build graph
    network = TextCNN()
    placeholders, label_onehot, logits = network.build()

    preds = tf.nn.softmax(logits)

    auc_estimate, auc_op = tf.metrics.auc(label_onehot, preds)
    loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = tf.losses.softmax_cross_entropy(label_onehot, logits) + loss_reg

    ## train config
    global_steps = tf.Variable(0, trainable=False)
    boundaries = [config.train_size//config.batch_size*15, config.train_size//config.batch_size*40]
    values = [0.01, 0.001, 0.0005]
    lr = tf.train.piecewise_constant(global_steps, boundaries, values)
    opt = tf.train.AdamOptimizer(lr)
    # in order to update BN in every iter
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train = opt.minimize(loss)


    ## init tensorboard
    tf.summary.scalar('loss_regularization', loss_reg)
    tf.summary.scalar('loss_crossEntropy', loss - loss_reg)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('AUC', auc_estimate)
    tf.summary.scalar('learning_rate', lr)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'tf_log', 'train'),
                                         tf.get_default_graph())
    test_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'tf_log', 'validation'),
                                        tf.get_default_graph())

    ## create a session
    tf.set_random_seed(12345) # ensure consistent results
    global_cnt = 0
    epoch_start = 0
    g_list = tf.global_variables()
    saver = tf.train.Saver(var_list=g_list)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())  # init all variables
        if args.continue_path:  # load a model snapshot
            ckpt = tf.train.get_checkpoint_state(args.continue_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            epoch_start = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[1])
            global_cnt = epoch_start * config.train_size//config.batch_size

        ## training
        for epoch in range(epoch_start + 1, config.nr_epoch + 1):
            for _ in range(config.train_size//config.batch_size):
                global_cnt += 1
                images, labels = d.one_batch_train().__next__()
                feed_dict = {
                    placeholders['data']: images,
                    placeholders['label']: labels,
                    global_steps: global_cnt,
                    placeholders['is_training']: True,
                }
                _, _,  loss_v, loss_reg_v, auc_v, lr_v, summary = sess.run([auc_op, train, loss, loss_reg,
                                                                        auc_estimate, lr, merged],
                                                                       feed_dict=feed_dict)
                if global_cnt % config.show_interval == 0:
                    train_writer.add_summary(summary, global_cnt)
                    print(
                        "e:{},{}/{}".format(epoch, (global_cnt % config.train_size) // config.batch_size,
                                            config.train_size//config.batch_size),
                        'loss: {:.3f}'.format(loss_v),
                        'loss_reg: {:.3f}'.format(loss_reg_v),
                        'AUC: {:.3f}'.format(auc_v),
                        'lr: {:.3f}'.format(lr_v),
                    )

            ## validation
            if epoch % config.test_interval == 0:
                loss_sum = 0
                acc_sum = 0
                for i in range(config.val_size//config.batch_size):
                    images, labels = d.one_batch_val().__next__()
                    feed_dict = {
                        placeholders['data']: images,
                        placeholders['label']: labels,
                        global_steps: global_cnt,
                        placeholders['is_training']: False
                    }
                    loss_v, auc_v, summary = sess.run([loss, auc_estimate, merged],
                                                      feed_dict=feed_dict)
                    loss_sum += loss_v
                    acc_sum += auc_v
                test_writer.add_summary(summary, global_cnt)
                print("\n**************Validation results****************")
                print('loss_avg: {:.3f}'.format(loss_sum / (config.val_size//config.batch_size)),
                      'AUC_avg: {:.3f}'.format(acc_sum / (config.val_size//config.batch_size)))
                print("************************************************\n")

            ## save model
            if epoch % config.snapshot_interval == 0:
                saver.save(sess, os.path.join(config.log_model_dir, 'epoch-{}'.format(epoch)),
                           global_step=global_cnt)

        print('Training is done, exit.')
        prediction_out = np.array(sess.run(preds, feed_dict={placeholders['data']: d.test, global_steps: global_cnt,
                                        placeholders['is_training']: True}))[:, 1]
        pd.DataFrame({'ID': pd.read_csv(config.test_path)['ID'], 'Pred': prediction_out}).to_csv(os.path.join(config.submission_path, "{}.csv".format("textCNN_1")), index=False)
        print('Prediction is done, out')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os._exit(1)
