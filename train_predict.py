from model_lgbm import modelLightGBM
from data_word2vec import Dataset as dw2v
import os
import pandas as pd
from common import config
from model_cnn import CNNModel
import tensorflow as tf


class PredictProbilityOut:

    def __init__(self):
        self.probility = 0
        self.count = 0

    def update(self, y, alpha=1):
        self.count += alpha
        self.probility += y

    def subPredict(self, df_test):
        sub_df = pd.DataFrame({'ID': df_test['ID'], 'Pred': self.probility / self.count})
        sub_df.to_csv(config.submission_path, index=True)


def main():
    predict = PredictProbilityOut()
    data = dw2v(read_model=False)
    print("#########LightGBM Prediction start!#########")
    gbm = modelLightGBM()
    predict.update(gbm.train(data.train, data.train_y, data.test))
    # print("#########CNN Prediction start!#########")
    # cnn = CNNModel()
    # placeholders, label_onehot, logits = cnn.build()
    # loss = tf.losses.softmax_cross_entropy(label_onehot, logits)
    # ## train config
    # opt = tf.train.AdamOptimizer(0.01)
    # # in order to update BN in every iter
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     train = opt.minimize(loss)
    # with tf.Session() as sess:
    #     for epoch in range(epoch_start + 1, config.nr_epoch + 1):
    #         loss = sess.run(loss, feed_dict={placeholders["data"]: data.train,
    #                                      placeholders["label"]: tf.one_hot(data.train_y),
    #                                      placeholders['is_training']: True})
    #         if epoch % config.snapshot_interval == 0:
    #             saver.save(sess, os.path.join(config.log_model_dir, 'epoch-{}'.format(epoch)),
    #                    global_step=global_cnt)
    print("#########Prediction Done! Get submission file!#########")
    predict.subPredict(pd.read_csv(config.test_path))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os._exit(1)
