import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import numpy as np
from data_tfidf import Dataset
import pandas as pd
from common import config
import os


class modelLightGBM:

    def __init__(self, lr=0.001, leave=8):
        self.param = {
            "objective": "binary",
            "metric": "auc",
            "boosting": 'gbdt',
            "max_depth": -1,
            "num_leaves": leave,
            "learning_rate": lr,
            "bagging_freq": 5,
            "bagging_fraction": 0.4,
            "feature_fraction": 0.5,
            "min_data_in_leaf": 80,
            "min_sum_heassian_in_leaf": 10,
            "tree_learner": "serial",
            "boost_from_average": "false",
            "verbosity": 1,
        }

    def train(self, X, y, X_test, kfold=5):
        skf = StratifiedKFold(n_splits=kfold, shuffle=True)
        predict_y = np.zeros(X_test.shape[0])
        for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X[trn_idx, :], y[trn_idx]
            X_val, y_val = X[val_idx, :], y[val_idx]

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            print('Start training...')
            gbm = lgb.train(self.param,
                            dtrain,
                            num_boost_round=50000,
                            valid_sets=dval,
                            early_stopping_rounds=500,
                            verbose_eval=2500)

            print('Start predicting...')
            predict_y += gbm.predict(X_test)/kfold
        return predict_y


if __name__ == "__main__":
    data = Dataset()
    print("#########LightGBM Prediction start!#########")
    gbm = modelLightGBM()
    gbm.train(data.X, data.y, data.test)
