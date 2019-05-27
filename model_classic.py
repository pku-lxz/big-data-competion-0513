import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import numpy as np
from data_tfidf import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB

class model:

    def __init__(self, X, y, test, K=5):
        self.X, self.y, self.test, self.predict_y = X, y, test, np.zeros(test.shape[0])
        self.K = K
        self.skf = StratifiedKFold(n_splits=K, shuffle=True)

    def _lgb(self):
        lgbparam = {"objective": "binary", "metric": "auc", "boosting": 'gbdt',
                    "max_depth": -1,
                    "num_leaves": 30,
                    "learning_rate": 0.1,
                    "bagging_freq": 5,
                    "bagging_fraction": 0.8,
                    "feature_fraction": 1,
                    "min_data_in_leaf": 3,
                    "tree_learner": "serial",
                    "boost_from_average": "false",
                    "verbosity": 1}
        for fold, (trn_idx, val_idx) in enumerate(self.skf.split(self.X, self.y)):
            X_train, y_train = self.X[trn_idx, :], self.y[trn_idx]
            X_val, y_val = self.X[val_idx, :], self.y[val_idx]

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            print('Start training...')
            model = lgb.train(lgbparam, dtrain, num_boost_round=50000, valid_sets=dval, early_stopping_rounds=500,
                              verbose_eval=2500)

            print('Training done, exit. Start predicting...')
            self.predict_y += model.predict(self.test)[:, 1]/self.K

    def _lr(self):
        for fold, (trn_idx, val_idx) in enumerate(self.skf.split(self.X, self.y)):
            X_train, y_train = self.X[trn_idx, :], self.y[trn_idx]
            X_val, y_val = self.X[val_idx, :], self.y[val_idx]

            model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', C=0.1, max_iter=100000)
            print('Start training Logistic regression...')
            model.fit(X_train, y_train)
            print("\n**************Validation results****************")
            print('AUC_avg: {:.3f}'.format(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])))
            print("************************************************\n")
            print('Training done, exit. Start predicting...')
            self.predict_y += model.predict_proba(self.test)[:, 1] / self.K

    def _nb(self):
        for fold, (trn_idx, val_idx) in enumerate(self.skf.split(self.X, self.y)):
            X_train, y_train = self.X[trn_idx, :], self.y[trn_idx]
            X_val, y_val = self.X[val_idx, :], self.y[val_idx]

            model = GaussianNB()
            print('Start training naive bayes...')
            model.fit(X_train, y_train)
            print("\n**************Validation results****************")
            print('AUC_avg: {:.3f}'.format(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])))
            print("************************************************\n")
            print('Training done, exit. Start predicting...')
            self.predict_y += model.predict_proba(self.test)[:, 1] / self.K


if __name__ == "__main__":
    data = Dataset()
    mlmodel = model(data.X, data.y, data.test)
    mlmodel._lr()

