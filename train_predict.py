from model_lgbm import modelLightGBM
from data_word2vec import Dataset as dw2v
import os
import pandas as pd
from common import config
from model_cnn import TextCNN
import tensorflow as tf


class PredictProbilityOut:

    def __init__(self):
        self.probility = 0
        self.count = 0

    def update(self, y, alpha=1):
        self.count += alpha
        self.probility += y

    def subPredict(self, df_test, path):
        sub_df = pd.DataFrame({'ID': df_test['ID'], 'Pred': self.probility / self.count})
        sub_df.to_csv(path, index=True)


def main():
    pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os._exit(1)
