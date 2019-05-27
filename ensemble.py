import os
import pandas as pd
from common import config
import numpy as np


class PredictProbilityOut:

    def __init__(self):
        self.probility = 0
        self.count = 0

    def update(self, file_name, alpha=1.0):
        self.count += alpha
        y = pd.read_csv(config.submission_path+file_name)['Pred'].values
        if len(np.array(y).shape) > 1:
            self.probility += alpha * np.array(y)[:, 1]
        else:
            self.probility += alpha * np.array(y)

    def subPredict(self, df_test, path):
        sub_df = pd.DataFrame({'ID': df_test['ID'], 'Pred': self.probility / self.count})
        sub_df.to_csv(path, index=False)


def main():
    p = PredictProbilityOut()
    p.update("textCNN_1.csv")
    p.update("textCNN_2.csv")
    p.update("textCNN_3.csv")
    p.update("bilstm_1.csv")
    p.update("bilstm_2.csv")
    p.update("bilstm_3.csv")
    p.update("bilstm_4.csv")
    p.update("bilstm_5.csv")
    p.update("bilstm_6.csv")
    p.subPredict(pd.read_csv(config.test_path), config.ensemble_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os._exit(1)
