import pandas as pd
from common import config
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class Dataset:
    def __init__(self):
        self.raw_data = self.data_preprocess(config.train_path)
        corpus = [sentence.lower() for sentence in self.raw_data['review']] + [sentence.lower() for sentence in pd.read_csv(config.test_path)['review'].values]
        self.corpus = self.convert_(corpus).toarray()
        self.X, self.y, self.test = self.corpus[:config.train_size + config.val_size], np.array([{"Negative": 0, "Positive": 1}[label] for label in self.raw_data["label"]]), self.corpus[-config.test_size - 1:-1]
        pass

    def data_preprocess(self, path):
        with open(path) as f:
            lines = f.readlines()
            idx = 1
            while idx < len(lines):
                if str(idx) == lines[idx][0:len(str(idx))]:
                    idx += 1
                else:
                    while str(idx) != lines[idx][0:len(str(idx))]:
                        lines[idx - 1] = lines[idx - 1] + lines[idx]
                        lines.pop(idx)
                        if lines[idx].__len__() < len(str(idx)):
                            lines[idx - 1] = lines[idx - 1] + lines[idx]
                            lines.pop(idx)
                    idx += 1
        df = {"ID": [line.split(",", 1)[0] for line in lines[1:]],
              "review": [line.split(",", 1)[1][:-10].lower() for line in lines[1:]],
              "label": [line.split(",", 1)[1][-9:-1] for line in lines[1:]]}
        return df


    @staticmethod
    def convert_(corpus):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus).toarray()
        np_feature_eng = np.concatenate([X, np.vstack(X.mean(axis=1))], axis=1)
        np_feature_eng = np.concatenate([np_feature_eng, np.vstack(np_feature_eng.std(axis=1))], axis=1)
        np_feature_eng = np.concatenate([np_feature_eng, np.square(np_feature_eng)], axis=1)
        np_feature_eng = np.concatenate([np_feature_eng, np.sinh(np_feature_eng[:, :200])], axis=1)

        np_feature_eng = np.concatenate([np_feature_eng, np.vstack(np.max(np_feature_eng[:, :21869], axis=1))], axis=1)

        np_feature_eng = np.concatenate([np_feature_eng, np.vstack(np.min(np_feature_eng[:, :21869], axis=1))], axis=1)

        np_feature_eng = np.concatenate([np_feature_eng, np.vstack(np.percentile(np_feature_eng, 25, axis=1))], axis=1)

        return np_feature_eng


if __name__ == "__main__":
    t = Dataset()
