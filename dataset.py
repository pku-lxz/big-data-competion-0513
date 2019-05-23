import pandas as pd
from common import config
import re
import numpy as np
from gensim.models import Word2Vec


class Dataset:
    def __init__(self, read_model=True):
        self.raw_data = self.data_preprocess(config.path+"train.csv")
        self.train_y = np.array([{"Negative": 0, "Positive": 1}[label] for label in self.raw_data["label"].values])
        self.train_X = [sentence for sentence in self.raw_data["review"].values]
        self.test_X = [re.split(r'\W+', sentence) for sentence in pd.read_csv(config.path+"test.csv")["review"].values]
        if not read_model:
            self.word_vocab(config.model_path)
        self.model = Word2Vec.load(config.model_path)
        self.train = self.padding_sentences([self.convert_data(x, self.model) for x in self.train_X], 25)
        self.test = self.padding_sentences([self.convert_data(x, self.model) for x in self.test_X], 25)

    @staticmethod
    def data_preprocess(path):
        with open(path) as f:
            lines = f.readlines()
            idx = 1
            while idx < len(lines):
                if str(idx) == lines[idx][0:len(str(idx))]:
                    idx += 1
                else:
                    while str(idx) != lines[idx][0:len(str(idx))]:
                        lines[idx - 1] = lines[idx-1] + lines[idx]
                        lines.pop(idx)
                        if lines[idx].__len__() < len(str(idx)):
                            lines[idx - 1] = lines[idx - 1] + lines[idx]
                            lines.pop(idx)
                    idx += 1
        df = {"ID": [line.split(",", 1)[0] for line in lines[1:]],
              "review": [line.split(",", 1)[1][:-10] for line in lines[1:]],
              "label": [line.split(",", 1)[1][-9:-1] for line in lines[1:]]}
        p = re.compile(r'\W+')
        participle = [re.split(p, review) for review in df["review"]]
        return pd.DataFrame({"ID": df["ID"], "review": participle, "label": df["label"]})

    @staticmethod
    def convert_data(sentence, vocab_dic):
        data = np.zeros((len(sentence), config.embeddingsize))
        for idx, word in enumerate(sentence):
            if word in vocab_dic:
                data[idx, :] = vocab_dic[word]
        return data

    @staticmethod
    def padding_sentences(sentences, length):
        paddles = np.zeros((len(sentences), length * config.embeddingsize))
        for i, sentence in enumerate(sentences):
            if length > sentence.shape[0]:
                paddle = np.concatenate((sentence, np.zeros((length-sentence.shape[0], config.embeddingsize))), axis=0)
                paddles[i] = np.reshape(paddle, (1, -1))
            elif length < sentence.shape[0]:
                paddle = sentence[:length]
                paddles[i] = np.reshape(paddle, (1, -1))
            else:
                paddles[i] = np.reshape(sentence, (1, -1))
        return paddles

    def word_vocab(self, path):
        model = Word2Vec(list(self.raw_data["review"].values) + self.test_X, iter=config.word2vecinter, size=config.embeddingsize, workers=4)
        model.save(path)


if __name__ == "__main__":
    t = Dataset()


