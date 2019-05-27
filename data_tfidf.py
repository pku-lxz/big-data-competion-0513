import pandas as pd
from common import config
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse.csr import csr_matrix
import scipy.sparse as sp
from gensim.models import Word2Vec
import re


class Dataset:
    def __init__(self, read_model=True):
        self.pattern = re.compile(r'(['u'\U0001F300-\U0001F64F'u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55])', re.UNICODE)
        self.raw_data0, self.raw_data1 = self.data_preprocess(config.train_path)
        corpus = self.convert_([sentence.lower() for sentence in self.raw_data0['review']] + [sentence.lower() for sentence in pd.read_csv(config.test_path)['review'].values])
        self.y = np.array([{"Negative": 0, "Positive": 1}[label] for label in self.raw_data0["label"]])
        tr, te = self.word2vec_data(config.model_path, read_model)
        self.X = np.concatenate([corpus[:config.train_size + config.val_size], tr], axis=1)
        self.test = np.concatenate([corpus[-config.test_size - 1:-1], te], axis=1)

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
        participle = [re.split(self.pattern, review) for review in df["review"]]
        return df, participle

    @staticmethod
    def convert_(corpus):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        np_feature_eng = sp.hstack([X, csr_matrix.max(X, axis=1)])
        np_feature_eng = sp.hstack([np_feature_eng, csr_matrix.min(X, axis=1)])
        np_feature_eng = sp.hstack([np_feature_eng, csr_matrix.mean(X, axis=1)]).toarray()
        np_feature_eng = np.concatenate([np_feature_eng, np.square(np_feature_eng)], axis=1)
        np_feature_eng = np.concatenate([np_feature_eng, np.vstack(np_feature_eng.std(axis=1))], axis=1)
        np_feature_eng = np.concatenate([np_feature_eng, np.vstack(np.percentile(np_feature_eng, 25, axis=1))], axis=1)
        return np_feature_eng

    @staticmethod
    def participle(sentences):
        new_sentences = []
        for sentence in sentences:
            new_sentence = []
            for word in sentence:
                if len(word) > 1:
                    new_sentence.extend([char for char in re.split(r'(\w+)', word) if char])
                elif word:
                    new_sentence.append(word)
            new_sentences.append([char for char in new_sentence if char != " "])
        return new_sentences

    def word2vec_data(self, path, read):
        train_X = self.participle([sentence for sentence in self.raw_data1])
        test_X = self.participle(
            [re.split(self.pattern, sentence) for sentence in pd.read_csv(config.test_path)["review"].values])
        if not read:
            model = Word2Vec(train_X + test_X, size=config.embeddingsize, workers=4)
            model.save(path)
        else:
            model = Word2Vec.load(config.model_path)
        tr = self.padding_sentences([self.convert_data(x, model) for x in train_X], config.padding_size)
        te = self.padding_sentences([self.convert_data(x, model) for x in test_X], config.padding_size)
        return tr, te

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
                paddle = np.concatenate((sentence, np.zeros((length - sentence.shape[0], config.embeddingsize))),
                                        axis=0)
                paddles[i, :] = paddle.reshape((1, -1))
            elif length < sentence.shape[0]:
                paddle = sentence[:length]
                paddles[i, :] = paddle.reshape((1, -1))
            else:
                paddles[i, :] = sentence.reshape((1, -1))
        return paddles


if __name__ == "__main__":
    t = Dataset()
