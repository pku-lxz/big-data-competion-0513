import pandas as pd
from common import config
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse.csr import csr_matrix
import scipy.sparse as sp
from gensim.models import Word2Vec
import re
import pickle
from sklearn.decomposition import TruncatedSVD


class Dataset:
    def __init__(self, read_model=True, read_data=True):
        self.pattern = re.compile(r'(['u'\U0001F300-\U0001F64F'u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55])', re.UNICODE)
        self.raw_data0, self.raw_data1 = self.data_preprocess(config.train_path)
        self.y = np.array([{"Negative": 0, "Positive": 1}[label] for label in self.raw_data0["label"]])
        if read_data:
            with open(config.data_classic_train_path, 'rb') as file: self.X = pickle.load(file)
            with open(config.data_classic_test_path, 'rb') as file: self.test = pickle.load(file)
        else:
            corpus = self.convert_tfidf(
                [sentence.lower() for sentence in self.raw_data0['review']] + [sentence.lower() for sentence in
                                                                               pd.read_csv(config.test_path)[
                                                                                   'review'].values])
            tr, te = self.word2vec_data(config.word2vecmodel_path, read_model)
            self.X = np.concatenate([corpus[:config.train_size + config.val_size], tr], axis=1)
            self.test = np.concatenate([corpus[-config.test_size - 1:-1], te], axis=1)
            with open(config.data_classic_train_path, 'wb') as file: pickle.dump(self.X, file)
            with open(config.data_classic_test_path, 'wb') as file: pickle.dump(self.test, file)

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
    def convert_tfidf(corpus):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        np_feature_eng = sp.hstack([X, csr_matrix.max(X, axis=1)])
        np_feature_eng = sp.hstack([np_feature_eng, csr_matrix.min(X, axis=1)])
        np_feature_eng = sp.hstack([np_feature_eng, csr_matrix.mean(X, axis=1)])
        print('Start training PCA...')
        pca = TruncatedSVD(n_components=config.components)
        pca.fit(np_feature_eng)
        print('PCA DONE!')
        np_feature_eng = pca.transform(np_feature_eng).toarray()
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
            new_sentences.append([char for char in new_sentence if char not in ['      ', '… ', '"', '.', ' ', '" ', '."', ',"', ', ', ' , ', ',', '. ', '.  ', '"""', '"" ']])
        return new_sentences

    def word2vec_data(self, path, read):
        train_X = self.participle([sentence for sentence in self.raw_data1])
        test_X = self.participle(
            [re.split(self.pattern, sentence.lower()) for sentence in pd.read_csv(config.test_path)["review"].values])
        if not read:
            model = Word2Vec(train_X + test_X, size=config.embeddingsize, workers=4, iter=config.word2vec_inter)
            model.save(path)
        else:
            model = Word2Vec.load(config.word2vecmodel_path)
        tr = self.padding_sentences([self.convert_data(x, model) for x in train_X])
        te = self.padding_sentences([self.convert_data(x, model) for x in test_X])
        return tr, te

    @staticmethod
    def convert_data(sentence, vocab_dic):
        data = np.zeros((len(sentence), config.embeddingsize))
        for idx, word in enumerate(sentence):
            if vocab_dic.wv.__contains__(word):
                data[idx, :] = vocab_dic.wv.__getitem__(word)
        return data

    @staticmethod
    def padding_sentences(sentences):
        paddles = np.zeros((len(sentences), config.padding_size * config.embeddingsize))
        for i, sentence in enumerate(sentences):
            if config.padding_size > sentence.shape[0]:
                paddle = np.concatenate((sentence, np.zeros((config.padding_size - sentence.shape[0], config.embeddingsize))),
                                        axis=0)
                paddles[i, :] = paddle.reshape((1, -1))
            elif config.padding_size < sentence.shape[0]:
                paddle = sentence[:config.padding_size]
                paddles[i, :] = paddle.reshape((1, -1))
            else:
                paddles[i, :] = sentence.reshape((1, -1))
        return paddles


if __name__ == "__main__":
    t = Dataset()
