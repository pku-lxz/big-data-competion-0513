import pandas as pd
from common import config
import re
import numpy as np
from gensim.models import Word2Vec

class Dataset:
    def __init__(self, read_model=True):
        self.pattern = re.compile(r'(['u'\U0001F300-\U0001F64F'u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55])', re.UNICODE)
        self.raw_data = self.data_preprocess(config.train_path)
        self.train_y = np.array([{"Negative": 0, "Positive": 1}[label] for label in self.raw_data["label"].values])
        self.train_X = self.participle([sentence for sentence in self.raw_data["review"].values])
        self.test_X = self.participle([re.split(self.pattern, sentence) for sentence in pd.read_csv(config.test_path)["review"].values])
        if not read_model:
            self.word_vocab(config.model_path)
        self.model = Word2Vec.load(config.model_path)
        self.train = self.padding_sentences([self.convert_data(x, self.model) for x in self.train_X], 25)
        self.test = self.padding_sentences([self.convert_data(x, self.model) for x in self.test_X], 25)

    def data_preprocess(self, path):
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
        participle = [re.split(self.pattern, review) for review in df["review"]]
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
        paddles = np.zeros((len(sentences), length, config.embeddingsize))
        for i, sentence in enumerate(sentences):
            if length > sentence.shape[0]:
                paddle = np.concatenate((sentence, np.zeros((length-sentence.shape[0], config.embeddingsize))), axis=0)
                paddles[i, :, :] = paddle
            elif length < sentence.shape[0]:
                paddle = sentence[:length]
                paddles[i, :, :] = paddle
            else:
                paddles[i, :, :] = sentence
        return paddles

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

    def word_vocab(self, path):
        model = Word2Vec(self.train_X + self.test_X, iter=config.word2vec_inter, size=config.embeddingsize, workers=4)
        model.save(path)


if __name__ == "__main__":
    t = Dataset()
