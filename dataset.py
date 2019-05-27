import pandas as pd
from common import config
import re
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(self):
        self.pattern = re.compile(r'(['u'\U0001F300-\U0001F64F'u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55])', re.UNICODE)
        self.raw_data = self.data_preprocess(config.train_path)
        self.dic = self.vocab_dict()
        self.test = self.padding_sentences([self.convert_data(x, self.dic) for x in self.participle([re.split(self.pattern, sentence.lower()) for sentence in pd.read_csv(config.test_path)["review"].values])], config.padding_size)
        self.X, self.x_val, self.y, self.y_val = train_test_split(
                        self.padding_sentences([self.convert_data(x, self.dic) for x in self.participle([sentence for sentence in self.raw_data["review"].values])], config.padding_size),
                        np.array([{"Negative": 0, "Positive": 1}[label] for label in self.raw_data["label"].values]),
                        test_size=config.val_size)

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
              "review": [line.split(",", 1)[1][:-10].lower() for line in lines[1:]],
              "label": [line.split(",", 1)[1][-9:-1] for line in lines[1:]]}
        participle = [re.split(self.pattern, review) for review in df["review"]]
        return pd.DataFrame({"ID": df["ID"], "review": participle, "label": df["label"]})

    def vocab_dict(self):
        vocab = set([word for sentence in self.participle([sentence for sentence in self.raw_data["review"].values]) for word in sentence])
        vocab_dict = dict(zip(vocab, range(2, len(vocab), 1)))
        vocab_dict["< PAD >"], vocab_dict["< UNK >"] = 0, 1
        return vocab_dict

    @staticmethod
    def convert_data(sentence, vocab_dic):
        data = np.zeros((len(sentence)))
        for idx, word in enumerate(sentence):
            if word in vocab_dic:
                data[idx] = vocab_dic[word]
            else:
                data[idx] = vocab_dic["< UNK >"]
        return data

    @staticmethod
    def padding_sentences(sentences, length):
        paddles = np.zeros((len(sentences), length))
        for i, sentence in enumerate(sentences):
            if length > sentence.shape[0]:
                paddle = np.concatenate((sentence, np.zeros((length - sentence.shape[0]))), axis=0)
                paddles[i, :] = paddle
            elif length < sentence.shape[0]:
                paddle = sentence[:length]
                paddles[i, :] = paddle
            else:
                paddles[i, :] = sentence
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

    def one_batch_train(self, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        num_batches_per_epoch = int((config.train_size - 1) / config.batch_size) + 1
        for epoch in range(config.nr_epoch):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(config.train_size))
                shuffled_data_X, shuffled_data_y = self.X[shuffle_indices, :], self.y[shuffle_indices]
            else:
                shuffled_data_X, shuffled_data_y = self.X, self.y
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * config.batch_size
                end_index = min((batch_num + 1) * config.batch_size, config.train_size)
                yield shuffled_data_X[start_index:end_index], shuffled_data_y[start_index:end_index]

    def one_batch_val(self, shuffle=False):
        """
        Generates a batch iterator for a dataset.
        """
        num_batches_per_epoch = int((config.val_size - 1) / config.batch_size) + 1
        for epoch in range(config.nr_epoch):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(config.train_size))
                shuffled_data_X, shuffled_data_y = self.X[shuffle_indices, :], self.y[shuffle_indices]
            else:
                shuffled_data_X, shuffled_data_y = self.X, self.y
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * config.batch_size
                end_index = min((batch_num + 1) * config.batch_size, config.train_size)
                yield shuffled_data_X[start_index:end_index], shuffled_data_y[start_index:end_index]


if __name__ == "__main__":
    t = Dataset()
