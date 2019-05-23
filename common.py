import os


class config:

    path = "/Users/lamprad/Desktop/DATA/big-data-competition/"
    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")
    model_path = os.path.join(path, "model")
    submission_path = os.path.join(path, "submission.csv")

    embeddingsize = 120
    word2vec_inter = 20000
    padding_size = 25

    weight_decay = 0.2

    shape = (padding_size, embeddingsize)
    nr_class = 2
    nr_epoch = 20
