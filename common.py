import os


class config:

    path = "/Users/lamprad/Desktop/DATA/big-data-competition/"
    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")
    model_path = os.path.join(path, "model")
    submission_path = os.path.join(path, "submission/")
    log_dir = os.path.join(path, 'train_log')

    '''where to write model snapshots to'''
    log_model_dir = os.path.join(log_dir, 'models')

    embeddingsize = 120
    word2vec_inter = 20000
    padding_size = 31

    train_size = 6328
    test_size = 2712
    batch_size = 64
    epochs = 60
    weight_decay = 0.2

    shape = (padding_size, embeddingsize)
    nr_class = 2
    nr_epoch = 20

    vocab_size = 21645
    lr = 0.01
    show_interval = 50
    test_interval = 100
    snapshot_interval = 5