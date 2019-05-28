import os


class config:

    path = "/Users/lamprad/Desktop/DATA/big-data-competition/"
    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")
    word2vecmodel_path = os.path.join(path, "model")
    data_classic_train_path = os.path.join(path, "train_classic.pickle")
    data_classic_test_path = os.path.join(path, "test_classic.pickle")
    submission_path = os.path.join(path, "submission/")
    log_dir = os.path.join(path, 'train_log')

    ensemble_path = os.path.join(submission_path, 'ensemble.csv')
    '''where to write model snapshots to'''
    log_model_dir = os.path.join(log_dir, 'models')

    embeddingsize = 200
    word2vec_inter = 80000
    padding_size = 25
    components = 6000

    train_size = 5728
    val_size = 600
    test_size = 2712
    batch_size = 128
    weight_decay = 0.2

    shape = (padding_size, embeddingsize)
    nr_class = 2
    nr_epoch = 60

    vocab_size = 18144
    lr = 0.001
    show_interval = 30
    test_interval = 5
    snapshot_interval = 5
