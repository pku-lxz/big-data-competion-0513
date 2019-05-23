from model_lgbm import modelLightGBM
from dataset import Dataset
import os

def main():
    data = Dataset()
    gbm = modelLightGBM()
    gbm.train(data.train, data.train_y)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os._exit(1)
