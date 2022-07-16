import pandas as pd
import os

BASE_PATH = os.path.join("data","spectf")


def read_train():
    df = pd.read_csv(
        os.path.join(BASE_PATH, 'SPECTF.train'),
        header=None
    )
    data = df.to_numpy()
    return data[:,1:],data[:,1]


def read_test():
    df = pd.read_csv(
        os.path.join(BASE_PATH, 'SPECTF.test'),
        header=None
    )
    data = df.to_numpy()
    return data[:, 1:], data[:, 1]
