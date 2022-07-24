import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFdr
from mrmr import mrmr_classif

from data.spectf.read_data import *

def main():
    # working dir has been up
    X_train, y_train = read_train()
    X_test, y_test = read_test()

    cls = SelectFdr(alpha=0.1)
    cls.feature_names_in_ = list(range(X_train.shape[1]))
    X_train = cls.fit_transform(X_train,y_train)
    print(X_train.shape)
    print(cls.get_feature_names_out())


if __name__ == "__main__":
    main()