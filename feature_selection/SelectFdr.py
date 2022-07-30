import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFdr
from mrmr import mrmr_classif
from sklearn.metrics import roc_auc_score
from data.spectf.read_data import *

def selectfdr_fs(X,y,k):
    # working dir has been up
    cls = SelectFdr(alpha=0.5)
    cls.feature_names_in_ = list(range(X.shape[1]))
    cls.fit(X,y)
    return cls.get_support().astype(int)


if __name__ == "__main__":
    X_train, y_train = read_train()
    X_test, y_test = read_test()
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    print(selectfdr_fs(X, y))