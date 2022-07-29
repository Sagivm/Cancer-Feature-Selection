import pandas as pd
from mrmr import mrmr_classif
from sklearn.datasets import make_classification
from data.spectf.read_data import *
import numpy as np
from util.util import *

def mrmr_fs(X,y,k):
    # working dir has been up

    X_df = pd.DataFrame(X, columns = list(range(X.shape[1])))
    y_df = pd.Series(y)


    # use mrmr classification
    selected_features = mrmr_classif(X_df, y_df, K = k)
    return get_active_features(X.shape[1],selected_features)

if __name__ == "__main__":
    X_train, y_train = read_train()
    X_test, y_test = read_test()
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    print(mrmr_fs(X, y,10))