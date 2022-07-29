import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from data.spectf.read_data import *
import sklearn_relief as relief
from ReliefF import ReliefF
from sklearn.preprocessing import StandardScaler
from util.util import *

def relief_fs(X,y,k):
    r = ReliefF(n_neighbors=20)
    r.fit(X,y)
    top_features = r.top_features[:k]
    return get_active_features(X.shape[1],top_features)

if __name__ == "__main__":
    X_train, y_train = read_train()
    X_test, y_test = read_test()
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    print(relief_fs(X, y,10))