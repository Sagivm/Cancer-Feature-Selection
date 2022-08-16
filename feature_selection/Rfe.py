import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from data.spectf.read_data import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from util.util import *

def rfe_fs(X,y,k):
    """
    Activate RFE on X,y to preserve k features
    :param X:
    :param y:
    :param k:
    :return:
    """

    estimator = SVC(kernel="linear")
    selector = RFE(estimator,n_features_to_select=k)
    selector.fit(X,y)
    features = list(map(lambda x: int(x[1:]),selector.get_feature_names_out()))
    return  get_active_features(X.shape[1],features)


if __name__ == "__main__":
    X_train, y_train = read_train()
    X_test, y_test = read_test()
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    print(rfe_fs(X, y,10))