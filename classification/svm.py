from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.svm import SVC
import numpy as np
import math
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from util.util import *
from feature_selection import *
from sklearn.model_selection import train_test_split
from data.scikit_feature_dataset import *
from data.microarray import *
from data.biocon import *
from data.benchmark import *
from sklearn.model_selection import StratifiedKFold ,KFold
from sklearn.metrics import matthews_corrcoef,roc_auc_score,average_precision_score
import os
import time

def svm(X,y,k):

    #k=20
    kf = StratifiedKFold(n_splits=10)
    loo = KFold(len(X))
    n_classes = len(np.unique(y))
    for selection_func in [("mrmr", mrmr_fs), ("relief", relief_fs), ("select", selectfdr_fs), ("rfe", rfe_fs),
                           ("com-svm", com_svmfrfe_fs), ("ecom-svm", com_esvmfrfe_fs), ("ga-svm", ga_svm_fs)]:
        scores = list()

        print(selection_func[0])
        start_fs_time = time.time()
        mask = selection_func[1](X, y, k)
        end_fs_time = time.time()

        for train_index, test_index in loo.split(X,y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            cls = SVC()
            mask = np.array(mask).astype(bool)
            X_train = X_train[:,mask]
            X_test = X_test[:,mask]

            start_fit_time = time.time()
            cls.fit(X_train, y_train)
            end_fit_time = time.time()

            start_predict_time = time.time()
            pred = cls.predict(X_test)
            end_predict_time = time.time()

            scores.append(cls.score(X_test, y_test))
          #   print(f"Score {cls.score(X_test, y_test)}")
          #   print(f"MCC: {matthews_corrcoef(y_test, cls.predict(X_test))}")
          # #  print(f"Auc: {roc_auc_score(to_categorical(y_test,n_classes), to_categorical(pred), multi_class='ovr')}")
          #   print(f"pr-Auc: {average_precision_score(to_categorical(y_test, n_classes), to_categorical(pred))}")
          #   print(f"FS time: {end_fs_time - start_fs_time}")
          #   print(f"Fit time: {end_fit_time - start_fit_time}")
          #   print(f"Predict time: {end_predict_time - start_predict_time}")
          #   print("\n")
            f = str_features(mask)
            if (selection_func[0] == "ga-svm" or selection_func[0] == "fdr"):
                f = ""

        print(sum(scores)/len(scores))
        with open("svm_auc.csv", "a") as file:
            file.write("\n".join([
                #           str(roc_auc_score(to_categorical(y_test, n_classes), to_categorical(pred), multi_class='ovr')),
                #            str(average_precision_score(to_categorical(y_test, n_classes), to_categorical(pred)))
                str(sum(scores) / len(scores))
            ]))
            file.write("\n")


if __name__ == "__main__":
    #
    # X, y = read_leukemia()
    # lab = LabelEncoder()
    # y = lab.fit_transform(y)
    # svm(X, y, 5)
    # svm(X, y, 10)
    # svm(X, y, 20)
    # svm(X, y, 50)
    #
    #
    # X, y = read_lung_small()
    # lab = LabelEncoder()
    # y = lab.fit_transform(y)
    # svm(X, y, 5)
    # svm(X, y, 10)
    # svm(X, y, 20)
    # svm(X, y, 50)
    #
    # X, y = read_golub()
    # lab = LabelEncoder()
    # y = lab.fit_transform(y)
    # svm(X, y, 5)
    # svm(X, y, 10)
    # svm(X, y, 20)
    # svm(X, y, 50)
    #
    # X, y = read_ayest()
    # lab = LabelEncoder()
    # y = lab.fit_transform(y)
    # svm(X, y, 5)
    # svm(X, y, 10)
    # svm(X, y, 20)
    # svm(X, y, 50)

    X, y = read_sorile()
    lab = LabelEncoder()
    y = lab.fit_transform(y)
    svm(X, y, 5)
    svm(X, y, 10)
    svm(X, y, 20)
    svm(X, y, 50)


    X, y = read_submar()
    lab = LabelEncoder()
    y = lab.fit_transform(y)
    svm(X, y, 5)
    svm(X, y, 10)
    svm(X, y, 20)
    svm(X, y, 50)