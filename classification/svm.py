from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import LabelEncoder
from util.util import *
from feature_selection import *
from sklearn.model_selection import train_test_split
from data.scikit_feature_dataset import *
from data.microarray import *
from data.biocon import *
from data.benchmark import *
from sklearn.model_selection import StratifiedKFold,LeaveOneOut
from sklearn.metrics import matthews_corrcoef,roc_auc_score
import os
import time

def svm(X,y,k):

    #k=20
    kf = StratifiedKFold(n_splits=5)
    loo = LeaveOneOut()

    for train_index, test_index in loo.split(X,y):

        for selection_func in [("X", com_esvmfrfe_fs)]:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print(selection_func[0])
            start_fs_time = time.time()
            mask = selection_func[1](X,y,k)
            end_fs_time = time.time()

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

            print(f"Score {cls.score(X_test,y_test)}")
            print(f"MCC: {matthews_corrcoef(y_test,cls.predict(X_test))}")
            # print(f"Auc: {roc_auc_score(y_test,pred,multi_class='ovr')}")
            print(f"FS time: {end_fs_time - start_fs_time}")
            print(f"Fit time: {end_fit_time - start_fit_time}")
            print(f"Predict time: {end_predict_time - start_predict_time}")
            print("\n")
            f=str_features(mask)
            if(selection_func[0] == "ga-svm" or selection_func[0] == "fdr"):
                    f=""
            with open("x.csv","a") as file:
                file.write("\n".join([
                    str(cls.score(X_test,y_test))+","+f,
                    str(matthews_corrcoef(y_test,cls.predict(X_test))) + "," + f,
                    # str(roc_auc_score(y_test,pred,multi_class='ovr')) + "," + f,
                    str(end_fs_time - start_fs_time)+","+f,
                    str(end_fit_time - start_fit_time)+","+f,
                    str(end_predict_time - end_fit_time)+","+f,
                    # f
                ]))
                file.write("\n")
        return



if __name__ == "__main__":
    X, y = read_leukemia()
    lab = LabelEncoder  ()
    y = lab.fit_transform(y)
    svm(X, y, 5)
    svm(X, y, 10)
    svm(X, y, 20)
    svm(X, y, 50)

    X, y = read_lung_small()
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

    X, y = read_sorile()
    lab = LabelEncoder()
    y = lab.fit_transform(y)
    svm(X, y, 5)
    svm(X, y, 10)
    svm(X, y, 20)
    svm(X, y, 50)