from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from feature_selection import *
from sklearn.model_selection import train_test_split
from data.scikit_feature_dataset import *
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef,roc_auc_score
import os
import time

def knn(X,y):

    k=10
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for selection_func in [("mrmr",mrmr_fs),("relieff",relief_fs), ("comsvm-frefe",com_svmfrfe_fs), ("fdr",selectfdr_fs), ("gs-svm",ga_svm_fs),("fre",rfe_fs)]:

            print(selection_func[0])
            start_fs_time = time.time()
            mask = selection_func[1](X,y,k)
            end_fs_time = time.time()

            neigh = KNeighborsClassifier(n_neighbors=3)
            # X_train = X_train[:,mask]
            # X_test = X_test[:,mask]

            start_fit_time = time.time()
            neigh.fit(X_train, y_train)
            end_fit_time = time.time()

            start_predict_time = time.time()
            pred = neigh.predict_proba(X_test)
            end_predict_time = time.time()

            print(f"Score {neigh.score(X_test,y_test)}")
            print(f"MCC: {matthews_corrcoef(y_test,neigh.predict(X_test))}")
            # print(f"Auc: {roc_auc_score(y_test,pred)}")
            print(f"FS time: {end_fs_time - start_fs_time}")
            print(f"Fit time: {end_fit_time - start_fit_time}")
            print(f"Predict time: {end_predict_time - start_predict_time}")



if __name__ == "__main__":
    X,y =read_basehock(os.path.join('data','scikit_feature_dataset','BASEHOCK.mat'))
    y = y[:,0]-1
    knn(X,y)
