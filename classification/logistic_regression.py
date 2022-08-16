# from sklearnex import patch_sklearn
# patch_sklearn()
import numpy as np
import os
import time
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold ,KFold
from sklearn.metrics import matthews_corrcoef,roc_auc_score,average_precision_score
from util.util import *
from feature_selection import *
from sklearn.model_selection import train_test_split
from data.scikit_feature_dataset import *
from data.microarray import *
from data.biocon import *
from data.benchmark import *
from keras.utils import to_categorical


def logic(X,y,k,activate_aug= False):
    """
        Activate logistic regression on X and y with preserving k features, activate aug if necessary
        :param X:
        :param y:
        :param k:
        :param activate_aug:
        :return:
        """
    kf = StratifiedKFold(n_splits=10)
    loo = KFold(len(X))
    n_classes = len(np.unique(y))

    for selection_func in [("wu", com_esvmfrfe_fs)]:
        scores = list()

        print(selection_func[0])
        start_fs_time = time.time()
        mask = selection_func[1](X, y, k)
        end_fs_time = time.time()

        for train_index, test_index in kf.split(X,y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            cls = LogisticRegression()
            mask = np.array(mask).astype(bool)
            X_train = X_train[:,mask]
            X_test = X_test[:,mask]

            # PCA
            if activate_aug:
                linear_transformer = KernelPCA(kernel='linear')
                rbf_transformer = KernelPCA(kernel='rbf')
                X_train = rbf_transformer.fit_transform(linear_transformer.fit_transform(X_train))
                X_test = rbf_transformer.transform(linear_transformer.transform(X_test))
                oversample = BorderlineSMOTE()
                X_train_gen, y_train_gen = oversample.fit_resample(X_train, y_train)
                y_train = np.hstack((y_train, y_train_gen))
                X_train = np.vstack((X_train, X_train_gen))
                print(X_train.shape)


            start_fit_time = time.time()
            cls.fit(X_train, y_train)
            end_fit_time = time.time()

            start_predict_time = time.time()
            pred = cls.predict(X_test)
            end_predict_time = time.time()

            print(f"Score {cls.score(X_test, y_test)}")
            print(f"MCC: {matthews_corrcoef(y_test, cls.predict(X_test))}")
            print(f"Auc: {roc_auc_score(to_categorical(y_test,n_classes), to_categorical(pred), multi_class='ovr')}")
            print(f"pr-Auc: {average_precision_score(to_categorical(y_test, n_classes), to_categorical(pred))}")
            print(f"FS time: {end_fs_time - start_fs_time}")
            print(f"Fit time: {end_fit_time - start_fit_time}")
            print(f"Predict time: {end_predict_time - start_predict_time}")
            print(f"{str_features(mask)}")

if __name__ == "__main__":

    X, y = read_wu()
    lab = LabelEncoder()
    y = lab.fit_transform(y)
    logic(X, y, 20,activate_aug=True)
