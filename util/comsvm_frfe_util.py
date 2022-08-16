import numpy as np
from util.ga_svm_utils import *
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
def expand_top_features(X_train,y_train,X_test,y_test,K,best_H_features,n_candiadte_groups=100):
    """
    After aquiring the top H best features we want to re-add some old removed features by randomly sampling them
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param K:
    :param best_H_features:
    :param n_candiadte_groups:
    :return:
    """
    # get not featured values
    cls = SVC()
    removed_features = list(filter(lambda x: x not in best_H_features, list(range(44))))
    results = list()
    # base svc
    cls = SVC()
    cls.fit(X_train[:, best_H_features],y_train)
    score = cls.score(X_test[:, best_H_features], y_test)
    results.append((sorted(best_H_features), score))
    # expanded svc
    for _ in range(n_candiadte_groups):
        features = np.hstack((
            best_H_features, np.random.choice(removed_features, K)))
        cls = SVC()
        cls.fit(X_train[:, features], y_train)
        score = cls.score(X_test[:, features], y_test)
        results.append((sorted(features), score))
    return sorted(results, key=lambda x: x[1], reverse=True)[0]

def get_dy_i(X_train,y_train,y_test):

    # get class distance matrix
    classes = np.unique(np.hstack((y_train, y_test)))
    cls_centers = list()
    # calculate d_i

    # calculate class centers and distance
    for cls in classes:
        cls_X = X_train[y_train == cls]
        cls_center = np.average(cls_X, axis=0)
        cls_centers.append(cls_center)
    cls_centers = np.vstack(cls_centers)
    # calculate distances
    cls_dist_mat = np.ndarray(shape=(len(classes), len(classes)))
    for i in classes:
        for j in classes:
            cls_dist_mat[i, j] = np.linalg.norm(cls_centers[i] - cls_centers[j])

    # if len(classes) == 2:
    #     classes=[classes[0]]

    max_dist = np.max(cls_dist_mat)
    d_i = np.ndarray(shape=(len(classes), len(y_train)))
    y_i = np.ndarray(shape=(len(classes), len(y_train)))

    for k in classes:
        for i, sample_cls in enumerate(y_train):
            # d_i[k, i] = 1 if sample_cls == k else 1 + cls_dist_mat[k, sample_cls] / max_dist
            y_i[k, i] = -1 if sample_cls == k else 1

    return d_i,y_i


def get_H_features(X_train,y_i,H):
    """
    Get best H features applying FRFE
    :param X_train:
    :param y_i:
    :param H:
    :return:
    """
    n_features = list(range(X_train.shape[1]))
    while len(n_features) > 2 * H:
        k_features = list()
        for k in y_i:
            cls = LinearSVC()
            cls.fit(X_train[:, n_features], np.transpose(k))
            k_features.append(cls.coef_)
        features = np.sum(np.absolute(k_features), axis=0)
        sorted_features = sorted(list(zip(n_features,features[0].tolist())),key=lambda x:x[1], reverse=True)
        best_features = list(map(lambda x: x[0],sorted_features[:int(len(features[0]) / 2)]))
        n_features = best_features

    cls = LinearSVC()
    selector = RFE(cls, n_features_to_select=H,)
    selector.fit(X_train[:, n_features], np.transpose(k))
    best_H_features = selector.get_feature_names_out(n_features).tolist()
    return best_H_features




def get_ensemble_H_features(X_train,y_i,H):
    """
    Get best H features using ensemble bagging method
    :param X_train:
    :param y_i:
    :param H:
    :return:
    """
    n_features = list(range(X_train.shape[1]))
    while len(n_features) > 2 * H:
        k_features = list()

        for k in y_i:
            for _ in range(4):
                X,y = get_samples(X_train[:, n_features],k)
                cls = LinearSVC()
                cls.fit(X, np.transpose(y))
                k_features.append(cls.coef_)
        features = np.sum(np.absolute(k_features), axis=0)
        sorted_features = sorted(list(zip(n_features, features[0].tolist())), key=lambda x: x[1], reverse=True)
        best_features = list(map(lambda x: x[0], sorted_features[:int(len(features[0]) / 2)]))
        n_features = best_features

    cls = LinearSVC()
    selector = RFE(cls, n_features_to_select=H,)
    selector.fit(X_train[:, n_features], np.transpose(k))
    best_H_features = selector.get_feature_names_out(n_features).tolist()
    return best_H_features

def get_samples(X,y):
    """
    get samples from X and y while preserving the distribution of the data between the classes
    :param X:
    :param y:
    :return:
    """
    classes = np.unique(y)
    new_samples=list()
    for k in classes:
        new_samples += np.random.choice(np.where(y==k)[0],len(np.where(y==k)[0])).tolist()
    return X[new_samples], y[new_samples]