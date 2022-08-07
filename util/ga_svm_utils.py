import os
import random
import numpy as np
import pandas as pd
from data.spectf.read_data import *
from sklearn.svm import SVC, LinearSVC
from genetic_selection import GeneticSelectionCV
from sklearn.preprocessing import StandardScaler

def evaluate_feature_selection(X_train, y_train, X_test, y_test, feature_selection):
    """
    Evaluate feature selection X_train using linear based kernel SVM
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param feature_selection:
    :return:
    """
    cls = LinearSVC(max_iter=2000,tol=1e-5,C=1e-3)
    cls.fit(
        X=X_train[:, feature_selection.astype(bool)],
        y=y_train)

    return cls.score(
        X_test[:, feature_selection.astype(bool)],
        y_test)

def crossover_couple(a,b,N_CROSS=4):
    """
    CrossOver a and b multiple times to result N_CROSS*2 children combinations
    :param a:
    :param b:
    :param N_CROSS:
    :return:
    """
    children = list()
    for _ in range(N_CROSS):
        crossover_point_index = random.randint(0, len(a))
        a_masked_left = a[:crossover_point_index]
        a_masked_right = a[crossover_point_index:]
        b_masked_left = b[:crossover_point_index]
        b_masked_right = b[crossover_point_index:]
        children.append(np.hstack((a_masked_left, b_masked_right)))
        children.append(np.hstack((b_masked_left, a_masked_right)))

    return children