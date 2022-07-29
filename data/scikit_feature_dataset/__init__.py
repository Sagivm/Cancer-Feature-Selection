import os
import scipy.io


def read_basehock():
    mat_file = os.path.join('data', 'scikit_feature_dataset', 'BASEHOCK.mat')
    try:
        mat = scipy.io.loadmat(mat_file)
        return mat['X'],mat['Y']
    except FileNotFoundError:
        raise NotImplementedError



def read_leukemia():
    mat_file = os.path.join('data','scikit_feature_dataset','Leukemia.mat')
    try:
        mat = scipy.io.loadmat(mat_file)
        return mat['X'],mat['Y']
    except FileNotFoundError:
        raise NotImplementedError


def read_lung_small():
    mat_file = os.path.join('data','scikit_feature_dataset','lung_small.mat')
    try:
        mat = scipy.io.loadmat(mat_file)
        return mat['X'],mat['Y']
    except FileNotFoundError:
        raise NotImplementedError



def read_lung():
    mat_file = os.path.join('data','scikit_feature_dataset','lung.mat')
    try:
        mat = scipy.io.loadmat(mat_file)
        return mat['X'],mat['Y']
    except FileNotFoundError:
        raise NotImplementedError




def read_Yale():
    mat_file = os.path.join('data','scikit_feature_dataset','Yale.mat')
    try:
        mat = scipy.io.loadmat(mat_file)
        return mat['X'],mat['Y']
    except FileNotFoundError:
        raise NotImplementedError