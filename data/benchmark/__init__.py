# from scipy.io import arff
import pandas as pd
import os
import arff
import numpy as np


def read_gevers_rectum():
    file_i = os.path.join('data', 'benchmark', 'Gevers2014_IBD_rectum.csv')
    file_o = os.path.join('data', 'benchmark', 'Gevers2014_IBD_rectum.mf.csv')
    try:
        X = pd.read_csv(file_i).to_numpy()[:,1:]
        y = np.unique(pd.read_csv(file_o).to_numpy()[:, 1:], return_inverse=True)[1]

        return X,y
    except FileNotFoundError:
        raise NotImplementedError


def read_gevers_ileum():
    file_i = os.path.join('data', 'benchmark', 'Gevers2014_IBD_ileum.csv')
    file_o = os.path.join('data', 'benchmark', 'Gevers2014_IBD_ileum.mf.csv')
    try:
        X = pd.read_csv(file_i).to_numpy()[:,1:]
        y = np.unique(pd.read_csv(file_o).to_numpy()[:, 1:], return_inverse=True)[1]

        return X,y
    except FileNotFoundError:
        raise NotImplementedError


def read_morgan():
    file_i = os.path.join('data', 'benchmark', 'Morgan2012_IBD.3.csv')
    file_o = os.path.join('data', 'benchmark', 'Morgan2012_IBD.3.mf.csv')
    try:
        X = pd.read_csv(file_i).to_numpy()[:,1:]
        y = np.unique(pd.read_csv(file_o).to_numpy()[:, 1:], return_inverse=True)[1]

        return X,y
    except FileNotFoundError:
        raise NotImplementedError



def read_wu():
    file_i = os.path.join('data', 'benchmark', 'Wu2011_Diet.csv')
    file_o = os.path.join('data', 'benchmark', 'Wu2011_Diet.mf.csv')
    try:
        X = pd.read_csv(file_i).to_numpy()[:,1:]
        y = np.unique(pd.read_csv(file_o).to_numpy()[:, 1:], return_inverse=True)[1]

        return X,y
    except FileNotFoundError:
        raise NotImplementedError



def read_firer():
    file_i = os.path.join('data', 'benchmark', 'Fierer2010_Subject.3.csv')
    file_o = os.path.join('data', 'benchmark', 'Fierer2010_Subject.3.mf.csv')
    try:
        X = pd.read_csv(file_i).to_numpy()[:,1:]
        y = np.unique(pd.read_csv(file_o).to_numpy()[:, 1:], return_inverse=True)[1]

        return X,y
    except FileNotFoundError:
        raise NotImplementedError
