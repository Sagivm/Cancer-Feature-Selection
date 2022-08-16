# from scipy.io import arff
import pandas as pd
import os
import arff
import numpy as np
def read_ayest():

    file = os.path.join('data',"biocon", 'ayeastCC.csv')
    try:
        data = np.transpose(pd.read_csv(file,header=0).to_numpy())[1:,:]
        col_mean = np.nanmean(data,axis=0)
        inds = np.where(pd.isna(data))
        data[inds] = np.take(col_mean, inds[1])
        X = data[:,1:]
        y = data[:,0].astype(int)
        return X, y
    except FileNotFoundError:
        raise NotImplementedError


def read_dlbcl():

    file = os.path.join('data',"biocon", 'DLBCL.csv')
    try:
        data = np.transpose(pd.read_csv(file,header=0).to_numpy())[1:,:]
        col_mean = np.nanmean(data,axis=0)
        inds = np.where(pd.isna(data))
        data[inds] = np.take(col_mean, inds[1])
        X = data[:,1:]
        y = data[:,0].astype(int)
        return X, y
    except FileNotFoundError:
        raise NotImplementedError

def read_curatedOvarianData():
    file = os.path.join('data', "biocon", 'curatedOvarianData.csv')
    try:
        data = np.transpose(pd.read_csv(file, header=0).to_numpy())[1:, :]
        col_mean = np.nanmean(data, axis=0)
        inds = np.where(pd.isna(data))
        data[inds] = np.take(col_mean, inds[1])
        X = data[:, 1:]
        y = data[:, 0].astype(int)
        return X, y
    except FileNotFoundError:
        raise NotImplementedError


def read_curatedOvarianCLL():
    file = os.path.join('data', "biocon", 'CLL.csv')
    try:
        data = np.transpose(pd.read_csv(file, header=0).to_numpy())[1:, :]
        col_mean = np.nanmean(data, axis=0)
        inds = np.where(pd.isna(data))
        data[inds] = np.take(col_mean, inds[1])
        X = data[:, 1:]
        y = data[:, 0].astype(int)
        return X, y
    except FileNotFoundError:
        raise NotImplementedError


def read_ALL():
    file = os.path.join('data', "biocon", 'ALL.csv')
    try:
        data = np.transpose(pd.read_csv(file, header=0).to_numpy())[1:, :]
        col_mean = np.nanmean(data, axis=0)
        inds = np.where(pd.isna(data))
        data[inds] = np.take(col_mean, inds[1])
        X = data[:, 1:]
        y = data[:, 0].astype(int)
        return X, y
    except FileNotFoundError:
        raise NotImplementedError




def prepare_df():
    run = [("ayest",read_ayest),("all",read_ALL),("CLL",read_curatedOvarianCLL),("curated",read_curatedOvarianData),("dlbcl",read_dlbcl)]
    for data_name,func in run:
        X,y = func()
        data = np.column_stack((X, y))
        df = pd.DataFrame(data)
        df.to_csv(f"{data_name}-data.csv")

