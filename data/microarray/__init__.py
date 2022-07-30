import pandas as pd
import numpy as np
import os

def read_golub():
    file_i = os.path.join('data', 'microarray', 'golub_inputs.csv')
    file_o = os.path.join('data', 'microarray', 'golub_outputs.csv')
    try:
        X = pd.read_csv(file_i).to_numpy()
        y = pd.read_csv(file_o).to_numpy().astype(int)
        return X,y
    except FileNotFoundError:
        raise NotImplementedError


def read_khan():
    file_i = os.path.join('data', 'microarray', 'khan_inputs.csv')
    file_o = os.path.join('data', 'microarray', 'khan_outputs.csv')
    try:
        X = pd.read_csv(file_i).to_numpy()
        y = pd.read_csv(file_o).to_numpy().astype(int)
        return X,y
    except FileNotFoundError:
        raise NotImplementedError

def read_su():
    file_i = os.path.join('data', 'microarray', 'su_inputs.csv')
    file_o = os.path.join('data', 'microarray', 'su_outputs.csv')
    try:
        X = pd.read_csv(file_i).to_numpy()
        y = pd.read_csv(file_o).to_numpy().astype(int)
        return X,y
    except FileNotFoundError:
        raise NotImplementedError


def read_sorile():
    file_i = os.path.join('data', 'microarray', 'sorlie_inputs.csv')
    file_o = os.path.join('data', 'microarray', 'sorlie_outputs.csv')
    try:
        X = pd.read_csv(file_i).to_numpy()
        y = pd.read_csv(file_o).to_numpy().astype(int)
        return X,y
    except FileNotFoundError:
        raise NotImplementedError


def read_submar():
    file_i = os.path.join('data', 'microarray', 'subramanian_inputs.csv')
    file_o = os.path.join('data', 'microarray', 'subramanian_outputs.csv')
    try:
        X = pd.read_csv(file_i).to_numpy()
        y = pd.read_csv(file_o).to_numpy().astype(int)
        return X,y
    except FileNotFoundError:
        raise NotImplementedError