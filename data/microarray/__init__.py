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