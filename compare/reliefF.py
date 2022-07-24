import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from data.spectf.read_data import *
import sklearn_relief as relief
from ReliefF import ReliefF
from sklearn.preprocessing import StandardScaler

def main():
    X_train, y_train = read_train()
    X_test, y_test = read_test()
    r = ReliefF(n_neighbors=20)
    r.fit(X_train,y_train)
    print(sorted(r.top_features))

if __name__ == "__main__":
    main()