import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from data.spectf.read_data import *
from sklearn.preprocessing import StandardScaler

def main():
    X_train, y_train = read_train()
    X_test, y_test = read_test()
    estimator = SVC(kernel="linear")
    selector = RFE(estimator)
    selector.fit(X_train,y_train)
    print(selector.get_feature_names_out())


if __name__ == "__main__":
    main()