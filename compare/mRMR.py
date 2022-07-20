import pandas as pd
from mrmr import mrmr_classif
from sklearn.datasets import make_classification
from data.spectf.read_data import *


# working dir has been up
X_train, y_train = read_train()
X_test, y_test = read_test()

X_df = pd.DataFrame(X_train, columns = list(range(X_train.shape[1])))
y_df = pd.Series(y_train)


# use mrmr classification
selected_features = mrmr_classif(X_df, y_df, K = 10)
print(sorted(selected_features))