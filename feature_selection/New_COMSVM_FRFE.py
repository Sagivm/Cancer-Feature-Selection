from util.comsvm_frfe_util import *
from sklearn.model_selection import train_test_split
import numpy as np
from util.util import *
def com_esvmfrfe_fs(X,y,k):

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

    H = int(3 / 4 * k)
    K = int(1 / 4 * k)
    # Scaling
    scaler = StandardScaler()
    scaler.fit(np.row_stack(X))
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    d_i, y_i = get_dy_i(X_train,y_train,y_test)

    best_H_features = get_ensemble_H_features(X_train,y_i,H)

    top_features = expand_top_features(X_train,y_train,X_test,y_test,K,best_H_features)
    return get_active_features(X.shape[1],top_features[0])

if __name__ == "__main__":
    X_train, y_train = read_train()
    X_test, y_test = read_test()
    X = np.vstack((X_train, X_test))
    y= np.hstack((y_train,y_test))

    print(com_esvmfrfe_fs(X,y,20))