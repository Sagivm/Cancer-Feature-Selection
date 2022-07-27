from util.comsvm_frfe_util import *

def main():

    H = 10
    K = 5

    X_train, y_train = read_train()
    X_test, y_test = read_test()

    # Scaling
    scaler = StandardScaler()
    scaler.fit(np.row_stack((X_train,X_test)))
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    d_i, y_i = get_dy_i(X_train,y_train,y_test)

    best_H_features = get_H_features(X_train,y_i,H)

    top_features = expand_top_features(X_train,y_train,X_test,y_test,K,best_H_features)
    print(top_features)

if __name__ == "__main__":
    main()