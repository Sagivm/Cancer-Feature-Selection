def get_active_features(n_features,selected_features):
    features = list()
    for i in range(n_features):
        features.append(1 if i in selected_features else 0)
    return features