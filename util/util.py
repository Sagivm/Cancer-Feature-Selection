import numpy as np
def get_active_features(n_features,selected_features):
    features = list()
    for i in range(n_features):
        features.append(1 if i in selected_features else 0)
    return features

def str_features(mask):
    features = np.where(mask==True)[0].tolist()
    features = list(map(lambda x: f"Feat{x}",list(features)))
    print("-".join(features))
    return "-".join(features)