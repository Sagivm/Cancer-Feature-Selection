import numpy as np
def get_active_features(n_features,selected_features):
    """
    Compute binary mask based on the selected features
    :param n_features:
    :param selected_features:
    :return:
    """
    features = list()
    for i in range(n_features):
        features.append(1 if i in selected_features else 0)
    return features

def str_features(mask):
    """
    Compute feature str based on selected binary mask
    :param mask:
    :return:
    """
    features = np.where(mask==True)[0].tolist()
    features = list(map(lambda x: f"Feat{x}",list(features)))
    return "-".join(features)