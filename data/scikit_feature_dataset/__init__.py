import scipy.io


def read_basehock(mat_file):
    try:
        mat = scipy.io.loadmat(mat_file)
        return mat['X'],mat['Y']
    except FileNotFoundError:
        raise NotImplementedError



def read_leukemia(mat_file):
    try:
        mat = scipy.io.loadmat(mat_file)
        return mat['X'],mat['Y']
    except FileNotFoundError:
        raise NotImplementedError