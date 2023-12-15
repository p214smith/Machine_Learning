import numpy as np

def get_means_and_std_dev(X_train):
    length, width = X_train.shape
    means = np.mean(X_train,0)
    std_dev = np.std(X_train,0)
    return means ,std_dev