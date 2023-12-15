import numpy as np

def sigmoid(z):
    out = 1/(1 + np.exp(-z))
    return out