import numpy as np
from sigmoid import sigmoid

def sigmoidGradient(z):
    gradient = np.multiply(z,1-z)
    return gradient