import numpy as np
from sigmoid import sigmoid as sg

def gradFunction(theta, X_train, y_train):
    theta1 = np.reshape(theta,(len(theta),1))
    y_vec = np.reshape(y_train,(len(y_train),1))
    z = X_train.dot(theta1)
    h_x = sg(z)
    X_transpose = np.transpose(X_train)
    grad = X_transpose.dot(h_x - y_vec)
    return grad.flatten()