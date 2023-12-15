import numpy as np
from sigmoid import sigmoid
def predict(theta1, theta2, X):
    s = X.shape
    X = np.transpose(X)
    new_row = np.ones((1,s[0]))
    new_x = np.concatenate((new_row,X),0)
    z2 = np.dot(theta1,new_x)
    layer2 = sigmoid(z2)
    new_layer_2 = np.concatenate((new_row,layer2),0)
    z3 = np.dot(theta2, new_layer_2)
    h_x = sigmoid(z3)
    p = np.argmax(h_x,axis = 0) + 1
    return p , h_x