import numpy as np
from sigmoid import sigmoid as sg

def costFunction(theta,X_train,y_train):
    theta1 = np.reshape(theta,(len(theta),1))
    z = X_train.dot(theta1)
    h_x = sg(z)
    m = len(X_train)
    cost = -((np.dot(y_train,np.log(h_x))+ np.dot(1-y_train, np.log(1 - h_x))))/m
    return cost