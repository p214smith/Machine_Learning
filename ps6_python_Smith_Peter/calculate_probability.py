import numpy as np
import math

def calculate_probability(X_train,mean,std_dev):
    length = len(X_train)
    prob = 0.0
    for i in range(length):
        prob += math.log((1/(math.sqrt(2*math.pi)*std_dev[i]))*(math.exp(-(((X_train[i]-mean[i])**2)/(2*(std_dev[i]**2))))))
    return prob

def calculate_Max_Likelihood(X_train,mean,sigma,C):
    length = len(X_train)
    X_train = np.reshape(X_train,(4,1))
    G = -.5 * np.dot(np.dot(np.transpose(X_train - mean),np.linalg.pinv(sigma)),(X_train - mean)) + C
    return G