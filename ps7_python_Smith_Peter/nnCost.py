import numpy as np
from predict import predict
def nnCost(Theta1,Theta2,X,y,K,Lambda):
    m = len(y)
    Labels = np.zeros((K,m))
    for i in range(m):
        Labels[y[i]-1,i] = 1
    p , h_k = predict(Theta1,Theta2,X)
    cost = -1/m * (np.sum(np.multiply(Labels,np.log(h_k))+ np.multiply(1-Labels,np.log(1-h_k)))) + Lambda/(2*m)*(np.sum(np.square(Theta1[:,1:]))+ np.sum(np.square(Theta2[:,1:])))
    return cost