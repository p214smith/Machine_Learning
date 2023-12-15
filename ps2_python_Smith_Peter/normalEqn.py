import numpy as np

def normalEqn(X_train,y_train):
    X_vec = np.c_[np.ones((len(X_train),1)),X_train]
    x_transpose = np.transpose(X_vec)
    X_transpose_X = np.linalg.pinv(x_transpose.dot(X_vec))
    theta = X_transpose_X.dot(x_transpose.dot(y_train))
    return theta

#x = np.array([0,2,3,4])
#y = np.array([4,8,10,12])
#theta = normalEqn(x,y)
#print("Theta for normal equation is: ",theta)