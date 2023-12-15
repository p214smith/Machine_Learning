import numpy as np

def Reg_normalEqn(X_train,y_train,lambda_x):
    X_transpose = np.transpose(X_train)
    D = np.eye(len(X_transpose),len(X_transpose))
    D[0,0]=0
    X_transpose_X = np.linalg.pinv(X_transpose.dot(X_train)+ lambda_x * D)
    theta = X_transpose_X.dot(X_transpose.dot(y_train))
    return theta
