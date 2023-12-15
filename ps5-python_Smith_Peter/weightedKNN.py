import numpy as np
from scipy.spatial import distance


def weightedKNN(X_train,y_train,X_test,sigma):
    distances = distance.cdist(X_test,X_train,'euclidean')
    Classes ,totals = np.unique(y_train,return_counts=True)
    weights = np.exp(-((distances**2)/(sigma**2)))
    rows, cols = weights.shape
    y_predict = np.zeros(rows,dtype=int)
    total_Ws = np.zeros(len(Classes))
    for i in range(rows):
        total_Ws = np.zeros(len(Classes))
        for j in range(len(Classes)):
            for k in range(cols):
                if (y_train[k] == Classes[j]):
                    total_Ws[j] += weights[i,j]
            total_Ws[j] = total_Ws[j]
        y_predict[i] = Classes[np.argmax(total_Ws)]
    return y_predict