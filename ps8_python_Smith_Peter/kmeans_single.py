import numpy as np
import cv2 as cv
from scipy.spatial import distance

def kmeans_single(X, K, iters):
    data_shape = X.shape
    image1 = X.reshape(-1,X.shape[-1])
    min_val = np.amin(X)
    max_val = np.amax(X)
    means = np.random.randint(min_val,max_val,size=(K,image1.shape[1]))

    i = 0
    j = 0
    m = 0
    ssd = np.empty(K)
    while m < iters:
        dista = distance.cdist(means,image1,'euclidean')
        minima = np.argmin(dista,axis=0)
        i = 0
        while i < K:
            uniqueks , totals = np.unique(minima,return_counts=True)
            cols = image1.shape[1]
            new_array = np.empty([totals[i],cols])
            l = 0
            j = 0
            while j < image1.shape[0]:
                if minima[j] == i:
                    new_array[l,:] = image1[j,:]
                    l += 1
                j += 1
            means[i,:] = np.mean(new_array,axis=0)
            ssd[i] = np.sqrt(np.sum((new_array-means[i])**2))/totals[i]
            i += 1
        m += 1
    new_minima = minima.reshape(data_shape[0],data_shape[1])
    return new_minima, means, ssd
 
