import numpy as np
import cv2 as cv
from scipy.spatial import distance
import copy

def kmeans_multiple(X, K, iters,R):
    data_shape = X.shape
    image1 = X.reshape(-1,X.shape[-1])
    min_val = np.amin(X)
    max_val = np.amax(X)
    means = np.random.rand(K,image1.shape[1]) * (max_val - min_val)
    means_new = copy.deepcopy(means)
    dista = distance.cdist(means,image1,'euclidean')
    minima = np.argmin(dista,axis=0)
    i = 0
    j = 0
    m = 0
    n = 0
    ssd = np.empty(K)
    while i < K:
        ssd[i] = 1000000.0
        i += 1
    i = 0
    ssd_new = np.empty(K)
    while n < R:
        means_new = np.random.rand(K,image1.shape[1]) * (max_val - min_val)
        m = 0
        while m < iters:
            dista = distance.cdist(means_new,image1,'euclidean')
            minima_new = np.argmin(dista,axis=0)
            uniqueks , totals = np.unique(minima_new,return_counts=True)
            i = 0
            for idx,val in np.ndenumerate(totals):
                cols = image1.shape[1]
                new_array = np.empty([val,cols])
                l = 0
                j = 0
                while j < image1.shape[0]:
                    if minima_new[j] == uniqueks[idx]:
                        new_array[l,:] = image1[j,:]
                        l += 1
                    j += 1
                means_new[uniqueks[idx],:] = np.mean(new_array,axis=0)
                ssd_new[uniqueks[idx]] = np.sqrt(np.sum((new_array-means[uniqueks[idx]])**2))/val
                
                i += 1
            m += 1
        if np.sum(ssd) >= np.sum(ssd_new):
            ssd = copy.deepcopy(ssd_new)
            means = copy.deepcopy(means_new)
            minima = minima_new
        
        n += 1
    new_minima = minima.reshape(data_shape[0],data_shape[1])
    return new_minima, means, ssd
#img = cv.imread("./Input/im3.png")
#kmeans_multiple(img,5,7,1)

