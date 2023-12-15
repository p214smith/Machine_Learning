import numpy as np
import cv2 as cv
from scipy.spatial import distance
import copy
from kmeans_multiple import kmeans_multiple

def segment_kmeans(image, K, iters, R):
    ids, means, ssd = kmeans_multiple(image,K,iters,R)
    i = 0
    j = 0
    image_size = image.shape
    while i < image_size[0]:
        j = 0
        while j < image_size[1]:
            image[i,j,:] = means[ids[i,j],:]
            j += 1
        i += 1
    return image