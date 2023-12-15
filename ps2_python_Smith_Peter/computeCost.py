import numpy as np

def computeCost(x,y,theta):
    cost = 0
    mi = x.shape
    m = mi[0]
    theta0 = theta[0]
    theta1 = theta[1]
    i = 0
    while i < m:
        cost += ((theta0 + theta1 * x[i])-y[i]) **2
        i += 1
    cost = cost /(2*m)
    return cost

#x = np.array([0,2,3,4])
#print(x)
#y = np.array([4,8,10,12])
#theta = (0,0.5)
#theta11 = (1,1)
#cost = computeCost(x,y,theta)
#print("Cost of theta (i): ",cost)
#cost = computeCost(x,y,theta11)
#print("Cost of theta (ii): ", cost)