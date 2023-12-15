import numpy as np
def gradientDescent(X_train, y_train, alpha, iters):
    X_vec = np.c_[np.ones((len(X_train),1)),X_train]
    theta = np.random.randn(len(X_vec[0]),1)
    y = np.reshape(y_train,(len(y_train),1))
    X_transpose = np.transpose(X_vec)
    cost_vector = []
    m = len(X_train)
    i = 0
    j = 0
    while i < iters:
        grad = 2/m * X_transpose.dot(X_vec.dot(theta) - y)
        print(grad.shape)
        theta = theta - alpha * grad
        y_prediction = X_vec.dot(theta)
        cost = 1/(2*m)*((y_prediction - y)**2.0)
        total_cost = 0
        j = 0
        while j < m:
            total_cost += cost[j][0]
            j += 1
        cost_vector.append(total_cost)
        i +=1
    return theta, cost_vector

x = np.array([0,2,3,4])
y = np.array([4,8,10,12])
theta , cost = gradientDescent(x,y,0.001,15)
print("Theta for gradient descent is: ",theta)
print("Cost is: ",cost)