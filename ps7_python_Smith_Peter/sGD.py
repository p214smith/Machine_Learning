import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from nnCost import nnCost
from matplotlib import pyplot as plt
def sGD(input_layer_size,hidden_layer_size,num_labels,X_train,y_train,Lambda,alpha,MaxEpochs):
    m = len(y_train)
    s = X_train.shape
    cost = []
    Labels = np.zeros((num_labels,m))
    for i in range(m):
        Labels[y_train[i]-1,i] = 1
    X = np.transpose(X_train)
    new_row = np.ones((1,s[0]))
    new_x = np.concatenate((new_row,X),0)
    theta1 =(np.random.rand(hidden_layer_size,input_layer_size+1)-.5)*0.3
    if num_labels > 2:
        theta2 =(np.random.rand(num_labels,hidden_layer_size+1)-.5)*0.3
    else:
        theta2 =(np.random.rand(1,hidden_layer_size+1)-.5)*0.3
    for epoch in range(MaxEpochs):
        for i in range(m):
            z2 = np.dot(theta1,np.reshape(new_x[:,i],(input_layer_size+1,1)))
            l2 = sigmoid(z2)
            l2 =np.reshape(l2,(hidden_layer_size,1))
            new_layer_2 = np.concatenate((np.ones((1,1)),l2),0)
            z3 = np.dot(theta2, new_layer_2)
            l3 = sigmoid(z3)
            l3 = np.reshape(l3,(num_labels,1))
            e3 = np.subtract(l3 ,np.reshape(Labels[:,i],(num_labels,1)))
            e2 = np.multiply(np.dot(np.transpose(theta2),e3) , sigmoidGradient(new_layer_2))
            delta3 = np.multiply(e3 , sigmoidGradient(l3))
            delta2 = np.multiply(e2 , sigmoidGradient(new_layer_2))
            for k in range(num_labels):
                for j in range(hidden_layer_size + 1):
                    if j == 0:
                        theta2[k,j] = theta2[k,j] - (alpha * e3[k]*new_layer_2[j])
                    else:
                        theta2[k,j] = theta2[k,j] - (alpha * (e3[k]*new_layer_2[j] + (Lambda * theta2[k,j])))
            for k in range(hidden_layer_size):
                for j in range(input_layer_size + 1):
                    if k == 0:
                        theta1[k,j] = theta1[k,j] - (alpha * e2[k+1] * new_x[j,i])
                    else:
                        theta1[k,j] = theta1[k,j] - (alpha * (e2[k+1] * new_x[j,i] + (Lambda * theta1[k,j])))
        cost.append(nnCost(theta1,theta2,X_train,y_train,num_labels,Lambda))
    plt.plot(range(MaxEpochs),cost)
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.show()
    return theta1, theta2