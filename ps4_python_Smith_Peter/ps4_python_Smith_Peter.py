import numpy as np
from Reg_normalEqn import Reg_normalEqn as rg
import scipy as sc
import matplotlib.pyplot as plt
from sklearn import neighbors as nb
from logReg_multi import logReg_multi as lr

# def computeCost(x,y,theta):
#     cost = 0
#     mi = x.shape
#     m = mi[0]

#     i = 0
    
#     cost = ((np.dot(x,theta))-y)**2

#     cost = sum(cost) /(2*m)
#     return cost

# problem_1_data = sc.io.loadmat('./input/hw4_data1.mat')
# x_data = np.array(problem_1_data["X_data"])
# y_data = np.array(problem_1_data["y"])
# data_set = np.c_[np.ones((len(x_data),1)),x_data,y_data]
# i = 0
# lambda_p1 = np.array([0.0,0.001,0.003,0.005,0.007,0.009,0.012,0.017])
# train_error = np.zeros((len(lambda_p1),1),dtype=float)
# test_error = np.zeros((len(lambda_p1),1),dtype = float)
# while i < 20:
#     np.random.shuffle(data_set)
#     x_train = data_set[:882,0:501]
#     x_test = data_set[882:,0:501]
#     y_train = data_set[:882,501]
#     y_test = data_set[882:,501]
#     j = 0
#     while j < len(lambda_p1):
#         thetas = rg(x_train,y_train,lambda_p1[j])
#         train_error[j] += computeCost(x_train,y_train,thetas)/20
#         test_error[j] += computeCost(x_test,y_test,thetas)/20
#         j += 1
#     i += 1
# plt.plot(lambda_p1,train_error,linestyle= '-',color='red',marker='*',label='Training Error')
# plt.plot(lambda_p1,test_error,linestyle= '-',color='blue',marker='o',label='Testing Error')
# plt.legend()
# plt.xlabel('lambda')
# plt.ylabel('Average Error')
# plt.show()
problem_2_data = sc.io.loadmat('./input/hw4_data2.mat')
X1 = problem_2_data["X1"]
X2 = problem_2_data["X2"]
X3 = problem_2_data["X3"]
X4 = problem_2_data["X4"]
X5 = problem_2_data["X5"]
y1 = problem_2_data["y1"]
y2 = problem_2_data["y2"]
y3 = problem_2_data["y3"]
y4 = problem_2_data["y4"]
y5 = problem_2_data["y5"]
print(X1.shape)
y1 = np.reshape(y1,(len(y1)))
y2 = np.reshape(y2,(len(y2)))
y3 = np.reshape(y3,(len(y3)))
y4 = np.reshape(y4,(len(y4)))
y5 = np.reshape(y5,(len(y5)))
X_train1 = np.vstack((X1,X2,X3,X4))
y_train1 = np.concatenate((y1,y2,y3,y4))
X_train2 = np.vstack((X1,X2,X3,X5))
y_train2 = np.concatenate((y1,y2,y3,y5))
X_train3 = np.vstack((X1,X2,X5,X4))
y_train3 = np.concatenate((y1,y2,y5,y4))
X_train4 = np.vstack((X1,X5,X3,X4))
y_train4 = np.concatenate((y1,y5,y3,y4))
X_train5 = np.vstack((X5,X2,X3,X4))
y_train5 = np.concatenate((y5,y2,y3,y4))
score1 = 0.0
score2 = 0.0
score15 = 0.0

# score = np.zeros((8,1))
# i = 0
# j = 0
# while j < 15:
#     kofi = nb.KNeighborsClassifier(i+1)
#     kofi.fit(X_train1,y_train1)
#     score[i] += kofi.score(X5,y5)/5
#     kofi.fit(X_train2,y_train2)
#     score[i] += kofi.score(X4,y4)/5
#     kofi.fit(X_train3,y_train3)
#     score[i] += kofi.score(X3,y3)/5
#     kofi.fit(X_train4,y_train4)
#     score[i] += kofi.score(X2,y2)/5
#     kofi.fit(X_train5,y_train5)
#     score[i] += kofi.score(X1,y1)/5
#     i+= 1
#     j += 2
# plt.plot((1,3,5,7,9,11,13,15),score)
# plt.xlabel("k neighbors")
# plt.ylabel("Mean Score")
# plt.show()

# problem_3_data = sc.io.loadmat('./input/hw4_data3.mat')
# X_train = problem_3_data["X_train"]
# X_test = problem_3_data["X_test"]
# y_train = problem_3_data["y_train"]
# y_test = problem_3_data["y_test"]

# y_test_prediction = lr(X_train,y_train,X_test)
# y_train_prediction = lr(X_train,y_train,X_train)
# y_data = np.c_[y_test_prediction,y_test]
# print(y_data)
# i = 0
# j = 0
# while i < len(X_test):
#     if y_test[i] == y_test_prediction[i]:
#         j += 1
#     i += 1
# test_accuracy = j/(i)
# print(test_accuracy)
# i = 0
# j = 0
# while i < len(X_train):
#     if y_train[i] == y_train_prediction[i]:
#         j += 1
#     i += 1
# train_accuracy = j/(i)
# print(train_accuracy)
