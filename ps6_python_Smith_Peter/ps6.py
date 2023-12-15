import numpy as np
import math
import scipy as sc
from get_means_std_dev import get_means_and_std_dev
from calculate_probability import calculate_probability, calculate_Max_Likelihood
problem_3_data = sc.io.loadmat('./input/hw4_data3.mat')
X_train = problem_3_data["X_train"]
X_test = problem_3_data["X_test"]
y_train = problem_3_data["y_train"]
y_test = problem_3_data["y_test"]
y_test = np.reshape(y_test,(25))
classes, totals = np.unique(y_train,return_counts=True)

X_train_1 = np.zeros((totals[0],4))
X_train_2 = np.zeros((totals[1],4))
X_train_3 = np.zeros((totals[2],4))
i = 0
j = 0
while i < totals[0]:
    if y_train[j] == classes[0]:
        X_train_1[i,:] = X_train[j,:]
        i += 1
    j += 1
i = 0
j = 0
while i < totals[1]:
    if y_train[j] == classes[1]:
        X_train_2[i,:] = X_train[j,:]
        i += 1
    j += 1
i = 0
j = 0
while i < totals[2]:
    if y_train[j] == classes[2]:
        X_train_3[i,:] = X_train[j,:]
        i += 1
    j += 1
print("Shape of X_train_1 = " ,str(X_train_1.shape))
print("Shape of X_train_2 = " ,str(X_train_2.shape))
print("Shape of X_train_3 = " ,str(X_train_3.shape))
X_1_means,X_1_std = get_means_and_std_dev(X_train_1)
X_2_means,X_2_std = get_means_and_std_dev(X_train_2)
X_3_means,X_3_std = get_means_and_std_dev(X_train_3)
print("X1 means= ",str(X_1_means))
print("X1 std_Dev= ",str(X_1_std))
print("X2 means= ",str(X_2_means))
print("X2 std_Dev= ",str(X_2_std))
print("X3 means= ",str(X_3_means))
print("X3 std_Dev= ",str(X_3_std))
prob = np.zeros((25,3))
for i in range(25):
    prob[i,0] = calculate_probability(X_test[i,:],X_1_means,X_1_std)
    prob[i,1] = calculate_probability(X_test[i,:],X_2_means,X_2_std)
    prob[i,2] = calculate_probability(X_test[i,:],X_3_means,X_3_std)
y_pred = np.argmax(prob,1) + 1
accuracy = 0.0
for i in range(25):
    if y_test[i] == y_pred[i]:
        accuracy += 1.0
accuracy = accuracy/25 * 100
print("Accuracy of classifier = ",str(accuracy),"%")

sigma_1 = np.cov(np.transpose(X_train_1))
sigma_2 = np.cov(np.transpose(X_train_2))
sigma_3 = np.cov(np.transpose(X_train_3))
print("Sigma 1 shape = ",str(sigma_1.shape))
print(sigma_1)
print("Sigma 2 shape = ",str(sigma_2.shape))
print(sigma_2)
print("Sigma 3 shape = ",str(sigma_3.shape))
print(sigma_3)
X_1_means = np.reshape(X_1_means,(4,1))
X_2_means = np.reshape(X_2_means,(4,1))
X_3_means = np.reshape(X_3_means,(4,1))
print("Class 1 Means = ",str(X_1_means))
print("Class 2 Means = ",str(X_2_means))
print("Class 3 Means = ",str(X_3_means))
c1 = -2*math.log(2*math.pi) - 0.5 * math.log(np.linalg.norm(sigma_1,ord=1)) + math.log(1/3)
c2 = -2*math.log(2*math.pi) - 0.5 * math.log(np.linalg.norm(sigma_2,ord=1)) + math.log(1/3)
c3 = -2*math.log(2*math.pi) - 0.5 * math.log(np.linalg.norm(sigma_3,ord=1)) + math.log(1/3)
prob1 = np.zeros((25,3))
for i in range(25):
    prob[i,0] = calculate_Max_Likelihood(X_test[i,:],X_1_means,sigma_1,c1)
    prob[i,1] = calculate_Max_Likelihood(X_test[i,:],X_2_means,sigma_2,c2)
    prob[i,2] = calculate_Max_Likelihood(X_test[i,:],X_3_means,sigma_3,c3)
y_pred = np.argmax(prob,1) + 1
accuracy = 0.0
for i in range(25):
    if y_test[i] == y_pred[i]:
        accuracy += 1.0
accuracy = accuracy/25 * 100
print("Accuracy of max likelihood classifier = ",str(accuracy),"%")