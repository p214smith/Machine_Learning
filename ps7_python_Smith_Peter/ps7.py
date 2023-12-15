import numpy as np
import scipy as sc
from predict import predict
from sigmoid import sigmoid
from nnCost import nnCost
from sGD import sGD
from sigmoidGradient import sigmoidGradient
data = sc.io.loadmat('./input/HW7_Data.mat')
thetas = sc.io.loadmat('./input/HW7_weights_2.mat')
Theta1 = np.array(thetas["Theta1"])
Theta2 = np.array(thetas["Theta2"])
x_data = np.array(data["X"])
y_data = np.array(data["y"])
p , h_x = predict(Theta1,Theta2,x_data)
predictions = 0
for i in range(len(p)):
    if p[i] == y_data[i]:
        
        predictions += 1
accuracy = predictions/len(p) * 100
print("Accuracy of model is ", str(accuracy))
cost_0 = nnCost(Theta1,Theta2,x_data,y_data,3,0)
print("The cost with lambda = 0 : ",str(cost_0))
cost_1 = nnCost(Theta1,Theta2,x_data,y_data,3,1)
print("The cost with lambda = 1 : ",str(cost_1))
cost_2 = nnCost(Theta1,Theta2,x_data,y_data,3,2)
print("The cost with lambda = 2 : ",str(cost_2))
z = np.array([-10,0,10])
z = sigmoid(z)
gradient = sigmoidGradient(z)
print(gradient)

t1, t2 = sGD(4,8,3,x_data,y_data,0.01,.015,100)
IDX = np.arange(150)

np.random.shuffle(IDX)
x_test = x_data[IDX[:22],:]
x_train = x_data[IDX[22:],:]
y_test = y_data[IDX[:22],:]
y_train = y_data[IDX[22:],:]
t1, t2 = sGD(4,8,3,x_train,y_train,0.01,.015,1000)
p , h_x = predict(t1,t2,x_train)
predictions = 0
for i in range(len(p)):
    if p[i] == y_train[i]:
        
        predictions += 1
accuracy = predictions/len(p) * 100
print("Accuracy of training samples is ", str(accuracy))
print("Cost of training samples = ", str(nnCost(t1,t2,x_train,y_train,3,0.01)))
p , h_x = predict(t1,t2,x_test)
predictions = 0
for i in range(len(p)):
    if p[i] == y_test[i]:
        
        predictions += 1
accuracy = predictions/len(p) * 100
print("Accuracy of testing samples is ", str(accuracy))
print("Cost of testing samples = ", str(nnCost(t1,t2,x_test,y_test,3,0.01)))

# np.random.shuffle(IDX)
# x_test = x_data[IDX[:22],:]
# x_train = x_data[IDX[22:],:]
# y_test = y_data[IDX[:22],:]
# y_train = y_data[IDX[22:],:]
# t1, t2 = sGD(4,8,3,x_train,y_train,0,.015,100)
# p , h_x = predict(t1,t2,x_train)
# predictions = 0
# for i in range(len(p)):
#     if p[i] == y_train[i]:
        
#         predictions += 1
# accuracy = predictions/len(p) * 100
# print("Accuracy of training samples is ", str(accuracy))
# print("Cost of training samples = ", str(nnCost(t1,t2,x_train,y_train,3,0)))
# p , h_x = predict(t1,t2,x_test)
# predictions = 0
# for i in range(len(p)):
#     if p[i] == y_test[i]:
        
#         predictions += 1
# accuracy = predictions/len(p) * 100
# print("Accuracy of testing samples is ", str(accuracy))
# print("Cost of testing samples = ", str(nnCost(t1,t2,x_test,y_test,3,0)))

# np.random.shuffle(IDX)
# x_test = x_data[IDX[:22],:]
# x_train = x_data[IDX[22:],:]
# y_test = y_data[IDX[:22],:]
# y_train = y_data[IDX[22:],:]
# t1, t2 = sGD(4,8,3,x_train,y_train,0.01,.015,50)
# p , h_x = predict(t1,t2,x_train)
# predictions = 0
# for i in range(len(p)):
#     if p[i] == y_train[i]:
        
#         predictions += 1
# accuracy = predictions/len(p) * 100
# print("Accuracy of training samples is ", str(accuracy))
# print("Cost of training samples = ", str(nnCost(t1,t2,x_train,y_train,3,0.01)))
# p , h_x = predict(t1,t2,x_test)
# predictions = 0
# for i in range(len(p)):
#     if p[i] == y_test[i]:
        
#         predictions += 1
# accuracy = predictions/len(p) * 100
# print("Accuracy of testing samples is ", str(accuracy))
# print("Cost of testing samples = ", str(nnCost(t1,t2,x_test,y_test,3,0.01)))

# np.random.shuffle(IDX)
# x_test = x_data[IDX[:22],:]
# x_train = x_data[IDX[22:],:]
# y_test = y_data[IDX[:22],:]
# y_train = y_data[IDX[22:],:]
# t1, t2 = sGD(4,8,3,x_train,y_train,0.01,.015,100)
# p , h_x = predict(t1,t2,x_train)
# predictions = 0
# for i in range(len(p)):
#     if p[i] == y_train[i]:
        
#         predictions += 1
# accuracy = predictions/len(p) * 100
# print("Accuracy of training samples is ", str(accuracy))
# print("Cost of training samples = ", str(nnCost(t1,t2,x_train,y_train,3,0.01)))
# p , h_x = predict(t1,t2,x_test)
# predictions = 0
# for i in range(len(p)):
#     if p[i] == y_test[i]:
        
#         predictions += 1
# accuracy = predictions/len(p) * 100
# print("Accuracy of testing samples is ", str(accuracy))
# print("Cost of testing samples = ", str(nnCost(t1,t2,x_test,y_test,3,0.01)))

# np.random.shuffle(IDX)
# x_test = x_data[IDX[:22],:]
# x_train = x_data[IDX[22:],:]
# y_test = y_data[IDX[:22],:]
# y_train = y_data[IDX[22:],:]
# t1, t2 = sGD(4,8,3,x_train,y_train,0.1,.015,50)
# p , h_x = predict(t1,t2,x_train)
# predictions = 0
# for i in range(len(p)):
#     if p[i] == y_train[i]:
        
#         predictions += 1
# accuracy = predictions/len(p) * 100
# print("Accuracy of training samples is ", str(accuracy))
# print("Cost of training samples = ", str(nnCost(t1,t2,x_train,y_train,3,0.1)))
# p , h_x = predict(t1,t2,x_test)
# predictions = 0
# for i in range(len(p)):
#     if p[i] == y_test[i]:
        
#         predictions += 1
# accuracy = predictions/len(p) * 100
# print("Accuracy of testing samples is ", str(accuracy))
# print("Cost of testing samples = ", str(nnCost(t1,t2,x_test,y_test,3,0.1)))

# np.random.shuffle(IDX)
# x_test = x_data[IDX[:22],:]
# x_train = x_data[IDX[22:],:]
# y_test = y_data[IDX[:22],:]
# y_train = y_data[IDX[22:],:]
# t1, t2 = sGD(4,8,3,x_train,y_train,0.1,.015,100)
# p , h_x = predict(t1,t2,x_train)
# predictions = 0
# for i in range(len(p)):
#     if p[i] == y_train[i]:
        
#         predictions += 1
# accuracy = predictions/len(p) * 100
# print("Accuracy of training samples is ", str(accuracy))
# print("Cost of training samples = ", str(nnCost(t1,t2,x_train,y_train,3,0.1)))
# p , h_x = predict(t1,t2,x_test)
# predictions = 0
# for i in range(len(p)):
#     if p[i] == y_test[i]:
        
#         predictions += 1
# accuracy = predictions/len(p) * 100
# print("Accuracy of testing samples is ", str(accuracy))
# print("Cost of testing samples = ", str(nnCost(t1,t2,x_test,y_test,3,0.1)))

# np.random.shuffle(IDX)
# x_test = x_data[IDX[:22],:]
# x_train = x_data[IDX[22:],:]
# y_test = y_data[IDX[:22],:]
# y_train = y_data[IDX[22:],:]
# t1, t2 = sGD(4,8,3,x_train,y_train,1,.015,50)
# p , h_x = predict(t1,t2,x_train)
# predictions = 0
# for i in range(len(p)):
#     if p[i] == y_train[i]:
        
#         predictions += 1
# accuracy = predictions/len(p) * 100
# print("Accuracy of training samples is ", str(accuracy))
# print("Cost of training samples = ", str(nnCost(t1,t2,x_train,y_train,3,1)))

# p , h_x = predict(t1,t2,x_test)
# predictions = 0
# for i in range(len(p)):
#     if p[i] == y_test[i]:
        
#         predictions += 1
# accuracy = predictions/len(p) * 100
# print("Accuracy of testing samples is ", str(accuracy))
# print("Cost of testing samples = ", str(nnCost(t1,t2,x_test,y_test,3,1)))

# np.random.shuffle(IDX)
# x_test = x_data[IDX[:22],:]
# x_train = x_data[IDX[22:],:]
# y_test = y_data[IDX[:22],:]
# y_train = y_data[IDX[22:],:]
# t1, t2 = sGD(4,8,3,x_train,y_train,1,.015,100)
# p , h_x = predict(t1,t2,x_train)
# predictions = 0
# for i in range(len(p)):
#     if p[i] == y_train[i]:
        
#         predictions += 1
# accuracy = predictions/len(p) * 100
# print("Accuracy of training samples is ", str(accuracy))
# print("Cost of training samples = ", str(nnCost(t1,t2,x_train,y_train,3,1)))

# p , h_x = predict(t1,t2,x_test)
# predictions = 0
# for i in range(len(p)):
#     if p[i] == y_test[i]:
        
#         predictions += 1
# accuracy = predictions/len(p) * 100
# print("Accuracy of testing samples is ", str(accuracy))
# print("Cost of testing samples = ", str(nnCost(t1,t2,x_test,y_test,3,1)))