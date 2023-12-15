import numpy as np
import gradientDescent
import computeCost as cc
import normalEqn as neq
import matplotlib.pyplot as plt

data1 = np.genfromtxt("./input/hw2_data1.csv",delimiter=",", usecols= [0])
hp = np.reshape(data1,(len(data1),1))
data2 = np.genfromtxt("./input/hw2_data1.csv",delimiter=",", usecols= [1])
price = np.reshape(data2,(len(data2),1))


X_vec = np.c_[np.ones((len(hp),1)),hp]
print("X size is ",X_vec.shape)
print("y vector is ", price.shape)
data3 = np.genfromtxt("./input/hw2_data1.csv",delimiter=",", usecols= [0,1])
np.random.shuffle(data3)
xtest = data3[:17,0]
ytest = data3[:17,1]
x_train = data3[18:,0]
y_train = data3[18:,1]
theta, cost = gradientDescent.gradientDescent(x_train,y_train,0.3,500)
its = np.array(range(0,500))
plt.scatter(its,cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost over iterations")
plt.show()
plt.xlabel("Price in $1,000s")
plt.ylabel("Horsepower of a car in 100s")
plt.title("Linear Regression Dataset")
plt.scatter(hp,price,marker='x',label= 'dataset')
plt.axline((0,int(theta[0])),slope=theta[1],label= 'hypothesis')
plt.legend()
plt.show()
print("Theta 0 = ",theta[0])
print("Theta 1 = ",theta[1])
test_cost = cc.computeCost(xtest,ytest,theta)
print("Prediction Error =", test_cost)
theta1 = neq.normalEqn(x_train,y_train)
print("Theta 0 normal equation = ",theta1[0])
print("Theta 1 normal equation = ",theta1[1])
test_cost1 = cc.computeCost(xtest,ytest,theta1)
print("Prediction Error normal equation =", test_cost1)
theta, cost = gradientDescent.gradientDescent(x_train,y_train,0.001,300)
its = np.array(range(0,300))
plt.scatter(its,cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs iterations#, alpha = 0.001")
plt.show()
theta, cost = gradientDescent.gradientDescent(x_train,y_train,0.003,300)
its = np.array(range(0,300))
plt.scatter(its,cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs iterations#, alpha = 0.003")
plt.show()
theta, cost = gradientDescent.gradientDescent(x_train,y_train,0.03,300)
its = np.array(range(0,300))
plt.scatter(its,cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs iterations#, alpha = 0.03")
plt.show()
#theta, cost = gradientDescent.gradientDescent(x_train,y_train,3,300)
#its = np.array(range(0,300))
#plt.scatter(its,cost)
#plt.xlabel("Iterations")
#plt.ylabel("Cost")
#plt.title("Cost vs iterations#, alpha = 3")
#plt.show()
data1 = np.genfromtxt("./input/hw2_data2.txt",delimiter=",", usecols= [0,1])
data2 = np.genfromtxt("./input/hw2_data2.txt",delimiter=",", usecols= [2])
mean_sq_ft = np.mean(data1[:,0])
mean_bedrooms = np.mean(data1[:,1])
print("Mean of square feet = ",mean_sq_ft)
print("Mean of bedrooms = ", mean_bedrooms)
std_dev_sq_ft = np.std(data1[:,0])
std_dev_bedrooms = np.std(data1[:,1])
print("Standard Deviation of square feet = ",std_dev_sq_ft)
print("Standard Deviation of bedrooms = ", std_dev_bedrooms)
data3 = np.genfromtxt("./input/hw2_data2.txt",delimiter=",", usecols= [0,1])
x_vec = np.c_[np.ones((len(data3),1)),data3]
print("Shape of X = ",x_vec.shape)
y_vec = np.reshape(data2,(len(data2),1))
print("shape of y = ",y_vec.shape)

data1[:,0]= (data1[:,0] - mean_sq_ft)/std_dev_sq_ft
data1[:,1] = (data1[:,1] - mean_bedrooms)/std_dev_bedrooms

theta, cost = gradientDescent.gradientDescent(data1,data2,0.1,750)
its = np.array(range(0,750))
plt.scatter(its,cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs iterations#, alpha = 0.01")
plt.show()
print("Theta0 = ",theta[0])
print("Theta1 = ",theta[1])
print("Theta2 = ",theta[2])
test_sq_ft = (1080 - mean_sq_ft)/std_dev_sq_ft
test_bedrooms = (2 - mean_bedrooms)/std_dev_bedrooms
print("Predicted cost of house is ",theta[0] + theta[1]*test_sq_ft + theta[2]*test_bedrooms)