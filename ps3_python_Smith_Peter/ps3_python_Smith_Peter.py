import numpy as np
import matplotlib.pyplot as plt
from sigmoid import sigmoid as sg
from gradFunction import gradFunction as gf
from costFunction import costFunction as cf
from scipy.optimize import fmin_bfgs

data1 = np.genfromtxt("./input/hw3_data1.txt",delimiter=",",usecols=[0,1])
matrix_x = np.c_[np.ones((len(data1),1)),data1]
print("Size of matrix X = ",matrix_x.shape)
data2 = np.genfromtxt("./input/hw3_data1.txt",delimiter=",",usecols=[2])
vector_y = np.reshape(data2,(len(data2),1))
print("Size of vector y = ", vector_y.shape)
x1 = matrix_x[:,1]
x2 = matrix_x[:,2]
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.title("Admitted vs. not Admitted")
plt.scatter(x1[vector_y[:,0]==0],x2[vector_y[:,0]==0],color="red",marker="o",label="Not Admitted")
plt.scatter(x1[vector_y[:,0]==1],x2[vector_y[:,0]==1],color="blue",marker="x",label="Admitted")
plt.legend()
plt.show()
dataset = np.genfromtxt("./input/hw3_data1.txt",delimiter=",",usecols=[0,1,2])
data3 = np.c_[np.ones((len(dataset),1)),dataset]
np.random.shuffle(data3)
xtest = data3[:10,[0,1,2]]
ytest = data3[:10,3]

x_train = data3[10:,[0,1,2]]

y_train = data3[10:,3]


z = np.linspace(-15.0,15.0,100)
gz = sg(z)
plt.plot(z,gz)
plt.xlabel("z")
plt.ylabel("gz")
plt.title("Sigmoid Function")
plt.show()
theta = np.array([2,0,0])
toy_x = np.array([[1,1,0],[1,1,3],[1,3,1],[1,3,4]])
toy_y = np.array([0,1,0,1])

cost = cf(theta,toy_x,toy_y)
print("cost of toy data is " ,cost)

theta = np.array([0,0,0])

optimize = fmin_bfgs(cf,theta,gf,args=(x_train,y_train))
print("Optimized values for theta = ", optimize)
print("Cost of optimzed values = ",cf(optimize,x_train,y_train))
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.title("Admitted vs. not Admitted")
plt.scatter(x1[vector_y[:,0]==0],x2[vector_y[:,0]==0],color="red",marker="o",label="Not Admitted")
plt.scatter(x1[vector_y[:,0]==1],x2[vector_y[:,0]==1],color="blue",marker="x",label="Admitted")
x_1 = np.linspace(30,100,1000)
y = (-optimize[0] - optimize[1]*x_1)/optimize[2]

plt.plot(x_1,y)
plt.plot()
plt.legend()
plt.show()
y_pred = optimize[0] + optimize[1]*xtest[:,1] + optimize[2]*xtest[:,2]
i = 0
while i < len(y_pred):
    if y_pred[i] > 0.0:
        y_pred[i] = 1
    else:
       y_pred[i] = 0
    i+=1
i = 0
j =0
while i < len(y_pred):
    if y_pred[i] == ytest[i]:
        j+=1
    i+= 1
print("Model accuracy = ", j/10)
test = optimize[0] + optimize[1]*70 + optimize[2]*55
print("The admission probability = ",test)
if test > 0 :
    print("Student should be admitted")
else:
    print("Student should not be admitted")

def normalEqn(X_train,y_train):
    X_vec = np.c_[np.ones((len(X_train),1)),X_train]
    x_transpose = np.transpose(X_vec)
    X_transpose_X = np.linalg.pinv(x_transpose.dot(X_vec))
    theta = X_transpose_X.dot(x_transpose.dot(y_train))
    return theta
data1 = np.genfromtxt("./input/hw3_data2.csv",delimiter=",",usecols=[0])
data2 = data1 * data1
X_train1 = np.c_[data1,data2]
y_train1 = np.genfromtxt("./input/hw3_data2.csv",delimiter=",",usecols=[1])
thetas = normalEqn(X_train1,y_train1)
print("Thetas for homes versus profits = ",thetas)
x_1 = np.linspace(500,1000,1000)
y_1 = thetas[0] + thetas[1] * x_1 + thetas[2] * (x_1**2)
plt.plot(x_1,y_1,label="Fitted Model")
plt.scatter(data1,y_train1,label="Training Data")
plt.legend()
plt.xlabel("Population in thousandths, n")
plt.ylabel("Profit")
plt.show()