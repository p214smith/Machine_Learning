import numpy as np
import scipy as sc
from weightedKNN import weightedKNN
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn import neighbors as nb
from sklearn import svm
import time
import sklearn.multiclass as mc
problem_3_data = sc.io.loadmat('./input/all/hw4_data3.mat')
X_train = problem_3_data["X_train"]
X_test = problem_3_data["X_test"]
y_train = problem_3_data["y_train"]
y_test = problem_3_data["y_test"]

sigmas = np.array([0.01,0.05,0.2,1.5,3.2])
accuracy = np.zeros(len(sigmas))
for i in range(len(sigmas)):
    y_predict = weightedKNN(X_train,y_train,X_test,sigmas[i])
    for j in range(len(y_test)):
        if (y_test[j] == y_predict[j] ):
            accuracy[i] += 1/len(y_test)
print("Accuracy for Sigma = 0.01: ","%.2f" % (accuracy[0] * 100.00),"%")
print("Accuracy for Sigma = 0.05: ","%.2f" % (accuracy[1] * 100.00),"%")
print("Accuracy for Sigma = 0.2: ","%.2f" % (accuracy[2] * 100.00),"%")
print("Accuracy for Sigma = 1.5: ","%.2f" % (accuracy[3] * 100.00),"%")
print("Accuracy for Sigma = 3.2: ","%.2f" % (accuracy[4] * 100.00),"%")
int_shuff = np.array([1,2,3,4,5,6,7,8,9,10])
for i in range(40):
    np.random.shuffle(int_shuff)
    for j in range(8):
        im_file = './input/all/s' + str(i+1) +'/' + str(int_shuff[j]) + '.pgm'
        with open(im_file,'rb') as pgfm:
            im = plt.imread(pgfm)
        im_str = './input/train/' + str(i+1) + '_' + str(j+1) + '.png'
        cv.imwrite(im_str,im)
    im_file = './input/all/s' + str(i+1) +'/' + str(int_shuff[8]) + '.pgm'
    with open(im_file,'rb') as pgfm:
        im = plt.imread(pgfm)
    im_str = './input/test/' + str(i+1) + '_' + str(9) + '.png'
    cv.imwrite(im_str,im)
    im_file = './input/all/s' + str(i+1) +'/' + str(int_shuff[9]) + '.pgm'
    with open(im_file,'rb') as pgfm:
        im = plt.imread(pgfm)
    im_str = './input/test/' + str(i+1) + '_' + str(10) + '.png'
    cv.imwrite(im_str,im)

img = cv.imread("./input/train/28_3.png")
imgplot = plt.imshow(img)
plt.title("Person ID: 28, Image: 3")
plt.show()
T = np.zeros([10304,320])


for i in range(40):
    for j in range(8):
        im_str = './input/train/' + str(i+1) + '_' + str(j+1) + '.png'
        img = cv.imread(im_str,0)
        img_vector = np.reshape(img,(10304))
        T[:,int(i*8 + j)]= img_vector
cv.imwrite('./output/ps5-2-1-a.png',T)
means = np.mean(T,axis=1)
mean_face = np.reshape(means,(112,92))
cv.imwrite('./output/ps5-2-1-b.png',mean_face)
A =np.zeros([10304,320])
for i in range(320):
    A[:,i] = T[:,i] - means
C = np.cov(A)
cv.imwrite('./output/ps5-2-1-c.png',C)

evalue, evect = np.linalg.eig(np.dot(np.transpose(A),A))
sorted_evalues = np.sort(evalue)[::-1]

v_k = 0
k = 0
sum_evalues = np.sum(sorted_evalues)
while v_k <= 0.95:
    v_k += sorted_evalues[k]/sum_evalues
    k += 1
vk = np.zeros(k)
v_k = 0
for i in range(k):
    v_k += sorted_evalues[i]/sum_evalues
    vk[i] = v_k
    
plt.plot(range(k),vk)
plt.xlabel("k")
plt.ylabel("v(k)")
plt.title("k vs v(k)")
plt.show()
print("Value of k: " + str(k))
vals, U = sc.sparse.linalg.eigs(C,k)
indices = np.flip(np.argsort(vals))
print("Shape of U: " + str(U.shape))
img = np.zeros((112*3,92*3),dtype=int)
fig, axs = plt.subplots(3,3)
plt.gray()
axs[0,0].imshow(np.real(np.reshape(U[:,indices[0]],(112,92))))
axs[0,1].imshow(np.real(np.reshape(U[:,indices[1]],(112,92))))
axs[0,2].imshow(np.real(np.reshape(U[:,indices[2]],(112,92))))
axs[1,0].imshow(np.real(np.reshape(U[:,indices[3]],(112,92))))
axs[1,1].imshow(np.real(np.reshape(U[:,indices[4]],(112,92))))
axs[1,2].imshow(np.real(np.reshape(U[:,indices[5]],(112,92))))
axs[2,0].imshow(np.real(np.reshape(U[:,indices[6]],(112,92))))
axs[2,1].imshow(np.real(np.reshape(U[:,indices[7]],(112,92))))
axs[2,2].imshow(np.real(np.reshape(U[:,indices[8]],(112,92))))

plt.show()
im_str = './input/train/' + str(1) + '_' + str(1) + '.png'
img = cv.imread(im_str,0)
img_vector = np.reshape(img,(10304))
X_train = np.zeros((320,k))
y_train = np.zeros(320,int)
for i in range(40):
    for j in range(8):
        im_str = './input/train/' + str(i+1) + '_' + str(j+1) + '.png'
        img = cv.imread(im_str,0)
        img_vector = np.reshape(img,(10304))
        y_train[i*8 +j] = i+1
        W = np.real(np.transpose(U).dot(img_vector - means))
        X_train[i*8 +j,:] = W
X_test = np.zeros((80,k))
y_test = np.zeros(80,int)
for i in range(40):
    im_str = './input/test/' + str(i+1) + '_' + str(9) + '.png'
    img = cv.imread(im_str,0)
    img_vector = np.reshape(img,(10304))
    y_test[i*2] = i + 1
    W = np.real(np.transpose(U).dot(img_vector - means))
    X_test[i*2,:] = W
    im_str = './input/test/' + str(i+1) + '_' + str(10) + '.png'
    img = cv.imread(im_str,0)
    img_vector = np.reshape(img,(10304))
    y_test[i*2 + 1] = i + 1
    W = np.real(np.transpose(U).dot(img_vector - means))
    X_test[i*2 + 1,:] = W
print("Shape of X_train: " + str(X_train.shape))
print("Shape of X_test: " + str(X_test.shape))
score = np.zeros((6))
i = 0
j = 1
while j < 12: 
    kofi = nb.KNeighborsClassifier(j)
    kofi.fit(X_train,y_train)
    score[i] = kofi.score(X_test,y_test)
    i += 1
    j += 2
print(score)
training_times = np.zeros((6))
testing_times = np.zeros((6))
scores = np.zeros((6))
model = svm.NuSVC(kernel='linear',decision_function_shape='ovo')
start = time.time()
model.fit(X_train,y_train)
stop = time.time()
training_times[0] = stop - start
start = time.time()
scores[0] = model.score(X_test,y_test)
stop = time.time()
testing_times[0] = stop - start
model1 = svm.NuSVC(kernel='linear',decision_function_shape='ovr',break_ties=True)
start = time.time()
model1.fit(X_train,y_train)
stop = time.time()
training_times[1] = stop - start
start = time.time()
scores[1] = model1.score(X_test,y_test)
stop = time.time()
testing_times[1] = stop - start
model = svm.NuSVC(kernel='poly',decision_function_shape='ovo',degree=3)
start = time.time()
model.fit(X_train,y_train)
stop = time.time()
training_times[2] = stop - start
start = time.time()
scores[2] = model.score(X_test,y_test)
stop = time.time()
testing_times[2] = stop - start
model1 = svm.NuSVC(kernel='poly',decision_function_shape='ovr',degree=3,break_ties=True)
start = time.time()
model1.fit(X_train,y_train)
stop = time.time()
training_times[3] = stop - start
start = time.time()
scores[3] = model1.score(X_test,y_test)
stop = time.time()
testing_times[3] = stop - start
model = svm.NuSVC(kernel='rbf',decision_function_shape='ovo')
start = time.time()
model.fit(X_train,y_train)
stop = time.time()
training_times[4] = stop - start
start = time.time()
scores[4] = model.score(X_test,y_test)
stop = time.time()
testing_times[4] = stop - start
model1 = svm.NuSVC(kernel='rbf',decision_function_shape='ovr',break_ties=True)
start = time.time()
model1.fit(X_train,y_train)
stop = time.time()
training_times[5] = stop - start
start = time.time()
scores[5] = model1.score(X_test,y_test)
stop = time.time()
testing_times[5] = stop - start
print(training_times)
print(testing_times)
print(scores)