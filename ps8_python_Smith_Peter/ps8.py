import numpy as np
import scipy as sc
from sklearn import neighbors as nb
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import sklearn.multiclass as mc
import matplotlib.pyplot as plt
import cv2 as cv
from collections import Counter
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from kmeans_multiple import kmeans_multiple
from segment_kmeans import segment_kmeans
from kmeans_single import kmeans_single

# problem_1_data = sc.io.loadmat('./input/hw8_data1.mat')
# X = problem_1_data["X"]
# y = problem_1_data["y"]
# IDX = np.arange(5000)
# np.random.shuffle(IDX)

# pics = X[IDX[:25],:]
# fig, axs = plt.subplots(5,5)
# for i in range(5):
#     for j in range(5):
#         axs[i,j].imshow(np.transpose(np.reshape(pics[i*5 + j,:],(20,20))),'gray')
#         axs[i,j].grid(False)
#         axs[i,j].axis('off')
# plt.show()
# X_test = X[IDX[:500],:]
# y_test = y[IDX[:500],:]
# X_1 = X[IDX[500:1400],:]
# y_1 = y[IDX[500:1400],:]
# X_2 = X[IDX[1400:2300],:]
# y_2 = y[IDX[1400:2300],:]
# X_3 = X[IDX[2300:3200],:]
# y_3 = y[IDX[2300:3200],:]
# X_4 = X[IDX[3200:4100],:]
# y_4 = y[IDX[3200:4100],:]
# X_5 = X[IDX[4100:5000],:]
# y_5 = y[IDX[4100:5000],:]
# X_1_dic = {"X_1":X_1,"y_1":y_1}
# X_test_dic = {"X_test":X_test,"y_test":y_test}
# X_2_dic = {"X_2":X_2,"y_2":y_2}
# X_3_dic = {"X_3":X_3,"y_3":y_3}
# X_4_dic = {"X_4":X_4,"y_4":y_4}
# X_5_dic = {"X_5":X_5,"y_5":y_5}
# sc.io.savemat("./input/X_test.mat",X_test_dic)
# sc.io.savemat("./input/X_1.mat",X_1_dic)
# sc.io.savemat("./input/X_2.mat",X_2_dic)
# sc.io.savemat("./input/X_3.mat",X_3_dic)
# sc.io.savemat("./input/X_4.mat",X_4_dic)
# sc.io.savemat("./input/X_5.mat",X_5_dic)
problem_1_data = sc.io.loadmat('./input/X_test.mat')
X_test = problem_1_data["X_test"]
y_test = problem_1_data["y_test"]
problem_1_data = sc.io.loadmat('./input/X_1.mat')
X_1 = problem_1_data["X_1"]
y_1 = problem_1_data["y_1"]
problem_1_data = sc.io.loadmat('./input/X_2.mat')
X_2 = problem_1_data["X_2"]
y_2 = problem_1_data["y_2"]
problem_1_data = sc.io.loadmat('./input/X_3.mat')
X_3 = problem_1_data["X_3"]
y_3 = problem_1_data["y_3"]
problem_1_data = sc.io.loadmat('./input/X_4.mat')
X_4 = problem_1_data["X_4"]
y_4 = problem_1_data["y_4"]
problem_1_data = sc.io.loadmat('./input/X_5.mat')
X_5 = problem_1_data["X_5"]
y_5 = problem_1_data["y_5"]
model1 = svm.NuSVC(kernel='poly',decision_function_shape='ovo',degree=3,break_ties=False)
model1.fit(X_1,np.ravel(y_1))
scores = np.zeros(6)
scores[0] = model1.score(X_1,np.ravel(y_1))
scores[1] = model1.score(X_2,np.ravel(y_2))
scores[2] = model1.score(X_3,np.ravel(y_3))
scores[3] = model1.score(X_4,np.ravel(y_4))
scores[4] = model1.score(X_5,np.ravel(y_5))
scores[5] = model1.score(X_test,np.ravel(y_test))
print(scores)
model2 = nb.KNeighborsClassifier(3)
model2.fit(X_2,np.ravel(y_2))
scores2 = np.zeros(6)
scores2[0] = model2.score(X_1,np.ravel(y_1))
scores2[1] = model2.score(X_2,np.ravel(y_2))
scores2[2] = model2.score(X_3,np.ravel(y_3))
scores2[3] = model2.score(X_4,np.ravel(y_4))
scores2[4] = model2.score(X_5,np.ravel(y_5))
scores2[5] = model2.score(X_test,np.ravel(y_test))
print(scores2)
model3 = LogisticRegression(max_iter=250)
model3.fit(X_3,np.ravel(y_3))
scores3 = np.zeros(6)
scores3[0] = model3.score(X_1,np.ravel(y_1))
scores3[1] = model3.score(X_2,np.ravel(y_2))
scores3[2] = model3.score(X_3,np.ravel(y_3))
scores3[3] = model3.score(X_4,np.ravel(y_4))
scores3[4] = model3.score(X_5,np.ravel(y_5))
scores3[5] = model3.score(X_test,np.ravel(y_test))
print(scores3)
model4 = DecisionTreeClassifier()
model4.fit(X_4,np.ravel(y_4))
scores4 = np.zeros(6)
scores4[0] = model4.score(X_1,np.ravel(y_1))
scores4[1] = model4.score(X_2,np.ravel(y_2))
scores4[2] = model4.score(X_3,np.ravel(y_3))
scores4[3] = model4.score(X_4,np.ravel(y_4))
scores4[4] = model4.score(X_5,np.ravel(y_5))
scores4[5] = model4.score(X_test,np.ravel(y_test))
print(scores4)
model5 = BaggingClassifier(n_estimators=60,max_samples=100,max_features=250,bootstrap_features=True)
model5.fit(X_5,np.ravel(y_5))
scores5 = np.zeros(6)
scores5[0] = model5.score(X_1,np.ravel(y_1))
scores5[1] = model5.score(X_2,np.ravel(y_2))
scores5[2] = model5.score(X_3,np.ravel(y_3))
scores5[3] = model5.score(X_4,np.ravel(y_4))
scores5[4] = model5.score(X_5,np.ravel(y_5))
scores5[5] = model5.score(X_test,np.ravel(y_test))
print(scores5)
predict = model5.predict(X_test)
predictions = np.zeros((500,5))
predictions[:,0] = model1.predict(X_test)
predictions[:,1] = model2.predict(X_test)
predictions[:,2] = model3.predict(X_test)
predictions[:,3] = model4.predict(X_test)
predictions[:,4] = model5.predict(X_test)
majority_vote = np.zeros(500,dtype=np.uint8)
for i in range(500):
    l = list(predictions[i,:])
    count = Counter(l)
    majority_vote[i] = count.most_common(1)[0][0]
correct_predicions = 0.0
for i in range(500):
    if majority_vote[i] == y_test[i]:
        correct_predicions += 1.0/500.0
print(correct_predicions)

# img = cv.imread("./Input/im3.png")
# new_image = segment_kmeans(img,3,7,5)
# cv.imwrite('./Output/img3_3_7_5.png',new_image)
# img = cv.imread("./Input/im3.png")
# new_image = segment_kmeans(img,5,7,5)
# cv.imwrite('./Output/img3_5_7_5.png',new_image)
# img = cv.imread("./Input/im3.png")
# new_image = segment_kmeans(img,7,7,5)
# cv.imwrite('./Output/img3_7_7_5.png',new_image)
# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,5)
# cv.imwrite('./Output/img3_5_15_5.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,5)
# cv.imwrite('./Output/img3_7_15_5.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,5)
# cv.imwrite('./Output/img3_3_30_5.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,5)
# cv.imwrite('./Output/img3_5_30_5.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,5)
# cv.imwrite('./Output/img3_7_30_5.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,15)
# cv.imwrite('./Output/img3_3_7_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,15)
# cv.imwrite('./Output/img3_5_7_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,15)
# cv.imwrite('./Output/img3_7_7_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,15)
# cv.imwrite('./Output/img3_3_15_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,15)
# cv.imwrite('./Output/img3_5_15_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,15)
# cv.imwrite('./Output/img3_7_15_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,15)
# cv.imwrite('./Output/img3_3_30_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,15)
# cv.imwrite('./Output/img3_5_30_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,15)
# cv.imwrite('./Output/img3_7_30_15.png',new_image)


# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,20)
# cv.imwrite('./Output/img3_3_7_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,20)
# cv.imwrite('./Output/img3_5_7_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,20)
# cv.imwrite('./Output/img3_7_7_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,20)
# cv.imwrite('./Output/img3_3_15_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,20)
# cv.imwrite('./Output/img3_5_15_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,20)
# cv.imwrite('./Output/img3_7_15_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,20)
# cv.imwrite('./Output/img3_3_30_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,20)
# cv.imwrite('./Output/img3_5_30_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,20)
# cv.imwrite('./Output/img3_7_30_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,5)
# cv.imwrite('./Output/img2_3_7_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,5)
# cv.imwrite('./Output/img2_5_2_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,5)
# cv.imwrite('./Output/img2_7_2_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,5)
# cv.imwrite('./Output/img2_3_15_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,5)
# cv.imwrite('./Output/img2_5_15_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,5)
# cv.imwrite('./Output/img2_7_15_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,5)
# cv.imwrite('./Output/img2_3_30_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,5)
# cv.imwrite('./Output/img2_5_30_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,5)
# cv.imwrite('./Output/img2_7_30_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,15)
# cv.imwrite('./Output/img2_3_7_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,15)
# cv.imwrite('./Output/img2_5_7_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,15)
# cv.imwrite('./Output/img2_7_7_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,15)
# cv.imwrite('./Output/img2_3_15_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,15)
# cv.imwrite('./Output/img2_5_15_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,15)
# cv.imwrite('./Output/img2_7_15_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,15)
# cv.imwrite('./Output/img2_3_30_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,15)
# cv.imwrite('./Output/img2_5_30_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,15)
# cv.imwrite('./Output/img2_7_30_15.png',new_image)


# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,20)
# cv.imwrite('./Output/img2_3_7_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,20)
# cv.imwrite('./Output/img2_5_7_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,20)
# cv.imwrite('./Output/img2_7_7_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,20)
# cv.imwrite('./Output/img2_3_15_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,20)
# cv.imwrite('./Output/img2_5_15_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,20)
# cv.imwrite('./Output/img2_7_15_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,20)
# cv.imwrite('./Output/img2_3_30_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,20)
# cv.imwrite('./Output/img2_5_30_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,20)
# cv.imwrite('./Output/img2_7_30_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,5)
# cv.imwrite('./Output/img1_3_7_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,5)
# cv.imwrite('./Output/img1_5_2_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,5)
# cv.imwrite('./Output/img1_7_2_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,5)
# cv.imwrite('./Output/img1_3_15_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,5)
# cv.imwrite('./Output/img1_5_15_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,5)
# cv.imwrite('./Output/img1_7_15_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,5)
# cv.imwrite('./Output/img1_3_30_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,5)
# cv.imwrite('./Output/img1_5_30_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,5)
# cv.imwrite('./Output/img1_7_30_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,15)
# cv.imwrite('./Output/img1_3_7_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,15)
# cv.imwrite('./Output/img1_5_7_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,15)
# cv.imwrite('./Output/img1_7_7_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,15)
# cv.imwrite('./Output/img1_3_15_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,15)
# cv.imwrite('./Output/img1_5_15_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,15)
# cv.imwrite('./Output/img1_7_15_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,15)
# cv.imwrite('./Output/img1_3_30_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,15)
# cv.imwrite('./Output/img1_5_30_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,15)
# cv.imwrite('./Output/img1_7_30_15.png',new_image)


# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,20)
# cv.imwrite('./Output/img1_3_7_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,20)
# cv.imwrite('./Output/img1_5_7_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,20)
# cv.imwrite('./Output/img1_7_7_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,20)
# cv.imwrite('./Output/img1_3_15_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,20)
# cv.imwrite('./Output/img1_5_15_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,20)
# cv.imwrite('./Output/img1_7_15_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,20)
# cv.imwrite('./Output/img1_3_30_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,20)
# cv.imwrite('./Output/img1_5_30_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,20)
# cv.imwrite('./Output/img1_7_30_20.png',new_image)