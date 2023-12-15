import numpy as np
from sklearn.linear_model import LogisticRegression

def logReg_multi(X_train,y_train,X_test):
    unique_class = np.unique(y_train)
    print(unique_class)
    prediction = np.zeros((len(unique_class),len(X_test)))
    for i ,classes in np.ndenumerate(unique_class):
        new_y = np.zeros((len(y_train),1))
        for idx, sample in np.ndenumerate(y_train):
            if sample == classes:
                new_y[idx] = 1
        np.reshape(new_y,(len(new_y)))
        mdl = LogisticRegression(random_state=0).fit(X_train,new_y)
        
        prob = mdl.predict_proba(X_test)
        prediction[i,:] = prob[:,1]
        
    predictions = np.argmax(prediction,axis=0)
    y_predict = np.zeros((len(X_test),1))
    for index , i in np.ndenumerate(predictions):
        y_predict[index] = unique_class[i]
    return y_predict