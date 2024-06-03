import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

diabetes_code = [2535,3572,5881,64800,64801,64802,64803,64804,7751,"V1221"]

def startData(data):  #input is df
    # data = data.to_numpy()
    # print(type(data))
    # print(len(data))
    # valid_idx = y.__index__
    X = preprocess_data(data[:,:3])  #get first three cols of data
    y = preprocess_label(data[:,3])  #get last col = diagnosis
    # print(type(y))    #ndarray
    # print(len(X))
    # print(len(y))
    X_tr, y_tr, X_va, y_va, X_te, y_te = separateData(X,y)  #
    print(type(X_tr))

def separateData(X_data, y_data):
    X_temp, X_te, y_temp, y_te = train_test_split(X_data, y_data, test_size=0.2 ,random_state=1234, shuffle=True)
    X_tr, X_va, y_tr, y_va = train_test_split(X_temp, y_temp, test_size=0.15 ,shuffle=True, random_state=1234)
    return X_tr, y_tr, X_va, y_va, X_te, y_te
    
def preprocess_data(data):
    # data.replace('?',"other",inplace=True)
    encoder = OneHotEncoder(sparse_output=False)
    encoded_columns = [encoder.fit_transform(data[:, [i]]) for i in range(data.shape[1])]
    encoded_data = np.hstack(encoded_columns)
    return encoded_data

def preprocess_label(labels):
    # labels = labels[labels != '?']
    lb = LabelBinarizer()
    binary_label = lb.fit_transform(labels.reshape(-1,1))
    return binary_label
