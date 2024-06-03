import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def startData(data):
    X = data.iloc[:,:3]  #get first three cols of data
    y = data.iloc[:,:4]  #get last col = diagnosis
    X_tr, y_tr, X_va, y_va, X_te, y_te = separateData(X,y)  #is Dataframe
    # print(type(X_tr))

def separateData(X_data, y_data):
    X_temp, X_te, y_temp, y_te = train_test_split(X_data, y_data, test_size=0.2 ,random_state=1234, shuffle=True)
    X_tr, X_va, y_tr, y_va = train_test_split(X_temp, y_temp, test_size=0.15 ,shuffle=True, random_state=1234)
    return X_tr, y_tr, X_va, y_va, X_te, y_te
    
    
