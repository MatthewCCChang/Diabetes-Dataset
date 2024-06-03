import numpy as np
import pandas as pd
import neural_network as nn
from sklearn.impute import SimpleImputer

def read_data():
    data = pd.read_csv('diabetic_data.csv')
    return data

def extract_data(data):
    basic_data = data[['race','age','gender','diag_1']]
    basic_data = basic_data[basic_data['diag_1'] != '?']
    
    race = basic_data[['race']]
    other = basic_data[['age','gender','diag_1']]
    
    imp = SimpleImputer(missing_values='?', strategy="constant", fill_value="other")
    clean_race = imp.fit_transform(race)
    # print(len(clean_race))
    np_data = other.to_numpy()
    # print(len(np_data))
    clean_data = np.concatenate((clean_race, np_data),axis=1)
    # print(len(clean_data))
    #add other stuff here for other types of data needed to extract
    return clean_data



if __name__ == '__main__':
    data = read_data()
    basic = extract_data(data)
    nn.startData(basic)