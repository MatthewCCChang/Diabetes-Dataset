import numpy as np
import pandas as pd
import neural_network as nn

def read_data():
    data = pd.read_csv('diabetic_data.csv')
    return data

def extract_data(data):
    basic_data = data[['race','age','gender','diag_1']]
    #add other stuff here for other types of data needed to extract
    return basic_data

if __name__ == '__main__':
    data = read_data()
    basic = extract_data(data)
    nn.startData(basic)