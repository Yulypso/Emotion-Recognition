#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, shutil, cv2, csv
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
import timeit
import pickle

def load_pickle(file_name):
    return pickle.load(open(file_name, 'rb'))

def load_test_dataset():
    df_temp = pd.read_csv('../Dataset/train_data.csv') 
    column_names = [str(i) for i in range(0, len(df_temp.axes[1])-1)] # 26508 columns
    df = pd.read_csv('../Dataset/test_data.csv', names=column_names)
    data = df.iloc[:, :].values
    return data 

def prediction(model, data, file_name):
    print('prediction ...')
    predictions = model.predict(data)
    with open('../Dataset/'+file_name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        row_list = []
        for index in range(len(predictions)): 
            row_list.append([predictions[index]])
        writer.writerows(row_list)
    print('writing '+file_name)

def evaluation():
    model = load_pickle('../Dataset/model.pickle')
    data = load_test_dataset()
    print(data.shape)
    prediction(model, data, 'predictions.csv')

def main():
    evaluation()
    sys.exit(0)

if __name__ == "__main__":
    main()