#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, shutil, cv2, csv
import numpy as np
import pandas as pd
import pickle


def load_pickle(file_name):
    return pickle.load(open(file_name, 'rb'))


def load_test_dataset():
    df_temp = pd.read_csv('../Dataset/features_test.csv') 
    column_names = [str(i) for i in range(0, len(df_temp.axes[1]))] # 26508 columns
    df = pd.read_csv('../Dataset/features_test.csv', names=column_names)
    data = df.iloc[:, :].values
    return data 


def prediction(model, data, file_name):
    print('prediction ...')
    predictions = model.predict(data) # get a list of prediction of data
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
    print('Dimension des donneés:', data.shape)
    prediction(model, data, 'predictions.csv')


def main():
    evaluation()
    sys.exit(0)


if __name__ == "__main__":
    main()