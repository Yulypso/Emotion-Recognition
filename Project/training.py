#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, shutil, cv2, csv
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.image as mpimg
import timeit

def load_dataset():
    df_temp = pd.read_csv('../Dataset/train_data.csv') 
    column_names = [str(i) for i in range(0, len(df_temp.axes[1])-1)] # 26508 columns
    column_names.append('label')
    df = pd.read_csv('../Dataset/train_data.csv', names=column_names)
    data = df.iloc[:, :-1].values
    label = df.iloc[:, -1:].values.reshape(len(df_temp.axes[0])+1,) # 705 rows
    return df, data, label

def training():
    df, data, label = load_dataset()

def main():
    training()
    sys.exit(0)

if __name__ == "__main__":
    main()