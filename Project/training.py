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

def get_label_name(label):
    label_list = ['neutral', 'angry', 'N.A', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    return label_list[label]


def load_train_dataset():
    df_temp = pd.read_csv('../Dataset/train_data.csv') 
    column_names = [str(i) for i in range(0, len(df_temp.axes[1])-1)] # 26508 columns
    column_names.append('label')
    df = pd.read_csv('../Dataset/train_data.csv', names=column_names)
    data = df.iloc[:, :-1].values
    label = df.iloc[:, -1:].values.reshape(len(df_temp.axes[0])+1,).astype(int) # 705 rows
    label_name = [get_label_name(label[i]) for i in range(len(label))]
    return df, data, label, label_name


def get_number_of_examples_per_classes(label):
    '''
    [463, 33, 0, 49, 16, 53, 23, 68]
    ['neutral', 'angry', 'N.A', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    '''
    nb_exemples_classes = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(label)):
        nb_exemples_classes[label[i]]+=1
    print(nb_exemples_classes)
    return nb_exemples_classes

def f_neural_network(data, label):
    '''
    Searching for the best nb of hidden neurons ...
    BORNE : Nombre de paramètres libres ≤ Nb_app x N
    - Nb_app: 705 
    - exemples N: 7 classes
    - nb param libre: (nb_features+1)C + (C+1)*nb_classes 

    A.N
    <=> (26508+1)*C + (C+1)*7 <= 705 x 7
    <=> 26509*C + 7*C + 7 <= 4935
    <=> 26516*C + 7 <= 4935
    <=> 26516*C <= 4928
    <=> C <= 0.19
    '''
    classifier = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=1, activation='tanh', solver='sgd', batch_size=1, alpha=0, learning_rate='adaptive', random_state=42))

def save_pickle(model, file_name):
    pickle.dump(model, open(file_name, 'wb'))


def f_svm(data, label):
    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.7, random_state=2) 
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    if os.path.exists('../Dataset/model.pickle'):
        os.remove('../Dataset/model.pickle')
    save_pickle(clf, '../Dataset/model.pickle')


def f_knn(data, label):
    knn_scores = []
    knn_classifiers = []
    for i in range(1, 12):
        #X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.7, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=i, algorithm='brute', n_jobs=-1)
        knn_scores.append(cross_val_score(knn, data, label, cv=10, n_jobs=-1, scoring='accuracy').mean())
        knn_classifiers.append(knn)
    knn_index = knn_scores.index(np.max(knn_scores))
    best_k = knn_scores[knn_index]
    print(f'Best k is {knn_index+1} for {np.round(knn_scores[knn_index], 2)} accuracy')

    plt.title('Taux de reconnaissance en fonction du nombre de voisins K')
    plt.xlabel('Nombre de voisins K')
    plt.ylabel('Taux de reconnaissance')
    plt.plot(range(1, 20), knn_scores)
    plt.axhline(knn_scores[knn_index], color='r')
    plt.axvline(knn_index+1, color='r')
    plt.show()
        #knn.fit(X_train, y_train)
        #print(knn.score(X_test, y_test))

def training():
    df, data, label, label_name = load_train_dataset()
    print(np.shape(data))
    f_svm(data, label)
    #nb_ex_per_classes = get_number_of_examples_per_classes(label)
    #f_knn(data, label)
    

def main():
    training()
    sys.exit(0)

if __name__ == "__main__":
    main()