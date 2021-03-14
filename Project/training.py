#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, shutil, cv2, csv
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle


def get_label_name(label):
    label_list = ['neutral', 'angry', 'N.A', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    return label_list[label]


def load_train_dataset():
    df_temp = pd.read_csv('../Dataset/features_train.csv') 
    column_names = [str(i) for i in range(0, len(df_temp.axes[1])-1)] 
    column_names.append('label')
    df = pd.read_csv('../Dataset/features_train.csv', names=column_names)
    data = df.iloc[:, :-1].values
    label = df.iloc[:, -1:].values.reshape(len(df_temp.axes[0])+1,).astype(int) 
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


def save_pickle(model, file_name):
    pickle.dump(model, open(file_name, 'wb'))


def f_svm(data, label):
    '''
    SVM algorithm 
    '''
    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.7, random_state=2) 
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    unique_label = np.unique([y_test, y_pred])
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=unique_label), 
        index=['actual:{:}'.format(x) for x in unique_label], 
        columns=['pred:{:}'.format(x) for x in unique_label]
    )
    print(cmtx)
    if os.path.exists('../Dataset/model.pickle'):
        os.remove('../Dataset/model.pickle')
    save_pickle(clf, '../Dataset/model.pickle')


def f_knn(data, label):
    '''
    knn algorithm 
    '''
    # searching for the best k
    knn_scores = []
    knn_classifiers = []
    for i in range(1, 12):
        knn = KNeighborsClassifier(n_neighbors=i, algorithm='brute', n_jobs=-1)
        knn_scores.append(cross_val_score(knn, data, label, cv=10, n_jobs=-1, scoring='accuracy').mean())
        knn_classifiers.append(knn)
    knn_index = knn_scores.index(np.max(knn_scores))
    best_k = knn_scores[knn_index]
    print(f'Best k is {knn_index+1} for {np.round(knn_scores[knn_index], 2)} accuracy')

    # training with the best k then predicting
    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.7, random_state=2)
    knn_classifiers[knn_index].fit(X_train, y_train)
    y_pred = knn_classifiers[knn_index].predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    unique_label = np.unique([y_test, y_pred])
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=unique_label), 
        index=['actual:{:}'.format(x) for x in unique_label], 
        columns=['pred:{:}'.format(x) for x in unique_label]
    )
    print(cmtx)

    # display graph
    plt.title('Taux de reconnaissance en fonction du nombre de voisins K')
    plt.xlabel('Nombre de voisins K')
    plt.ylabel('Taux de reconnaissance')
    plt.plot(range(1, 12), knn_scores)
    plt.axhline(knn_scores[knn_index], color='r')
    plt.axvline(knn_index+1, color='r')
    plt.show()


def training():
    df, data, label, label_name = load_train_dataset()
    print('Dimension des donneés:', np.shape(data))

    ### Uncomment to know the number of examples per classes within our train data
    #nb_ex_per_classes = get_number_of_examples_per_classes(label)

    # Uncomment to Train the model with SVM algorithm (It creates a pickle file)
    f_svm(data, label)

    ### Uncomment to Train the model with KNN algorithm (It doesn't create any pickle file)
    #f_knn(data, label)
    

def main():
    training()
    sys.exit(0)


if __name__ == "__main__":
    main()