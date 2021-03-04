#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, shutil, cv2
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

def display_roi(img, df, index):
    pos_x = df.iloc[index, 1:69] # [0->67] 
    pos_y = df.iloc[index, 69:137]

    for i in range(len(pos_x)):
        img = cv2.circle(img, 
                        (np.round(pos_x[i]).astype('uint64'), np.round(pos_y[i]).astype('uint64')), 
                        radius=2, 
                        color=(255, 255, 255), 
                        thickness=-1)

def display_picture(df, index):
    img = cv2.imread('../Dataset/trainset/'+ df['filename'][index] +'.png') 
    display_roi(img, df, index)
    cv2.imshow(df['filename'][index]+'   label '+str(df['label'][index]), img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

def roi_extraction(df, index):
    '''
    extract chosen roi
    1. pour chaque ligne (images) 
    2. on extrait les ROI choisi (features)
    3. on ajoute dans un tableau une ligne pour l'image avec en colonne les images cropped
    4. on ajoute dans un tableau le label de l'image
    [image 0]: f1, f2, ..., fn
    [image 0]: 4 #label
    '''   
    img = cv2.imread('../Dataset/trainset/'+ df['filename'][index] +'.png') 
    pos_x = df.iloc[index, 1:69] # [0->67] 
    pos_y = df.iloc[index, 69:137]
    label = df.iloc[index, 137]
    chosen_roi = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # choose points here

    image_data = [] # ROI extracted for the current image 64x64 images
    for i in chosen_roi:
        y = np.round(pos_y[chosen_roi[i]]).astype('int')
        x = np.round(pos_x[chosen_roi[i]]).astype('int')
        crop_img = img[y-32:y+32, x-32:x+32].copy() # (y, x) and not (x, y)
        cv2.imshow("landmark "+str(i), crop_img)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  

def read_data():
    df = pd.read_csv('../Dataset/trainset/trainset.csv', encoding='utf-8')
    display_picture(df, 715)
    roi_extraction(df, 715)
    
def feature_extraction():
    read_data()

def main():
    feature_extraction()
    sys.exit(0)

if __name__ == "__main__":
    main()