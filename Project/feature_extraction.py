#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, shutil, cv2, csv
import numpy as np
import pandas as pd

def label_name(label):
    label_list = ['neutral', 'angry', 'N.A', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    return label_list[label]

def display_roi(img, df, index):
    pos_x = df.iloc[index, 1:69] 
    pos_y = df.iloc[index, 69:137]

    for i in range(len(pos_x)):
        img = cv2.circle(img, 
                        (np.round(pos_x[i]).astype('uint64'), np.round(pos_y[i]).astype('uint64')), 
                        radius=2, 
                        color=(255, 255, 255), 
                        thickness=-1)

def display_picture(df, index, path):
    img = cv2.imread(path + df['filename'][index] +'.png') 
    display_roi(img, df, index)
    if 'trainset' in path:
        cv2.imshow(df['filename'][index]+'   label '+str(df['label'][index]), img)
    elif 'testset' in path:
        cv2.imshow(df['filename'][index], img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

def x_point(value, list_pos_x):
    return np.round(list_pos_x[value]).astype('int')

def y_point(value, list_pos_y):
    return np.round(list_pos_y[value]).astype('int')

def left_eyebrow(img, list_pos_x, list_pos_y, label, dim, index):
    '''
    return cropped left eyebrow
    '''
    # departure point
    x = x_point(17, list_pos_x)
    y = y_point(20, list_pos_y)

    # length and height specification
    length = abs(x_point(27, list_pos_x) - x_point(17, list_pos_x))
    height = max([
        abs(x_point(20, list_pos_x) - x_point(19, list_pos_x)),
        abs(y_point(19, list_pos_y) - y_point(17, list_pos_y))
    ])

    crop = img[y:y+height, x:x+length].copy()
    blur = cv2.medianBlur(crop, 5)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) 
    negative = cv2.bitwise_not(gray)
    kernel = np.ones((5, 5), np.uint8)
    morphological_opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    substraction = cv2.subtract(negative, morphological_opening)
    _, otsu_threshold = cv2.threshold(substraction, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    dilatation = cv2.dilate(otsu_threshold, kernel, iterations = 1)
    blur_2 = cv2.medianBlur(dilatation, 3)
    resized = cv2.resize(blur_2, dim)

    if index != -1:
        cv2.imshow("left eyebrow: " + label_name(label), blur_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  
    return resized

def right_eyebrow(img, list_pos_x, list_pos_y, label, dim, index):
    '''
    return cropped right eyebrow
    '''
    # departure point
    x = x_point(27, list_pos_x)
    y = y_point(23, list_pos_y)

    # length and height specification
    length = abs(x_point(27, list_pos_x) - x_point(26, list_pos_x))
    height = max([
        abs(x_point(23, list_pos_x) - x_point(22, list_pos_x)),
        abs(y_point(23, list_pos_y) - y_point(26, list_pos_y))
    ])

    crop = img[y:y+height, x:x+length].copy()
    blur = cv2.medianBlur(crop, 5)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) 
    negative = cv2.bitwise_not(gray)
    kernel = np.ones((5, 5), np.uint8)
    morphological_opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    substraction = cv2.subtract(negative, morphological_opening)
    _, otsu_threshold = cv2.threshold(substraction, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    dilatation = cv2.dilate(otsu_threshold, kernel, iterations = 1)
    blur_2 = cv2.medianBlur(dilatation, 3)
    resized = cv2.resize(blur_2, dim)

    if index != -1:
        cv2.imshow("right eyebrow: " + label_name(label), resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  
    return resized

def between_eyebrow(img, list_pos_x, list_pos_y, label, dim, index):
    '''
    return cropped between eyebrow
    '''
    # departure point
    x = x_point(21, list_pos_x)
    y = min([y_point(21, list_pos_y), y_point(22, list_pos_y)])

    # length and height specification
    length = abs(x_point(22, list_pos_x) - x_point(21, list_pos_x))
    height = abs(y_point(29, list_pos_y) - y_point(27, list_pos_y))

    crop = img[y:y+height, x:x+length].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    resized = cv2.resize(laplacian, dim)

    if index != -1:
        cv2.imshow("between eyebrow: " + label_name(label), resized)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  
    return resized

def left_eye(img, list_pos_x, list_pos_y, label, dim, index):
    '''
    return cropped left eye
    '''
    # departure point
    x = x_point(36, list_pos_x)
    y = y_point(38, list_pos_y)

    # length and height specification
    length = abs(x_point(39, list_pos_x) - x_point(36, list_pos_x))
    height = abs(y_point(40, list_pos_y) - y_point(38, list_pos_y))

    crop = img[y:y+height, x:x+length].copy()
    negative = cv2.bitwise_not(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
    resized = cv2.resize(thresh, dim)

    if index != -1:
        cv2.imshow("left eye: " + label_name(label), resized)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  
    return resized

def right_eye(img, list_pos_x, list_pos_y, label, dim, index):
    '''
    return cropped right eye
    '''
    # departure point
    x = x_point(42, list_pos_x)
    y = y_point(43, list_pos_y)

    # length and height specification
    length = abs(x_point(45, list_pos_x) - x_point(42, list_pos_x))
    height = abs(y_point(47, list_pos_y) - y_point(43, list_pos_y))

    crop = img[y:y+height, x:x+length].copy()
    negative = cv2.bitwise_not(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
    resized = cv2.resize(thresh, dim)

    if index != -1:
        cv2.imshow("right eye: " + label_name(label), resized)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  
    return resized


def left_eye_area(img, list_pos_x, list_pos_y, label, dim, index):
    '''
    return cropped left eye area
    '''
    # departure point
    x = x_point(36, list_pos_x) - abs(x_point(39, list_pos_x) - x_point(36, list_pos_x))
    y = y_point(38, list_pos_y)-5

    # length and height specification
    length = abs(x_point(39, list_pos_x) - x_point(36, list_pos_x))
    height = abs(y_point(27, list_pos_y) - y_point(29, list_pos_y))

    crop = img[y:y+height, x:x+length].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    blur_2 = cv2.medianBlur(laplacian.astype('uint8'), 3)
    crop_2 = blur_2[0:int((np.shape(blur_2)[1])/2), 0:np.shape(blur_2)[0]].copy()
    negative = cv2.bitwise_not(crop_2)
    resized = cv2.resize(negative, dim)

    if index != -1:
        cv2.imshow("left eye area: " + label_name(label), resized)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  
    return resized

def right_eye_area(img, list_pos_x, list_pos_y, label, dim, index):
    '''
    return cropped right eye area
    '''
    # departure point
    x = x_point(45, list_pos_x)
    y = y_point(43, list_pos_y)-5

    # length and height specification
    length = abs(x_point(45, list_pos_x) - x_point(42, list_pos_x))
    height = abs(y_point(27, list_pos_y) - y_point(29, list_pos_y))

    crop = img[y:y+height, x:x+length].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    blur_2 = cv2.medianBlur(laplacian.astype('uint8'), 3)
    crop_2 = blur_2[0:int((np.shape(blur_2)[1])/2), 0:np.shape(blur_2)[0]].copy()
    negative = cv2.bitwise_not(crop_2)
    resized = cv2.resize(negative, dim)
    
    if index != -1:
        cv2.imshow("right eye area: " + label_name(label), resized)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  
    return resized

def nose(img, list_pos_x, list_pos_y, label, dim, index):
    '''
    return cropped nose
    '''
    # departure point
    x = x_point(31, list_pos_x)-5
    y = y_point(28, list_pos_y)

    # length and height specification
    length = abs(x_point(35, list_pos_x) - x_point(31, list_pos_x))+5
    height = abs(y_point(28, list_pos_y) - y_point(33, list_pos_y))+5

    crop = img[y:y+height, x:x+length].copy()
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8)) # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  
    l2 = clahe.apply(l)  # apply CLAHE to the L channel
    lab = cv2.merge((l2,a,b))  # merge channels
    contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB color space to BGR
    gray = cv2.cvtColor(contrast, cv2.COLOR_RGB2GRAY) 
    resized = cv2.resize(gray, dim, interpolation=cv2.INTER_CUBIC)
    resized_2 = cv2.resize(resized, dim, interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(resized_2,kernel,iterations = 1)
    blur = cv2.medianBlur(erosion.astype('uint8'), 5)
    _, otsu_threshold = cv2.threshold(blur.astype('uint8'), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    negative = cv2.bitwise_not(otsu_threshold)
    dilation = cv2.dilate(resized_2,kernel,iterations = 1)
    resized = cv2.resize(dilation, dim)

    if index != -1:
        cv2.imshow("nose: " + label_name(label), dilation)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  
    return resized

def nose_left(img, list_pos_x, list_pos_y, label, dim, index):
    # departure point
    x = x_point(41, list_pos_x)
    y = y_point(29, list_pos_y)

    # length and height specification
    length = abs(x_point(41, list_pos_x) - x_point(29, list_pos_x))
    height = abs(y_point(27, list_pos_y) - y_point(30, list_pos_y))

    crop = img[y:y+height, x:x+length].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) 
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    blur = cv2.medianBlur(opening.astype('uint8'), 5)
    negative = cv2.bitwise_not(blur)
    resized = cv2.resize(negative, dim)

    if index != -1:
        cv2.imshow("nose left: " + label_name(label), resized)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  
    return resized

def nose_right(img, list_pos_x, list_pos_y, label, dim, index):
    # departure point
    x = x_point(29, list_pos_x)
    y = y_point(29, list_pos_y)

    # length and height specification
    length = abs(x_point(41, list_pos_x) - x_point(29, list_pos_x))
    height = abs(y_point(27, list_pos_y) - y_point(30, list_pos_y))

    crop = img[y:y+height, x:x+length].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) 
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    blur = cv2.medianBlur(opening.astype('uint8'), 5)
    negative = cv2.bitwise_not(blur)
    resized = cv2.resize(negative, dim)

    if index != -1:
        cv2.imshow("nose right: " + label_name(label), resized)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  
    return resized

def mouth(img, list_pos_x, list_pos_y, label, dim, index):
    '''
    return cropped mouth
    '''
    # departure point
    x = x_point(41, list_pos_x)
    y = min([
        y_point(48, list_pos_y), 
        y_point(51, list_pos_y), 
        y_point(54, list_pos_y)
    ])-5

    # length and height specification
    length = abs(x_point(46, list_pos_x) - x_point(41, list_pos_x))
    height = abs(
        max([
            y_point(48, list_pos_y),
            y_point(54, list_pos_y),
            y_point(57, list_pos_y)
        ]) -
        min([
            y_point(48, list_pos_y),
            y_point(51, list_pos_y),
            y_point(54, list_pos_y)
        ])
    )+15

    crop = img[y:y+height, x:x+length].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) 
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    blur = cv2.medianBlur(opening.astype('uint8'), 5)
    negative = cv2.bitwise_not(blur)
    resized = cv2.resize(negative, dim)

    if index != -1:
        cv2.imshow("mouth: " + label_name(label), resized)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  
    return resized


def roi_extraction(df, index=-1, file_name='', nb_examples=0):
    '''
    1. Displays ROI feature extracted at image[index] if index != -1
    2. Save ROI features extracted into CSV file if index == -1
    '''   

    ### For displaying features extracted ###
    if index != -1:
        if file_name == 'train_data.csv':
            img = cv2.imread('../Dataset/trainset/'+ df['filename'][index] +'.png') 
            print('feature extraction for '+df['filename'][index]+'.png, '+'index: '+str(index))
            list_pos_x = df.iloc[index, 1:69] 
            list_pos_y = df.iloc[index, 69:137]
            label = df.iloc[index, 137]    
        elif file_name == 'test_data.csv':
            img = cv2.imread('../Dataset/testset/'+ df['filename'][index] +'.png') 
            print('feature extraction for '+df['filename'][index]+'.png, '+'index: '+str(index))
            list_pos_x = df.iloc[index, 1:69] 
            list_pos_y = df.iloc[index, 69:137]
            label = 0
        
        ROI_1 = left_eyebrow(img, list_pos_x, list_pos_y, label, (94, 20), index)
        ROI_2 = right_eyebrow(img, list_pos_x, list_pos_y, label, (94, 20), index)
        ROI_3 = between_eyebrow(img, list_pos_x, list_pos_y, label, (44, 30), index)
        ROI_4 = left_eye(img, list_pos_x, list_pos_y, label, (40, 12), index)
        ROI_5 = right_eye(img, list_pos_x, list_pos_y, label, (40, 12), index)
        ROI_6 = right_eye_area(img, list_pos_x, list_pos_y, label, (40, 24), index)
        ROI_7 = left_eye_area(img, list_pos_x, list_pos_y, label, (40, 24), index)
        ROI_8 = nose(img, list_pos_x, list_pos_y, label, (46, 60), index)
        ROI_9 = nose_left(img, list_pos_x, list_pos_y, label, (62, 68), index)
        ROI_10 = nose_right(img, list_pos_x, list_pos_y, label, (62, 68), index)
        ROI_11 = mouth(img, list_pos_x, list_pos_y, label, (118, 50), index)

    ### Extracting features for training step ###
    else:
        with open('../Dataset/'+file_name, 'w', newline='') as file:
            print('feature extraction...')
            writer = csv.writer(file, delimiter=',')
            row_list = []

            for index in range(0, nb_examples): 
                ### features from trainset ###
                if file_name == 'features_train.csv':
                    if index not in [4, 51, 52, 206, 207, 208, 209, 210, 211, 212, 213, 214, 340, 341, 342, 343, 344]:
                        # excluded image list (17 images)
                        # 4: landmarks not appropriated
                        # 51, 52: hair on face
                        # 206, 207, 208, 209, 210, 211, 212, 213, 214: hair on face
                        # 340, 341, 342, 343, 344: hair on face
                        img = cv2.imread('../Dataset/trainset/'+ df['filename'][index] +'.png') 
                        list_pos_x = df.iloc[index, 1:69] 
                        list_pos_y = df.iloc[index, 69:137]
                        label = df.iloc[index, 137]  

                        ROI_1 = left_eyebrow(img, list_pos_x, list_pos_y, label, (94, 20), -1)
                        ROI_2 = right_eyebrow(img, list_pos_x, list_pos_y, label, (94, 20), -1)
                        ROI_3 = between_eyebrow(img, list_pos_x, list_pos_y, label, (44, 30), -1)
                        ROI_4 = left_eye(img, list_pos_x, list_pos_y, label, (40, 12), -1)
                        ROI_5 = right_eye(img, list_pos_x, list_pos_y, label, (40, 12), -1)
                        ROI_6 = right_eye_area(img, list_pos_x, list_pos_y, label, (40, 24), -1)
                        ROI_7 = left_eye_area(img, list_pos_x, list_pos_y, label, (40, 24), -1)
                        ROI_8 = nose(img, list_pos_x, list_pos_y, label, (46, 60), -1)
                        ROI_9 = nose_left(img, list_pos_x, list_pos_y, label, (62, 68), -1)
                        ROI_10 = nose_right(img, list_pos_x, list_pos_y, label, (62, 68), -1)
                        ROI_11 = mouth(img, list_pos_x, list_pos_y, label, (118, 50), -1)

                        row = np.concatenate([
                            ROI_1, ROI_2, ROI_3, 
                            ROI_4, ROI_5, ROI_6,
                            ROI_7, ROI_9, ROI_9,
                            ROI_10, ROI_11, int(label)
                        ], axis=None)

                        row_list.append(row)

                ### features from testset ###
                elif file_name == 'features_test.csv':
                    img = cv2.imread('../Dataset/testset/'+ df['filename'][index] +'.png') 
                    list_pos_x = df.iloc[index, 1:69] 
                    list_pos_y = df.iloc[index, 69:137]
                    label = None

                    ROI_1 = left_eyebrow(img, list_pos_x, list_pos_y, label, (94, 20), -1)
                    ROI_2 = right_eyebrow(img, list_pos_x, list_pos_y, label, (94, 20), -1)
                    ROI_3 = between_eyebrow(img, list_pos_x, list_pos_y, label, (44, 30), -1)
                    ROI_4 = left_eye(img, list_pos_x, list_pos_y, label, (40, 12), -1)
                    ROI_5 = right_eye(img, list_pos_x, list_pos_y, label, (40, 12), -1)
                    ROI_6 = right_eye_area(img, list_pos_x, list_pos_y, label, (40, 24), -1)
                    ROI_7 = left_eye_area(img, list_pos_x, list_pos_y, label, (40, 24), -1)
                    ROI_8 = nose(img, list_pos_x, list_pos_y, label, (46, 60), -1)
                    ROI_9 = nose_left(img, list_pos_x, list_pos_y, label, (62, 68), -1)
                    ROI_10 = nose_right(img, list_pos_x, list_pos_y, label, (62, 68), -1)
                    ROI_11 = mouth(img, list_pos_x, list_pos_y, label, (118, 50), -1)
                
                    # flatten each ROI_n images
                    row = np.concatenate([
                        ROI_1, ROI_2, ROI_3, 
                        ROI_4, ROI_5, ROI_6,
                        ROI_7, ROI_9, ROI_9,
                        ROI_10, ROI_11
                    ], axis=None)

                    # for each flattened data, append to a list
                    row_list.append(row)

            # write every rows containing flattened data into the csv file
            writer.writerows(row_list)
            print(file_name + ' created')
            

def read_data():
    '''
    read trainset.csv and testset.csv to get ROI coordinates
    return dataframe for trainset and testset
    '''
    df_train = pd.read_csv('../Dataset/trainset/trainset.csv', encoding='utf-8')
    df_test = pd.read_csv('../Dataset/testset/testset.csv', encoding='utf-8')
    return df_train, df_test

    
def feature_extraction():
    df_train, df_test = read_data()

    ### set value to -1 for features extractions before training ###
    ### set value to image number to display it and its extracted features ###
    value = -1

    ### uncomment to display the picture with ROI from Trainset (value must be different from -1) ###
    #display_picture(df_train, value, '../Dataset/trainset/')

    ### uncomment to display the picture with ROI from Testset (value must be different from -1) ###
    #display_picture(df_test, value, '../Dataset/testset/')

    ### feature extractions if value == -1, else it shows features extracted of the image[value] ###
    roi_extraction(df_train, value, 'features_train.csv', np.shape(df_train)[0]) 

    ### comment if value is different from -1 ###
    roi_extraction(df_test, value, 'features_test.csv', np.shape(df_test)[0]) 
    

def main():
    feature_extraction()
    sys.exit(0)


if __name__ == "__main__":
    main()