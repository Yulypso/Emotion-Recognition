# Emotion-Recognition

## Author

[![Linkedin: Thierry Khamphousone](https://img.shields.io/badge/-Thierry_Khamphousone-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/tkhamphousone/)](https://www.linkedin.com/in/tkhamphousone)

## Getting Started

__Setup__
```bash
> git clone https://github.com/Yulypso/Emotion-Recognition.git
> cd Emotion-Recognition
> python3 -m venv .venv

# for MacOs/Linux
> source .venv/bin/activate

#for Windows
> py -3 -m venv .venv
> .venv\scripts\activate

# to install requirements 
> pip3 install -r requirements.txt
```

__[Check Dependency Graph](https://github.com/Yulypso/Emotion-Recognition/network/dependencies)__

<br/>

__Note__: In Visual Studio Code, don't forget to select the correct Python interpreter. <br/>

[CMD + SHIFT + P] > select Interpreter > Python 3.9.0 64-bits ('.venv') [./.venv/bin/python]

<br/>

__Run the code__
```bash
> cd project/bitmapProcessing

# feature extractions
> python3 feature_extraction.py

# model training
> python3 training.py

# class prediction
> python3 eval.py
```

__Stop the code__
```bash
#don't forget to deactivate the virtual environement (.venv)
> deactivate
```

<br/>

## Table of Contents

- __[Introduction](#introduction)__
- __[Data analysis](#data-analysis)__
- __[Feature descriptions](#feature-descriptions)__
- __[Extraction method](#extraction-method)__
- __[Preprocessing and Segmentation](#preprocessing---segmentation)__
- __[K-Nearest Neighbors](#k-nearest-neighbors)__
- __[Support Vector Machine](#support-vector-machine)__
- __[Results obtained](#results-obtained)__
- __[Conclusion](#conclusion)__
- __[Biblography](#bibliography)__


<br/>

## Introduction

The project carried out relates to the recognition of emotions by computer which is a machine learning or machine learning theme that could be useful, particularly in the commercial field where the analysis of a customer's emotions would allow the improvement of the services offered. by a company. Within education, the recognition of emotions makes it possible to recognize pupils and students within a class who have not understood the concept taught by the teacher and to refer them subsequently to additional help.

Recognition of emotions can be achieved through facial and vocal expression, as well as through body language. The work carried out focuses only on the analysis of facial expressions where several images of different people express an emotion. These images are grouped together in a database.

An emotion is defined as a "sudden turmoil, transient agitation caused by a keen feeling of fear, surprise, joy, etc." " (Larousse)
For the project, these feelings are classified into seven categories, namely "joy", "anger", "disgust", "sadness", "fear", "surprise" and "emotion. neutral ”.

The analysis of images or video streams generally follow a pipeline in a machine learning approach:

1. Image collecting
2. Face detection and feature point placement
3. Feature extractions
4. Feature preprocessing (image processing)
5. Training the model
6. Classifications and predictions

<br/>

## Data analysis

The first step of the project is to analyze the data with images of faces as well as points of interest contained in a CSV file.
So I first decided to develop a display of the images with their characteristic points. **(Figure 1)**

<br/>

<p align="center" width="100%">
    <img align="center" width="880" src="https://user-images.githubusercontent.com/59794336/110991209-a275b480-8374-11eb-9eb6-a0d4632cf786.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 1</strong> - Images of faces representing an emotion with the points of interest represented
    by white dots. (a) joy, (b) neutral emotion
</p>

<br/>

I noticed that some images were not suitable for machine learning because the feature points did not match the face or because a strand of hair could interfere with the preprocessing of the extracted images / features. **(Figure 2)**

<br/>

<p align="center" width="100%">
    <img align="center" width="880" src="https://user-images.githubusercontent.com/59794336/110991514-05674b80-8375-11eb-9bce-22471929636c.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 2</strong> - Face images not suitable for machine learning
</p>

<br/>

## Feature descriptions

After filtering out the images that were not fit for analysis, I added them to a list of excluded images within my program so that they would not be taken into account when extracting features.

__The filtered database has 705 images, or a sample size of 705.__

In order to retrieve the characteristics of the emotions on the faces, we can ask ourselves the question "Where are the most prominent characteristic features of an emotion on a face?".

I defined **11 areas of the face** that seemed to me to be the most prominent, with the most variations depending on the emotions. __(Figure 3)__

<br/>

<p align="center" width="100%">
    <img align="center" width="880" src="https://user-images.githubusercontent.com/59794336/110991831-68f17900-8375-11eb-99a6-1cc5139c4ef7.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 3</strong> - Representation of the 11 areas of interest chosen for feature extraction
</p>

**The classes carry the following labels:**
- "Joy" (5)
- "anger" (1)
- "disgust" (3)
- "sadness" (6)
- "fear" (4)
- "surprise" (7)
- "neutral emotion" (0)

Due to lack of data, we will not work on the emotion (2)

<br/>

**Some explanations :**

⇒ The areas around the eyes, the shape of the mouth as well as the areas around the
noses allow to recognize **joy**.

⇒ The area between the eyebrows can detect **anger**.

⇒ The whites of the eyes, The shape of the nose and that of the mouth make it possible to detect **surprise**.

⇒ The shape of the eyebrows, the mouth, the eyes as well as the area between the eyebrows make it possible to detect **sadness**.

Each of the 11 areas will be cropped, preprocessed, resized and extracted as "features" for machine learning. (**Figure 4**)

<br/>

<p align="center" width="100%">
    <img align="center" width="480" src="https://user-images.githubusercontent.com/59794336/110992304-08167080-8376-11eb-8968-5fd87786854c.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 4</strong> - image resizing size chart
</p>

<br/>

Each feature is an image that we will "flatten" and obtain a table at one
dimension.

**The feature size is 26508 (flattened) for an individual in the sample.**

<br/>

## Extraction method

The position of the characteristic points generated automatically, depend on the position of the face in the photo as well as on its shape.

On a neutral face centered in the image, the characteristic points look like **Figure 5**.

<br/>

<p align="center" width="100%">
    <img align="center" width="380" src="https://user-images.githubusercontent.com/59794336/110992596-680d1700-8376-11eb-9431-2fcb08bdd5f3.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 5</strong> - Characteristic points (landmarks) of the face
</p>

<br/>

<p align="center" width="100%">
    <img align="center" width="680" src="https://user-images.githubusercontent.com/59794336/110992693-90951100-8376-11eb-9d5e-6ca0cb1e740f.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 6</strong> - Table of coordinates and sizes of the areas of the face to be extracted
</p>

<br/>

The coordinates of the starting point, the length and the height of the cropped images are grouped in the table. (**Figure 6**)

Since coordinates and distances depend on feature points and not on fixed pixel values in the image, we can easily retrieve facial components regardless of their position in the image. (**Figure 7**)

<br/>

<p align="center" width="100%">
    <img align="center" width="680" src="https://user-images.githubusercontent.com/59794336/110993012-0dc08600-8377-11eb-8393-f785afa12c07.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 7</strong> - Facial components, (a) left eyebrow, (b) between eyebrows, (c) right eyebrow, (d) left eye side, (e) left eye, (f) right eye, (g) right eye side, (h ) nose left side, (i) nose, (j) nose right side, (k) mouth
</p>
 
<br/>

## Preprocessing - Segmentation

Each of the extracted images was transformed at the segmentation stage.

**Processing steps:**

⇒ **Left eyebrow and right eyebrow:**
- Median blur kernel 5x5
- Convert to grayscale (1 color channel)
- Negative
- Morphological opening (dilation + erosion) kernel 5x5 
- Substraction (negative, morphological opening)
- Otsu thresholding
- 3x3 kernel dilation
- Median blur kernel 3x3
- Resize
  
⇒ **Between the eyebrows:**
- Convert to grayscale (1 color channel) o Gaussian blur kernel 7x7
- Laplacian
- Resize

⇒ **Left eye and Right eye:**
- Negative
- Convert to grayscale (1 color channel) o Adaptive Threshold
- Resize
  
⇒ **Left eye side and Right eye side:**
- Convert to grayscale (1 color channel) o Gaussian blur kernel 7x7
- Laplacian
- Median blur kernel 3x3
- Negative
- Resize

⇒ **Nose:**
- CLAHE (Contrast Limited Adaptive Histogram Equalization) o Convert to grayscale (1 color channel)
- 3x3 kernel erosion
- Median blur kernel 5x5
- Otsu threshold
- Negative
- 3x3 kernel dilation
- Resize

⇒ **Nose on the left side and Nose on the right side:**
- Convert to grayscale (1 color channel) o Gaussian blur kernel 7x7
- Adaptive threshold
- Opening kernel 3x3
- Median blur kernel 5x5
- Negative
- Resize
  
⇒ **Mouth:**
- Convert to grayscale (1 color channel) 
- Gaussian blur kernel 7x7
- Adaptive threshold
- Opening kernel 3x3
- Median blur 5x5
- Negative
- Resize
  
The feature extraction was performed for the **training images** with the **trainset.csv** worksheet as well as for **the test images** with the **testset.csv** worksheet.

The data corresponding to the features have been saved respectively in the **features_train.csv** and **features_test.csv** files

<br/>

## Presentation of the chosen model

### K-Nearest-Neighbors

In order to choose a model, I tried several approaches, including building a model with the **K-Nearest Neighbors algorithm** first.

<br/>

<p align="center" width="100%">
    <img align="center" width="580" src="https://user-images.githubusercontent.com/59794336/110993755-fe8e0800-8377-11eb-8d6d-6b3331cb913e.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 8</strong> - Graph representing the recognition rate as a function of the number of neighbors k
</p>
 
<br/>

We can see that with the Knn method, we obtain a maximum **recognition rate of 0.79, or 79% for a number of neighbors equal to 7**.

By trying to predict data with the trained model, we obtain **an accuracy of 81%** and the associated confusion matrix is shown in **figure 9**. (Training base 70% of the data, Test base 30% of the data)

<br/>

<p align="center" width="100%">
    <img align="center" width="380" src="https://user-images.githubusercontent.com/59794336/110993946-4f056580-8378-11eb-819b-cad66f78bf36.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 9</strong> - Confusion matrix for the Knn algorithm
"Neutral emotion" (0), "anger" (1), "disgust" (3),
"Fear" (4),, "joy" (5), "sadness" (6), "surprise" (7)
</p>
 
<br/>

### Support Vector Machine

The **Support Vector Machine is the classification algorithm** I have chosen to **train my emotional recognition model**.

The principle of SVM is to seek to separate data by drawing a decision boundary such that the distance between the different classes is maximum. We will seek the greatest margin. (**Figure 10**)

This assumes that **the data is linearly separable**, which is rarely the case.
This is why I chose to use the **linear kernel** in order to be able to project the features in a vector space of a larger dimension and thus to be able to make this data linearly separable.

On the other hand, the fact of drawing a decision boundary with the greatest margin between the classes will make it possible to generalize our model and make it better when making predictions.

<br/>

<p align="center" width="100%">
    <img align="center" width="580" src="https://user-images.githubusercontent.com/59794336/110994149-a4417700-8378-11eb-90c6-0ad3ee07a8dd.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 10</strong> - Decision boundaries for the Support Vector Machine algorithm (a), (b), (c), possible decision boundaries
(d) SVM decision boundary such that the margin between classes is maximum
</p>
 
<br/>

In the figure I produced, the black line corresponds to the decision border drawn by the SVM algorithm and allows us to generalize the model unlike the lines (b) and (c) which are borders "very close" to the data.

By trying to predict data with the **trained SVM model**, we obtain an **accuracy of 96.6%** and the associated **confusion matrix** is shown in **figure 11**. (Training base 70% of the data, Test base 30% of the data)

<br/>

<p align="center" width="100%">
    <img align="center" width="380" src="https://user-images.githubusercontent.com/59794336/110994308-e5398b80-8378-11eb-9143-ae2ff572caac.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 11</strong> - Confusion matrix for the SVM algorithm
"Neutral emotion" (0), "anger" (1), "disgust" (3),
"Fear" (4),, "joy" (5), "sadness" (6), "surprise" (7)
</p>
 
<br/>

I estimate **96.6% accuracy** and therefore having **7 misclassified images out of 222** images seems fine to me.

We can read in our confusion matrix that our model was able to classify perfectly all the emotions **except 7 images which were classified as false positives for the class "neutral emotion"**.

This could be explained by the fact that people do not express their emotions in the same way on the face.

<br/>

## Results obtained

After training the chosen model with the **Support Vector Machines algorithm**, we can try to **lassify the test images which therefore do not have a label**.
These images are **unknown to the model**.

The **test base consists of 126 images** representing faces of different people with an emotion. We had previously extracted the features of these test images (**features_test.csv**) at the same time as for the training images (**features_train.csv**).

By performing the prediction of the classes for each of the 126 images, we obtain the results in the file **predictions.csv** comprising one column and 126 rows, i.e. one row per image.

<br/>

<p align="center" width="100%">
    <img align="center" width="380" src="https://user-images.githubusercontent.com/59794336/110994637-56793e80-8379-11eb-8e3d-127e8f5a7138.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 12</strong> - Class predictions for the 126 images of the test base (predictions.csv)
</p>
 
<br/>

<p align="center" width="100%">
    <img align="center" width="880" src="https://user-images.githubusercontent.com/59794336/110994710-7872c100-8379-11eb-98aa-89c7b0031503.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 13</strong> - Images of the test base without labels
(a) image 121, (b) image 122, (c) image 124, (d) image 126
</p>
 
<br/>

## Conclusion

The recognition of emotion through images has been achieved by extracting specific areas of the face where we can observe variations according to the expression of the emotion. I was able to obtain an accuracy of 96.6% by the model trained with the "Support Vector Machine" algorithm which I think is a good score.

<br/>

## Bibliography

**Andrew Ng**: Machine Learning by Stanford University https://www.coursera.org/learn/machine-learning/home/welcome

**Scikit learn**: sklearn.svm.SVC https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

**OpenCV: Image processing**
https: // opencv-python- tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py _table_of_contents_imgproc.html

**Zdzisław Kowalczuk * and Piotr Chudziak**: Identification of Emotions Based on Human Facial Expressions Using a Color-Space Approach

**Khadija Lekdioui**: Recognition of emotional states by visual analysis of the face and machine learning