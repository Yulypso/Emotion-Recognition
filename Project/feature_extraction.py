#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, shutil
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
import timeit

def read_csv():
    pass

def feature_extraction():
    pass

def main():
    sys.exit(0)

if __name__ == "__main__":
    main()