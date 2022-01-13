import os
import sklearn as sk
import pandas as pd
from sklearn import metrics

import numpy as np # linear algebra
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from subprocess import check_output
from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
import re
import matplotlib.pyplot as plt

TRAIN_PATH = "C:\\Users\\allam\\Documents\\Assignment\\6375\\HW3\\hw3_part1_data\\all_data\\train_c1500_d1000.csv"
VALIDATION_PATH = "C:\\Users\\allam\\Documents\\Assignment\\6375\\HW3\\hw3_part1_data\\all_data\\valid_c1500_d1000.csv"
TEST_PATH = "C:\\Users\\allam\\Documents\\Assignment\\6375\\HW3\\hw3_part1_data\\all_data\\test_c1500_d1000.csv"

#Processing train data
data = pd.read_csv(TRAIN_PATH, sep = ',', header = None)
#print(data.head())
X_train = data.iloc[:, 0:499]
y_train = data.iloc[:, 500]

#Processing vaidation data
validationData = pd.read_csv(VALIDATION_PATH, sep = ',', header = None)
X_valid = validationData.iloc[:, 0:499]
y_valid = validationData.iloc[:, 500]

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)
#print("Validation Dataset Accuracy:",metrics.accuracy_score(y_valid, y_pred))

param_option = {"criterion":['gini', 'entropy'], 
                "splitter":['best', 'random'],
                "max_depth":range(1, 100), 
                "min_samples_split":range(2, 40),
                "min_samples_leaf":range(1, 20)}

grid = RandomizedSearchCV(clf,
                   param_distributions = param_option, #param_grid = param_option,
                   n_iter = 500, 
                   cv = 5,
                   verbose = 1,
                   n_jobs = -1)

grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)
#print(grid.best_score_)
#print(data.shape)
#print(validationData.shape)
dataArray = [data, validationData]
completeData = pd.concat(dataArray)
completeData.head

#Processing test data
testData = pd.read_csv(TEST_PATH, sep = ',', header = None)
X_test = testData.iloc[:, 0:499]
y_test = testData.iloc[:, 500]

X_complete = completeData.iloc[:, 0:499]
y_complete = completeData.iloc[:, 500]

clfComplete = DecisionTreeClassifier(**grid.best_params_)
clfComplete = clfComplete.fit(X_complete,y_complete)
y_finalPred = clfComplete.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_finalPred)*100)
print("F1 Score:", metrics.f1_score(y_test, y_finalPred)*100)



