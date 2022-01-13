import os
import sklearn as sk
import pandas as pd
from sklearn import metrics

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree
import re
import winsound
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784',version=1,return_X_y=True)
X = X / 255

# rescale the data, use the traditional train/test split
# (40K: Train), (20K: Validation) and (10K: Test)
X_train, X_valid, X_test = X[:40000], X[40000:60000], X[60000:]
y_train, y_valid, y_test = y[:40000], y[40000:60000], y[60000:]

dtcCLF = DecisionTreeClassifier()
dtcCLF = dtcCLF.fit(X_train, y_train)
y_pred = dtcCLF.predict(X_valid)
#print("Validation Dataset Accuracy DTC:", metrics.accuracy_score(y_valid, y_pred))

param_option = {"criterion":['gini', 'entropy'], 
                "splitter":['best', 'random'],
                "max_depth":range(1, 100), 
                "min_samples_split":range(2, 40),
                "min_samples_leaf":range(1, 20)}

grid = RandomizedSearchCV(dtcCLF,
                   param_distributions = param_option, #param_grid = param_option,
                   n_iter = 20, #300, 
                   cv = 5,
                   verbose = 1,
                   n_jobs = -1)

grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)
#print(grid.best_score_)

X_complete = X[:60000]
y_complete = y[:60000]

dtcCLFComplete = DecisionTreeClassifier(**grid.best_params_)
dtcCLFComplete = dtcCLFComplete.fit(X_complete,y_complete)
y_finalPred = dtcCLFComplete.predict(X_test)
print("Accuracy of Decision Tree Classifier on MNIST dataset:", metrics.accuracy_score(y_test, y_finalPred)*100)

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

# Bagging Classifier
bcCLF = BaggingClassifier(DecisionTreeClassifier(**grid.best_params_))
bcCLF = bcCLF.fit(X_train, y_train)
y_pred = bcCLF.predict(X_valid)
#print("Validation Dataset Accuracy BC:", metrics.accuracy_score(y_valid, y_pred))

param_option = {'bootstrap_features': [False, True],
                'bootstrap': [False, True],
              'max_features': [0.5, 0.7, 1.0],
              'max_samples': [0.5, 0.7, 1.0],
              'n_estimators': [2, 5, 10, 20, 40, 60, 80, 100],}

gridBC = RandomizedSearchCV(bcCLF,
                   param_distributions = param_option,
                   n_iter = 20, #288, 
                   cv = 5,
                   verbose = 1,
                   n_jobs = -1)
gridBC.fit(X_train, y_train)
print(gridBC.best_params_)
print(gridBC.best_estimator_)
#print(gridBC.best_score_)

bcCLFComplete = BaggingClassifier(**gridBC.best_params_)
bcCLFComplete = bcCLFComplete.fit(X_complete,y_complete)
y_finalPred = bcCLFComplete.predict(X_test)
print("Accuracy for Bagging Classifier on MNIST dataset:", metrics.accuracy_score(y_test, y_finalPred)*100)

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 2000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

#Random Forest Classifier
rfcCLF = RandomForestClassifier()
rfcCLF = rfcCLF.fit(X_train, y_train)
y_pred = rfcCLF.predict(X_valid)
#print("Validation Dataset Accuracy RFC:",metrics.accuracy_score(y_valid, y_pred))

param_option = {"n_estimators": [10, 20, 40, 60, 80, 100],
                "criterion":['gini', 'entropy'],
                "max_depth":range(1, 100), 
                "min_samples_split":range(2, 40),
                "min_samples_leaf":range(1, 20)
                }

gridRFC = RandomizedSearchCV(rfcCLF,
                   param_distributions = param_option, #param_grid = param_option,
                   n_iter = 20, #200, 
                   cv = 5,
                   verbose = 1,
                   n_jobs = -1)

gridRFC.fit(X_train, y_train)
print(gridRFC.best_params_)
print(gridRFC.best_estimator_)
#print(gridRFC.best_score_)

rfcCLFComplete = RandomForestClassifier(**gridRFC.best_params_)
rfcCLFComplete = rfcCLFComplete.fit(X_complete,y_complete)
y_finalPred = rfcCLFComplete.predict(X_test)
print("Accuracy for Random Forest Classifier on MNIST dataset:", metrics.accuracy_score(y_test, y_finalPred)*100)

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 3000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

#Gradient Boosting Classifier
gbcCLF = GradientBoostingClassifier()
gbcCLF = gbcCLF.fit(X_train, y_train)
y_pred = gbcCLF.predict(X_valid)
#print("Validation Dataset Accuracy GBC:",metrics.accuracy_score(y_valid, y_pred))

param_option = {"loss":['deviance'],
                "learning_rate":[0.01, 0.05, 0.1, 0.2],
                "n_estimators": [10, 20, 40, 60, 80, 100],
                #"criterion":['friedman_mse', 'squared_error'],
                "max_depth":range(1, 100), 
                "min_samples_split":range(2, 40),
                "min_samples_leaf":range(1, 20)
                }

gridGBC = RandomizedSearchCV(gbcCLF,
                   param_distributions = param_option, #param_grid = param_option,
                   n_iter = 10, #50, 
                   cv = 5,
                   verbose = 2,
                   n_jobs = -1)

gridGBC.fit(X_train, y_train)
print(gridGBC.best_params_)
print(gridGBC.best_estimator_)
#print(gridGBC.best_score_)

gbcCLFComplete = GradientBoostingClassifier(**gridGBC.best_params_)
gbcCLFComplete = gbcCLFComplete.fit(X_complete,y_complete)
y_finalPred = gbcCLFComplete.predict(X_test)
print("Accuracy of Gradient Boosting Classifier on MNIST dataset:", metrics.accuracy_score(y_test, y_finalPred)*100)

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 4000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)