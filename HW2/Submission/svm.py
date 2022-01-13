#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import fetch_openml
# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784',version=1,return_X_y=True)
X = X / 255

# rescale the data, use the traditional train/test split
# (60K: Train) and (10K: Test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

print(X_train)

#Importing additional Libraries
from sklearn import svm 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#1  
svmClassifier = SVC(kernel='linear', C=2.0)
svmClassifier.fit(X_train,y_train)

predictedResult = svmClassifier.predict(X_test)
print(predictedResult)

accuracy = accuracy_score(y_test, predictedResult)
errorRate = 1 - accuracy
print(errorRate*100)
print(accuracy*100)

#2
svmClassifier2 = SVC(kernel='poly', gamma='auto', max_iter=10, degree=3)
svmClassifier2.fit(X_train,y_train)

predictedResult2 = svmClassifier2.predict(X_test)
print(predictedResult2)

accuracy = accuracy_score(y_test, predictedResult2)
errorRate = 1 - accuracy
print(errorRate*100)
print(accuracy*100)

#3
svmClassifier3 = SVC(kernel='rbf', gamma='scale', C=5.0, tol=1e-4)
svmClassifier3.fit(X_train,y_train)

predictedResult3 = svmClassifier3.predict(X_test)
print(predictedResult3)

accuracy = accuracy_score(y_test, predictedResult3)
errorRate = 1 - accuracy
print(errorRate*100)
print(accuracy*100)

#4
svmClassifier4 = SVC(kernel='sigmoid', tol=0.1, C=1.0)
svmClassifier4.fit(X_train,y_train)

predictedResult4 = svmClassifier4.predict(X_test)
print(predictedResult4)

accuracy = accuracy_score(y_test, predictedResult4)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)

#5
svmClassifier5 = SVC(kernel='linear',gamma='scale',max_iter=50,random_state=1)
svmClassifier5.fit(X_train,y_train)

predictedResult5 = svmClassifier5.predict(X_test)
print(predictedResult5)

accuracy = accuracy_score(y_test, predictedResult5)
errorRate = 1 - accuracy
print(errorRate*100)
print(accuracy*100)

#6
svmClassifier6 = SVC(kernel='poly',coef0=5.5,tol=5.5)
svmClassifier6.fit(X_train,y_train)

predictedResult6 = svmClassifier6.predict(X_test)
print(predictedResult6)

accuracy = accuracy_score(y_test, predictedResult6)
errorRate = 1 - accuracy
print(errorRate*100)
print(accuracy*100)

#7
svmClassifier7 = SVC(kernel='rbf', class_weight='balanced', max_iter=-1, C=1.0)
svmClassifier7.fit(X_train,y_train)


predictedResult7 = svmClassifier7.predict(X_test)
print(predictedResult7)

accuracy = accuracy_score(y_test, predictedResult7)
errorRate = 1 - accuracy
print(errorRate*100)
print(accuracy*100)

#8
svmClassifier8 = SVC(kernel='sigmoid', coef0=0.68,degree=15, random_state=1)
svmClassifier8.fit(X_train,y_train)

predictedResult8 = svmClassifier8.predict(X_test)
print(predictedResult8)

accuracy = accuracy_score(y_test, predictedResult8)
errorRate = 1 - accuracy
print(errorRate*100)
print(accuracy*100)

#9
svmClassifier9 = SVC(kernel='poly',degree=20,coef0=50.08,class_weight='balanced',max_iter=-1)
svmClassifier9.fit(X_train,y_train)

predictedResult9 = svmClassifier9.predict(X_test)
print(predictedResult9)

accuracy = accuracy_score(y_test, predictedResult9)
errorRate = 1 - accuracy
print(errorRate*100)
print(accuracy*100)

#10
svmClassifier10 = SVC(kernel='rbf', tol=90.76, gamma='auto', degree=10,class_weight='balanced')
svmClassifier10.fit(X_train,y_train)

predictedResult10 = svmClassifier10.predict(X_test)
print(predictedResult10)

accuracy = accuracy_score(y_test, predictedResult10)
errorRate = 1 - accuracy
print(errorRate*100)
print(accuracy*100)


