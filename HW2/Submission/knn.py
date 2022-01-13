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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#1
knnClassifier = KNeighborsClassifier(n_neighbors=9)
knnClassifier.fit(X_train,y_train)

predictedResult = knnClassifier.predict(X_test)
print(predictedResult)

accuracy = accuracy_score(y_test, predictedResult)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)

#2
knnClassifier2 = KNeighborsClassifier(leaf_size=50, n_neighbors=9, p=2)
knnClassifier2.fit(X_train,y_train)

predictedResult2 = knnClassifier2.predict(X_test)
print(predictedResult2)

accuracy = accuracy_score(y_test, predictedResult2)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)

#3
knnClassifier3 = KNeighborsClassifier(leaf_size=10, n_neighbors=9, p=1, algorithm='ball_tree')
knnClassifier3.fit(X_train,y_train)

predictedResult3 = knnClassifier3.predict(X_test)
print(predictedResult3)

accuracy = accuracy_score(y_test, predictedResult3)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)

#4
knnClassifier4 = KNeighborsClassifier(n_neighbors=9, algorithm='kd_tree')
knnClassifier4.fit(X_train,y_train)

predictedResult4 = knnClassifier4.predict(X_test)
print(predictedResult4)

accuracy = accuracy_score(y_test, predictedResult4)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)

#5
knnClassifier5 = KNeighborsClassifier(n_neighbors=9, algorithm='brute')
knnClassifier5.fit(X_train,y_train)

predictedResult5 = knnClassifier5.predict(X_test)
print(predictedResult5)

accuracy = accuracy_score(y_test, predictedResult5)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)

#6
knnClassifier6 = KNeighborsClassifier(algorithm='ball_tree',n_neighbors=9,leaf_size=10,n_jobs=3)
knnClassifier6.fit(X_train,y_train)

predictedResult6 = knnClassifier6.predict(X_test)
print(predictedResult6)

accuracy = accuracy_score(y_test, predictedResult6)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)

#7
knnClassifier7 = KNeighborsClassifier(algorithm='brute',n_neighbors=9 ,n_jobs=5, p=2, metric='minkowski')
knnClassifier7.fit(X_train,y_train)

predictedResult7 = knnClassifier7.predict(X_test)
print(predictedResult7)

accuracy = accuracy_score(y_test, predictedResult7)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)

#8
knnClassifier8 = KNeighborsClassifier(algorithm='kd_tree',p=2,leaf_size=30,weights='uniform', n_neighbors=11)
knnClassifier8.fit(X_train,y_train)

predictedResult8 = knnClassifier8.predict(X_test)
print(predictedResult8)

accuracy = accuracy_score(y_test, predictedResult8)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)

#9
knnClassifier9 = KNeighborsClassifier(algorithm='auto',leaf_size=50,weights='distance',n_jobs=5, n_neighbors=13)
knnClassifier9.fit(X_train,y_train)

predictedResult9 = knnClassifier9.predict(X_test)
print(predictedResult9)

accuracy = accuracy_score(y_test, predictedResult9)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)

#10
knnClassifier10 = KNeighborsClassifier(algorithm='brute',n_jobs=10,weights='uniform', n_neighbors=15)
knnClassifier10.fit(X_train,y_train)

predictedResult10 = knnClassifier10.predict(X_test)
print(predictedResult10)

accuracy = accuracy_score(y_test, predictedResult10)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)

