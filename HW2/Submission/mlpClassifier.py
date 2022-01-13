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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#1
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(6,),  random_state=1)
mlp.fit(X_train,y_train)

predictedResult = mlp.predict(X_test)
print(predictedResult)

accuracy = accuracy_score(y_test, predictedResult)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)


#2
mlp2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp2.fit(X_train,y_train)

predictedResult2 = mlp2.predict(X_test)
print(predictedResult2)

accuracy = accuracy_score(y_test, predictedResult2)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)



#3
mlp3 = MLPClassifier(solver='sgd', activation='logistic',alpha=1,learning_rate_init=.1,early_stopping=True)
mlp3.fit(X_train,y_train)

predictedResult3 = mlp3.predict(X_test)
print(predictedResult3)

accuracy = accuracy_score(y_test, predictedResult3)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)


#4
mlp4 = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)
mlp4.fit(X_train,y_train)

predictedResult4 = mlp4.predict(X_test)
print(predictedResult4)

accuracy = accuracy_score(y_test, predictedResult4)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)


#5
mlp5 = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=480, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.1)
mlp5.fit(X_train,y_train)

predictedResult5 = mlp5.predict(X_test)
print(predictedResult5)

accuracy = accuracy_score(y_test, predictedResult5)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)



#6
mlp6 = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=480, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.1)
mlp6.fit(X_train,y_train)

predictedResult6 = mlp6.predict(X_test)
print(predictedResult6)

accuracy = accuracy_score(y_test, predictedResult6)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)



#7
mlp7 = MLPClassifier(solver='lbfgs', alpha=0.1, random_state=1, max_iter=2000, early_stopping=True, hidden_layer_sizes=[100, 100])
mlp7.fit(X_train,y_train)

predictedResult7 = mlp7.predict(X_test)
print(predictedResult7)

accuracy = accuracy_score(y_test, predictedResult7)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)



#8
mlp8 = MLPClassifier(solver='lbfgs', alpha=10.0, random_state=1, max_iter=2000, early_stopping=True, hidden_layer_sizes=[100, 100])
mlp8.fit(X_train,y_train)

predictedResult8 = mlp8.predict(X_test)
print(predictedResult8)

accuracy = accuracy_score(y_test, predictedResult8)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)


#9
mlp9 = MLPClassifier(solver='lbfgs', activation='tanh',alpha=1,hidden_layer_sizes=(150,100,50,50), random_state=1,max_iter=1000)
mlp9.fit(X_train,y_train)

predictedResult9 = mlp9.predict(X_test)
print(predictedResult9)

accuracy = accuracy_score(y_test, predictedResult9)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)


#10
mlp10 = MLPClassifier(solver='lbfgs', alpha=1.0, random_state=1, max_iter=2000, early_stopping=True, hidden_layer_sizes=[100, 100])
mlp10.fit(X_train,y_train)

predictedResult10 = mlp10.predict(X_test)
print(predictedResult10)

accuracy = accuracy_score(y_test, predictedResult10)
errorRate = 1 - accuracy
print(errorRate)
print(accuracy)




