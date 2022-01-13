
from __future__ import print_function
import numpy as np
import sys
import time
import contextlib
from CLT_RandomForest import CLT_RandomForest
from Util import *
from CLT_class import CLT
import random
from sklearn.utils import resample
from collections import defaultdict
import os
import timeit

class RandomForest():
    
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks

    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    def learn(self, dataset, n_components, hyperparameterR):
        
        self.n_components = n_components
        # For RandomForest, weigts can be uniform - keeping them 1
        weights=np.ones((n_components,dataset.shape[0]))
        self.mixture_probs = [1/n_components]*n_components
    
        self.clt_list = [CLT_RandomForest() for i in range(n_components)]
        
        for k in range(n_components):
            resampledData = resample(dataset)
            self.clt_list[k].learn(resampledData)
            self.clt_list[k].update(resampledData, weights[k], hyperparameterR)

    '''
        Compute the log-likelihood score of the dataset
    '''
    
    def computeLL(self, dataset):
        
        ll = 0.0
        for k in range(self.n_components):
            ll += self.clt_list[k].computeLL(dataset)
        return ll/dataset.shape[0]

forest = RandomForest()

# Reading and storing the respective dataset files
serialNumber = 0
index = defaultdict(list)
for i, file in enumerate(os.listdir("C:\\Users\\allam\\Documents\\Assignment\\6375\\HW4\\datasets\\dataset")):
    if(i % 3 == 0):
        serialNumber += 1
    index[serialNumber].append(file)

print("Serial number\t Dataset\n")
for key, values in index.items():
    print(key, "\t\t", values[1])

print("\nEnter the serial number for which dataset to run")
selection = input()

print("Please wait, while we learn ensemble models for...")
print(index[int(selection)][1])
print()

for key, values in index.items():
    if(key == int(selection)):
        dataset=Util.load_dataset("C:\\Users\\allam\\Documents\\Assignment\\6375\\HW4\\datasets\\dataset\\"+index[key][1])
        validateset = Util.load_dataset("C:\\Users\\allam\\Documents\\Assignment\\6375\\HW4\\datasets\\dataset\\"+index[key][2])
        testset=Util.load_dataset("C:\\Users\\allam\\Documents\\Assignment\\6375\\HW4\\datasets\\dataset\\"+index[key][0])
        break

# Validation 
# for key, values in index.items():
#     print("Running the model for ", index[key][1])
#     dataset=Util.load_dataset("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset\\"+index[key][1])
#     validateset = Util.load_dataset("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset\\"+index[key][2])
#     testset=Util.load_dataset("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset\\"+index[key][0])
#     # Latent variable can take values from [2, 5, 10, 20]
#     for hiddenVariable in [2, 5, 20, 10]:
#         for paramR in [5, 75, 130, 450]:
#             forest.learn(dataset, n_components=hiddenVariable, hyperparameterR = paramR)
#             print("Running on the validation set when the hidden variable can take up to", hiddenVariable,"values")
#             print("And hyparameter r is -", paramR)
#             print("Log likelihood = ", forest.computeLL(validateset))
# #sys.stdout.close()

# Test time code
# Below are the values of hyperparameters derived based on validation set
kTest = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
paramR = [5, 130, 130, 450, 130, 5, 5, 75, 450, 450]
hiddenVariableList = defaultdict(int)
paramRList = defaultdict(int)
i = 0
for key, values in index.items():
    hiddenVariableList[key] = kTest[i]
    paramRList[key] = paramR[i]
    i += 1

print("Running the model 5 times and printing log likelihood for each initialization")
for i in range(5):
    start = timeit.default_timer()
    print("Evaluating the model for appropriate values of hyperparameters ...")
    forest.learn(dataset, n_components=hiddenVariableList[int(selection)], hyperparameterR = paramRList[int(selection)])
    print("\nThe number of values the hidden variable can take is k = ", hiddenVariableList[int(selection)])
    print("Value of parameter R = ", paramRList[int(selection)])
    print("\nLog likelihood = ", forest.computeLL(testset))
    stop = timeit.default_timer()
    print("Runtime - ", stop-start)