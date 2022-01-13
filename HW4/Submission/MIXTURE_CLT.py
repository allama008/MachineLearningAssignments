from __future__ import print_function
import os
from collections import defaultdict
import numpy as np
import os 
from sys import exit
import numpy as np
import sys
import time
from Util import *
from CLT_class import CLT
import pandas as pd
class MIXTURE_CLT():
    
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks
        

    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    def learn(self, dataset, n_components=2, max_iter=10, epsilon=1e-1):
        # For each component and each data point, we have a weight
        weights=np.zeros((dataset.shape[0],n_components),dtype=float)
        # print("OG Here")
        # print(weights.shape,dataset.shape)
        
        # Randomly initialize the chow-liu trees and the mixture probabilities
        # Your code for random initialization goes here
       # weights=np.random.random((dataset.shape[0],n_components),dtype=float)
        for i in range(n_components):
            self.clt_list.append(CLT())
            (self.clt_list[i]).learn(dataset[np.random.randint(0,dataset.shape[0])].reshape(1,-1))
            
        a=np.random.random(size=n_components)
        a=a/(a.sum())
        self.mixture_probs=a
        gamma=np.zeros((dataset.shape[0],n_components),dtype=float)
        gamma_denom=np.zeros(dataset.shape[0],dtype=float)
        like=0
        old_like=1
        for itr in range(max_iter):
            #E-step: Complete the dataset to yield a weighted dataset
            # We store the weights in an array weights[ncomponents,number of points]
            
            #Your code for E-step here
            print("Iteration no. -{} ".format(itr))
            old_weights=np.copy(weights)
            for  i in range(dataset.shape[0]):
                for  j in range(n_components):
                    gamma_denom[i]=gamma_denom[i]+self.mixture_probs[j]*self.clt_list[j].getProb(dataset[i])
            
            for i in range(dataset.shape[0]):
                for j in range(n_components):
                    gamma[i,j]=(self.mixture_probs[j]*self.clt_list[j].getProb(dataset[i]))/gamma_denom[i]
            print(gamma)
            inv_L=np.zeros(n_components,dtype=float)
            for j in range(n_components):
                for i in range(dataset.shape[0]):
                    inv_L[j]=inv_L[j] + gamma[i,j]
            print(inv_L)
            for j in range(n_components):
                for i in range(dataset.shape[0]):
                    weights[i,j]=gamma[i,j]/inv_L[j]
    
            if(np.abs(old_like-like)<= epsilon):
                print("Converging on Average Log Likelihood")
            
            
            # M-step: Update the Chow-Liu Trees and the mixture probabilities
            
            #Your code for M-Step here
            for k in range(n_components):
                self.mixture_probs[k]=inv_L[k]/dataset.shape[0]
                # print("Here Now")
                # print(weights[:,k].shape)
                self.clt_list[k].update(dataset,weights[:,k])
                
            print("Likelihood:",(self.computeLL(dataset)/dataset.shape[0]))
            old_like=like
            like=(self.computeLL(dataset)/dataset.shape[0])
    
    """
        Compute the log-likelihood score of the dataset
    """
    def computeLL(self, dataset):
        ll = 0.0
        for i in range (dataset.shape[0]):
            likelihood=0.0
            for j in range(self.n_components):
                likelihood=likelihood+self.mixture_probs[j]*self.clt_list[j].getProb(dataset[i])

            ll=ll+ np.log(likelihood)
        # Write your code below to compute likelihood of data
        #   Hint:   Likelihood of a data point "x" is sum_{c} P(c) T(x|c)
        #           where P(c) is mixture_prob of cth component and T(x|c) is the probability w.r.t. chow-liu tree at c
        #           To compute T(x|c) you can use the function given in class CLT
        return ll
    

    
'''
    After you implement the functions learn and computeLL, you can learn a mixture of trees using
    To learn Chow-Liu trees, you can use
    mix_clt=MIXTURE_CLT()
    ncomponents=10 #number of components
    max_iter=50 #max number of iterations for EM
    epsilon=1e-1 #converge if the difference in the log-likelihods between two iterations is smaller 1e-1
    dataset=Util.load_dataset(path-of-the-file)
    mix_clt.learn(dataset,ncomponents,max_iter,epsilon)
    
    To compute average log likelihood of a dataset w.r.t. the mixture, you can use
    mix_clt.computeLL(dataset)/dataset.shape[0]
'''

serialNumber = 0
index = defaultdict(list)
dirname, filename = os.path.split(os.path.abspath("C:\\Users\\allam\\Documents\\Assignment\\6375\\HW4\\datasets"))
path = dirname+"/dataset/"
for i, file in enumerate(os.listdir(path)):
    if(i % 3 == 0):
        serialNumber += 1
    index[serialNumber].append(file)

print("Serial number\t Dataset\n")
for key, values in index.items():
    print(key, "\t\t", values[1])

print("\nEnter the serial number for which dataset to run")
selection = input()
for key, values in index.items():
    if(key == int(selection)):

        dataset_train = Util.load_dataset(f"{path}/accidents.ts.data")
        dataset_valid = Util.load_dataset(f"{path}/accidents.valid.data")
        dataset_test = Util.load_dataset(f"{path}/accidents.test.data")
        break
        
mix_clt=MIXTURE_CLT()
mix_clt.n_components=20
   
#Tuning on validation set   
mix_clt.learn(dataset_train,mix_clt.n_components,10)
llikelihood=mix_clt.computeLL(dataset_valid)  
print("Log likelihood = {}".format(llikelihood))
print("Average Log Likelihood= {}".format(llikelihood/dataset_valid.shape[0]))
    