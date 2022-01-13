#Importing additional Libraries
import os
import numpy as np
import pandas as pd
import sklearn as sk
from IPython.display import display
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

TRAIN_PATH = 'C:\\Users\\allam\\Documents\\Assignment\\6375\\HW2\\netflix\\TrainingRatings.txt'
TEST_PATH = 'C:\\Users\\allam\\Documents\\Assignment\\6375\\HW2\\netflix\\TestingRatings.txt'

movData = pd.read_csv(TRAIN_PATH, sep = ',', names = ['MovieID', 'UserID', 'Rating'], dtype = {'UserID': 'int32', 'MovieID': 'int32', 'Rating': 'float32'})
movDesc = pd.read_csv('C:\\Users\\allam\\Documents\\Assignment\\6375\\HW2\\netflix\\movie_titles.txt', sep = ',', names = ['MovieID', 'YearOfRelease', 'Title'], encoding = 'latin-1')
testData = pd.read_csv(TEST_PATH, sep = ',', names = ['MovieID', 'UserID', 'Rating'], dtype = {'UserID': 'int32', 'MovieID': 'int32', 'Rating': 'float32'})

from scipy.sparse import csr_matrix
userMovDf = movData.pivot(index = 'UserID', columns = 'MovieID', values = 'Rating').fillna(0)
#userMovDf

def pearsonCoeff(normalDf, activeUser, otherUser):
    cosine_distance = float(spatial.distance.cosine(userMovDf.loc[[activeUser], :], userMovDf.loc[[otherUser], :]))
    return 1 - cosine_distance

def predict(trainDf, activeuserId, movieId):
    total = 0.0
    kappaSum = 0.0
    predictedActiveUser = 0.0
    #weightSum = 0.0
    activeUserMean = (trainDf.loc[[activeuserId], :].sum(axis = 1)).loc[activeuserId] / np.sum(np.count_nonzero(trainDf.loc[[activeuserId], :], axis = 0)) # va (first term)
    #print(activeUserMean)
    for i, row in trainDf.iterrows():
        vij = row[movieId]
        # weightUser = pearsonCoeff(trainDf, activeuserId, i)
        # otherUserMean = (trainDf.loc[[i], :].sum(axis = 1)).loc[i] / np.sum(np.count_nonzero(trainDf.loc[[i], :], axis = 0))
        # otherTerm = vij - otherUserMean
        if vij != 0:
            weightUser = pearsonCoeff(trainDf, activeuserId, i)
            otherUserMean = (trainDf.loc[[i], :].sum(axis = 1)).loc[i] / np.sum(np.count_nonzero(trainDf.loc[[i], :], axis = 0))
            otherTerm = vij - otherUserMean
            total += weightUser * otherTerm
            kappaSum += np.absolute(weightUser)
        # else:
        #     weightUser = 0.0
        #     otherTerm = 0.0
        # total += weightUser * otherTerm
        # kappaSum += np.absolute(weightUser)
        #kappaSum += 1/np.absolute(weightUser)
        #print("weightUser: {0}, otherUserMean: {1}, vij: {2}, otherTerm: {3}, total: {4}, kappaSum: {5}, weightSum: {6}".format(weightUser, otherUserMean, vij, otherTerm, total, kappaSum, weightSum))
    #predictedActiveUser = activeUserMean + kappaSum * total
    #print(predictedActiveUser)
    if kappaSum != 0:
        predictedActiveUser = activeUserMean + (total/kappaSum)
    return math.ceil(predictedActiveUser)

def userKnn(knntrainArr, incomingArr, n):
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(knntrainArr)
    distances, indices = nbrs.kneighbors(incomingArr)
    return indices[0][:]

userMovArr = np.asarray(userMovDf)
originalIndexArr = userMovDf.index
testData = testData.sample(frac=1, random_state=42).reset_index(drop=True)
#testData = testData.iloc[1:301] #Temporary
counter = 0

print('Processing...')
for i, row in testData.iterrows():
    testUser = row['UserID']
    indexArr = userKnn(userMovArr, userMovDf.loc[testUser].values.reshape(1, -1), 20)
    newArr = []
    orgIndexValArr = []
    for j in range(len(indexArr)):
        newArr.append(userMovArr[indexArr[j]])
        orgIndexValArr.append(originalIndexArr[indexArr[j]])
    newArr = np.asarray(newArr)
    generatedKnnDf = pd.DataFrame(newArr, index = orgIndexValArr, columns = userMovDf.columns)
    predictedRating = predict(generatedKnnDf, row['UserID'], row['MovieID'])
    testData.loc[(testData.UserID == row['UserID']), ['PredictedRating']] = predictedRating
    counter += 1
    if counter % 10 == 0:
        print('Processing completed for {0} test users'.format(counter))

systemMae = mean_absolute_error(testData['Rating'].to_numpy(), testData['PredictedRating'].to_numpy())
systemRmse = mean_squared_error(testData['Rating'].to_numpy(), testData['PredictedRating'].to_numpy())

print('The Mean Absolute Error is {0}: '.format(systemMae))
print('The Root Mean Square Error is {0}: '.format(systemRmse))
