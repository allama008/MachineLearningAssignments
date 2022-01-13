# Importing system Libraries
from os import walk
from string import punctuation
from random import shuffle
from collections import Counter
import os

#Importing additional Libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn as sk
import nltk

#Setting the root directory
TRAIN_PATH = "C:\\Users\\allam\\Documents\\Assignment\\6375\\Dataset\\Train\\hw1_train\\train"
TEST_PATH = "C:\\Users\\allam\\Documents\\Assignment\\6375\\Dataset\\Test\\hw1_test\\test"

#Creating the HAM and SPAM list.
ham_list = []
spam_list = []

for directories, subdirs, files in os.walk(TRAIN_PATH):
    if(os.path.split(directories)[1] == 'ham'):
        for filename in files:
            with open(os.path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                ham_list.append(data)
    if(os.path.split(directories)[1] == 'spam'):
        for filename in files:
            with open(os.path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                spam_list.append(data)

# print(len(ham_list))
# print(len(spam_list))

#print(type(ham_list))

# ham_list = ham_list.lower()
#ham_list = ham_list.replace('\W', ' ')
# spam_list = spam_list.lower()
#spam_list = ham_list.replace('\W', ' ')

hamDf = pd.DataFrame(ham_list, columns=['Messages'])
labelList = ['ham'] * len(ham_list)
labelDf = pd.DataFrame(labelList, columns=['Label'])
finalHamDf = pd.concat([labelDf, hamDf], axis=1, ignore_index=True)

spamDf = pd.DataFrame(spam_list, columns=['Messages'])
labelList = ['spam'] * len(spam_list)
labelDf = pd.DataFrame(labelList, columns=['Label'])
finalSpamDf = pd.concat([labelDf, spamDf], axis=1, ignore_index=True)
#print(finalSpamDf)

SPHamlist = [finalHamDf, finalSpamDf]
SPHamDf = pd.concat(SPHamlist, ignore_index=True)
SPHamDf.columns = ['HamSpamLabel', 'Email_Message']
#print(SPHamDf)

#print(SPHamDf['Label'].value_counts(normalize=True))

#Data Cleaning
#print(SPHamDf.head(5))
SPHamDf['Email_Message'] = SPHamDf['Email_Message'].str.replace(
   '\W', ' ') # Removes punctuation
SPHamDf['Email_Message'] = SPHamDf['Email_Message'].str.lower()
#print(SPHamDf.head(5))

#Bag of Words
SPHamDf['Email_Message'] = SPHamDf['Email_Message'].str.split()
#SPHamDf['Email_Message'] = SPHamDf['Email_Message'].str.replace(' ', '') 

# print('Checking1')
# print(SPHamDf['Email_Message'].head())
# print(type(list(SPHamDf['Email_Message'])))
# print('End Checking1')

vocabulary = []
for sms in SPHamDf['Email_Message']:
   for word in sms:
      vocabulary.append(word)

vocabulary = list(set(vocabulary))
# print('1.Checking')
# print(vocabulary)

word_counts_per_sms = {unique_word: [0] * len(SPHamDf['Email_Message']) for unique_word in vocabulary}

for index, sms in enumerate(SPHamDf['Email_Message']):
   for word in sms:
      word_counts_per_sms[word][index] += 1

word_counts = pd.DataFrame(word_counts_per_sms)
# print('Important Checking')
# print(SPHamDf.head())
# print(word_counts.head())

# print(len(SPHamDf))
# print(len(word_counts))

# print()
SPHamDfBoG = pd.concat([SPHamDf, word_counts], axis=1)
# SPHamDfBoG = SPHamDf.join(word_counts, how='outer')
# print('Checking indexing')
# print(SPHamDfBoG.head())
# print(SPHamDfBoG.drop(['HamSpamLabel', 'Email_Message'], axis = 1).loc[[1, 2]])


#SPHamDfBoG.to_csv(r"C:\\Users\\allam\\Documents\\Assignment\\6375\\Checking3.csv", index = False, header=True)

wordCountsBern = word_counts.clip(upper=1)
SPHamDfBern = pd.concat([SPHamDf, wordCountsBern], axis=1)



#Testing data


#Start creating the HAM and SPAM test data.
hamTestlist = []
spamTestlist = []

for directories, subdirs, files in os.walk(TEST_PATH):
    if(os.path.split(directories)[1] == 'ham'):
        for filename in files:
            with open(os.path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                hamTestlist.append(data)
    if(os.path.split(directories)[1] == 'spam'):
        for filename in files:
            with open(os.path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                spamTestlist.append(data)

hamTestDf = pd.DataFrame(hamTestlist, columns=['Messages'])
labelTestList = ['ham'] * len(hamTestlist)
labelTestDf = pd.DataFrame(labelTestList, columns=['Label'])
finalHamTestDf = pd.concat([labelTestDf, hamTestDf], axis=1, ignore_index=True)

spamTestDf = pd.DataFrame(spamTestlist, columns=['Messages'])
labelTestList = ['spam'] * len(spamTestlist)
labelTestDf = pd.DataFrame(labelTestList, columns=['Label'])
finalSpamTestDf = pd.concat([labelTestDf, spamTestDf], axis=1, ignore_index=True)
#print(finalSpamDf)

SPHamTestlist = [finalHamTestDf, finalSpamTestDf]
SPHamTestDf = pd.concat(SPHamTestlist, ignore_index=True)
SPHamTestDf.columns = ['HamSpamLabel', 'Email_Message']
SPHamTestDfBern = SPHamTestDf
SPHamTestDfLR = SPHamTestDf
#End creating the Ham and Spam test data

from MultiNaiveBayes import MultiNaiveBayes
MultiNaiveBayes(SPHamTestDf, TEST_PATH)

p_ham, p_spam, parameters_ham, parameters_spam = MultiNaiveBayes.train(SPHamDfBoG, vocabulary)
#print('Working till here')
#print(SPHamDf.head())
SPHamTestDf['Predicted'] = SPHamTestDf['Email_Message'].apply(MultiNaiveBayes.classifyTest, args=[p_ham, p_spam, parameters_ham, parameters_spam])
#print(SPHamTestDf.head())
#print('Is it working here')
correct = 0
total = SPHamTestDf.shape[0]

for row in SPHamTestDf.iterrows():
   row = row[1]
   if row['HamSpamLabel'] == row['Predicted']:
      correct += 1

print('Multinomial Naive Bayes Report')
print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', (correct/total)*100)


#Start working on Discrete Naive Bayes
from DiscreteNaiveBayes import DiscreteNaiveBayes
DiscreteNaiveBayes(SPHamTestDfBern, TEST_PATH)

p_ham, p_spam, parameters_ham, parameters_spam = DiscreteNaiveBayes.train(SPHamDfBern, vocabulary)

SPHamTestDfBern['Predicted'] = SPHamTestDfBern['Email_Message'].apply(DiscreteNaiveBayes.classifyTest, args=[p_ham, p_spam, parameters_ham, parameters_spam])
#print(SPHamTestDfBern.head())

correctBern = 0
totalBern = SPHamTestDfBern.shape[0]

for rowBern in SPHamTestDfBern.iterrows():
   rowBern = rowBern[1]
   if rowBern['HamSpamLabel'] == rowBern['Predicted']:
      correctBern += 1

print('Discrete Naive Bayes Report')
print('Correct Discrete:', correctBern)
print('Incorrect Discrete:', totalBern - correctBern)
print('Accuracy Discrete:', (correctBern/totalBern)*100)


from LRFinalVersion import LogisticRegression

_lambdaVal = int(input("Kindly enter the value of Lambda: "))
noOfIterations = int(input("Kindly enter the Number of Iterations: "))
classifyOption = int(input("Kindly enter the text classification option (0 - Bag of Words | 1 - Bernoulli Method): "))
lr = LogisticRegression(_lambdaVal, noOfIterations, 1, classifyOption, TRAIN_PATH, TEST_PATH)
#lr = LogisticRegression(4, 25, 1, 1)
lr.train()

spam_success_ratio, ham_success_ratio, total_success_ratio = lr.classify()
print('Logistic Regression Report')
print('Success Ratio For Spam Emails: %.4f%%' % (spam_success_ratio * 100))
print('Success Ratio For Ham Emails: %.4f%%' % (ham_success_ratio * 100))
print('Success Ratio For All Emails: %.4f%%' % (total_success_ratio * 100))

