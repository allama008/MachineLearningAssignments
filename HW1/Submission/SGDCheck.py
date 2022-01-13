#Start SGDClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import string
from nltk.corpus import stopwords
from multiprocessing.pool import ThreadPool as Pool

import numpy
import os
import pandas as pd

#Setting the root directory
TRAIN_PATH = "C:\\Users\\allam\\Documents\\Assignment\\6375\\Dataset\\Train\\hw1_train\\train"
TEST_PATH = "C:\\Users\\allam\\Documents\\Assignment\\6375\\Dataset\\Test\\hw1_test\\test"

def remove_punctuation_and_stopwords(sms):
    
    sms_no_punctuation = [ch for ch in sms if ch not in string.punctuation]
    sms_no_punctuation = "".join(sms_no_punctuation).split()
    
    sms_no_punctuation_no_stopwords = \
        [word.lower() for word in sms_no_punctuation if word.lower() not in stopwords.words("english")]
        
    return sms_no_punctuation_no_stopwords

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
SPHamDf['Spam'] = SPHamDf['HamSpamLabel'].map( {'spam': 1, 'ham': 0} ).astype(int)

sms_train, sms_test, label_train, label_test = train_test_split(SPHamDf['Email_Message'], SPHamDf['Spam'], random_state=5) #test_size=0.3
print(sms_train.head())

pipe_SGD = Pipeline([ ('bow'  , CountVectorizer(analyzer = remove_punctuation_and_stopwords) ),
                   ('tfidf'   , TfidfTransformer()),
                   ('clf_SGD' , SGDClassifier(random_state=5)),
                    ])

parameters_SGD = {
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    #'clf_SGD__max_iter': (5,10),
    'clf_SGD__alpha': (1e-05, 1e-04),
}

grid_SGD = GridSearchCV(pipe_SGD, parameters_SGD, cv=5,
                               n_jobs=None, verbose=1)

grid_SGD.fit(X=sms_train, y=label_train)

grid_SGD.best_params_
grid_SGD.best_score_
pred_test_grid_SGD = grid_SGD.predict(sms_test)
acc_SGD = accuracy_score(label_test, pred_test_grid_SGD)
print(acc_SGD)
print(grid_SGD.score(sms_test, label_test))