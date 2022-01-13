import os
import math
import re

from typing import Dict

class MultiNaiveBayes:
    def __init__(self, BoGTest, testDir):
        p_spam = float
        p_ham = float
        parameters_ham = dict()

    def train(SPHamDfBoG, vocabulary):
        # Isolating spam and ham messages first
        spam_messages = SPHamDfBoG[SPHamDfBoG['HamSpamLabel'] == 'spam']
        ham_messages = SPHamDfBoG[SPHamDfBoG['HamSpamLabel'] == 'ham']

        # P(Spam) and P(Ham)
        p_spam = len(spam_messages) / len(SPHamDfBoG)
        p_ham = len(ham_messages) / len(SPHamDfBoG)

        # N_Spam
        n_words_per_spam_message = spam_messages['Email_Message'].apply(len)
        n_spam = n_words_per_spam_message.sum()

        # N_Ham
        n_words_per_ham_message = ham_messages['Email_Message'].apply(len)
        n_ham = n_words_per_ham_message.sum()

        # N_Vocabulary
        n_vocabulary = len(vocabulary)

        # Laplace smoothing
        alpha = 1

        # Initiate parameters
        parameters_spam = {unique_word:0 for unique_word in vocabulary}
        parameters_ham = {unique_word:0 for unique_word in vocabulary}

        #print(type(parameters_ham))

        # Calculate parameters
        for word in vocabulary:
            n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
            p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
            parameters_spam[word] = p_word_given_spam

            n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
            p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
            parameters_ham[word] = p_word_given_ham

        return p_ham, p_spam, parameters_ham, parameters_spam

    def classifyTest(message, p_ham, p_spam, parameters_ham, parameters_spam):
        '''
        message: a string
        '''

        message = re.sub('\W', ' ', message)
        message = message.lower().split()

        p_spam_given_message = p_spam
        p_ham_given_message = p_ham

        for word in message:
            if word in parameters_spam:
                p_spam_given_message *= parameters_spam[word]

            if word in parameters_ham:
                p_ham_given_message *= parameters_ham[word]

        if p_ham_given_message > p_spam_given_message:
            return 'ham'
        elif p_spam_given_message > p_ham_given_message:
            return 'spam'
        else:
            return 'needs human classification'

