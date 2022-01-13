Keep all the three files in same folder
1) mlassign.py
2) MultiNaiveBayes.py
3) DiscreteNaiveBayes.py
4) LRFinalVersion.py
5) stop_words.txt
6) SGDCheck.py

To Execute Naïve Bayes

mlassign.py will execute both Multinomial Naïve Bayes and Discrete Naïve Bayes:
1. Kindly replace the values of the variable TRAIN_PATH and TEST_PATH in lines 16 and 17 with the complete path of training data and test data respectively. The path should be upto the folder that contains the ham and spam folder.

To Execute Logistic Regression

mlassign.py will execute Logistic Regression.
1. The command line will prompt for the below mentioned options to be entered.
	a. Regularization parameter (lambda)
	b. Number of iterations to execute
	c. Text Classification Option (0 for Bag of Words and 1 for Bernoulli)

The parameters for Logistic Regression were tuned manually. The process could not be automated due to shortage of time.

Accuracy of Multinomial Naive Bayes : 58%
Accuracy of Discrete Naive Bayes : 56%
Accuracy of Logistic Regression (Bag of Words): 92%
Accuracy of Logistic Regression (Bernoulli Method): 93%

Accuracy of SGDClassifier: 96%

1. Bag of Words was the best representation as it used actual weights instead of binary values placing equal importance to words across the data.

Logistic Regression performed better than the Naive Bayes Algorithms as Naive Bayes assumes that the features are conditionally independent. Real data sets can never be independent. Logistic Regression on the other hand has less assumptions. Naive Bayes has a higher bias but lower variance compared to logistic regression. Furthermore regularization is added to Logistic Regression model to force the model toward lower values of the parameters.
Accuracy of Multinomial Naive Bayes : 58%
Accuracy of Discrete Naive Bayes : 56%
Accuracy of Logistic Regression: 92%

Logistic Regression performed best on Bernoulli model because it assumes the response is conditionally Bernoulli distributed, given the values of the features. Logistic regression models the mean of a Bernoulli.

No, the implemented Logistic Regression model does not outperform the SGDClassifier.
Accuracy of Logistic Regression: 92%
Accuracy of SGDClassifier: 96%

For any queries, kindly contact mah200001@utdallas.edu