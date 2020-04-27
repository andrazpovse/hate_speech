# Machine learning libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from imblearn.over_sampling import SMOTE

from pretrained_regression import PretrainedRegression

# Plotting libraries and setup
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)

"""
We implement a basic logistic regression classifier to distinguish tweets
including and not including cyberbullying, inspired by the work of
Ho et al. (2020) and Kasture (2015).

Framework:

1) Get the input data
2) Parse it from CSV
2.1) Upsample the data to become balanced
3) Split the data into a train and test set
4) Train the model on the training set
5) Evaluate on the testing set
"""

# Enable/disable oversampling with SMOTE to get a balanced training set
OVERSAMPLE = False

# Fit own model with sklearn instead of using precalculated weights
FIT_NEW_MODEL = False

if __name__ == "__main__":
    tweet_data = pd.read_csv('../Data/Twitter/CleanedCyberBullyingTweets_LIWC2007.csv')

    # We change classes 'yes' and 'no' to computationally more friendly values, KEKW
    class_mapping = {'no': 0, 'yes': 1}
    tweet_data['class'].replace(class_mapping, inplace=True)

    # The original data is imbalanced (more samples for 'no' class than 'yes')
    # To check the count plot and see which class prevails, uncomment the lines below
    """
    sns.countplot(x='class', data=tweet_data, palette='hls')
    plt.show()
    """

    # We keep only the columns that the article mentions as the ones we use
    keep_only_columns = ['you', 'anger', 'negemo', 'anger', 'bio', 'body', 'health', 'ingest', 'death', 'swear', 'sexual', 'class']
    tweet_data = tweet_data[keep_only_columns]

    X = tweet_data.drop(columns=['class'])
    y = tweet_data[['class']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    if OVERSAMPLE:
        # We oversample only the training data, so that the training phase doesn't have
        # any effect on the test data which has to remain as real as possible
        oversampler = SMOTE(random_state=0)
        columns = X_train.columns

        X_train, y_train = oversampler.fit_sample(np.array(X_train), np.array(y_train))
        X_train = pd.DataFrame(data=X_train, columns=columns)
        y_train = pd.DataFrame(data=y_train, columns=['class'])

        # To review the effect of the oversampling, uncomment the lines below
        """
        print('The length of the oversampled data is ', len(X_train))
        print('The length of the no class in the data is ', len(y_train[y_train==0]))
        print('The length of the yes class in the data is ', len(y_train[y_train==1]))
        print('The proportion of the no vs. the yes class is ', len(y_train[y_train==0]) / len(y_train[y_train==1]))
        """

    # Convert pandas dataframes to numpy arrays to prevent sklearn crying about improper formats
    X_train = X_train.values
    y_train = y_train['class'].values
    X_test = X_test.values
    y_test = y_test['class'].values

    if FIT_NEW_MODEL:
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)

        y_pred = logreg.predict(X_test)

        print('Accuracy of LogRegression on the test set: ', logreg.score(X_test, y_test))

        print("----Confusion matrix----")
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        print(confusion_matrix)

    else:
        pr = PretrainedRegression()
        y_pred = pr.predict(X_test)
        print('Accuracy of LogRegression on the test set: ', pr.score(X_test, y_test))

        print("----Confusion matrix----")
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        print(confusion_matrix)
