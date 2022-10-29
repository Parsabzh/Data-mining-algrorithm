from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier




from preprocessing import split_data, create_CV
import pandas as pd
import optuna
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score



ITERATIONS = 10 # number of iterations per classifier - ngram pair

used_features = {
                  'naive_bayes':
                    {
                        'unigram': 735, 
                        'bigram': 1505 
                    },
                'logistic_regression':
                    {
                        'unigram': 881, 
                        'bigram': 4621
                    }, 
                'random_forest':
                    {
                        'unigram': 641, 
                        'bigram': 971
                    },
                'decision_tree':
                    {
                        'unigram': 1010, 
                        'bigram': 441
                    } 
                }   


classifiers = { 'naive_bayes': 
                    {
                        'unigram': MultinomialNB(),
                        'bigram': MultinomialNB()
                    },
                'logistic_regression':
                    {
                        'unigram': LogisticRegression(C=4451, penalty='l1', solver='liblinear'),
                        'bigram': LogisticRegression(C=1581, penalty='l1', solver='liblinear')
                    },
                'random_forest':
                    {
                        'unigram': RandomForestClassifier(max_depth=None, n_estimators=999999,max_features=30),
                        'bigram': RandomForestClassifier(max_depth=70, n_estimators=9999999,max_features=10)
                    },
                'decision_tree':
                    {
                        'unigram': DecisionTreeClassifier(max_depth=20,max_features=11),
                        'bigram': DecisionTreeClassifier(max_depth=60,max_features=330)
                    }
            }


classifiers = ['naive_bayes', 'logistic_regression', 'random_forest', 'decision_tree']



dt = pd.read_csv(f"data/converted_count_{NGRAM}.csv")
dt_binary = pd.read_csv(f"data/converted_binary_{NGRAM}.csv")

X_train_bin, y_train_bin, X_test_bin, y_test_bin = split_data(dt_binary)

X_train, y_train, X_test, y_test = split_data(dt)

mut_ind_score = mutual_info_classif(X_train_bin,y_train_bin, discrete_features=True)

mutual_info = pd.Series(mut_ind_score)
mutual_info.index = X_train.columns
mutual_info = mutual_info.sort_values(ascending=False)



for c, classifier in enumerate(classifiers):

    f = used_features[c]

    for vect in ['unigram', 'bigram']:

        feats = f[vect]

        for i in range(ITERATIONS):










######
###### Logistic regression
######

# C=3951
# feat_num=1751

# selected = mutual_info[:feat_num]

# clf = LogisticRegression(C=C, penalty='l1', solver='liblinear').fit(X_train.loc[:, selected.index], y_train)
# y_pred = clf.predict(X_test.loc[:, selected.index])

# print(f"\nLogistic Regression using C={C} and {feat_num} features")
# print(f"\nconfusion matrix:\n{confusion_matrix(y_test, y_pred)}")
# print(classification_report(y_test, y_pred))
