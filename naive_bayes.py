from sklearn.naive_bayes import MultinomialNB
from preprocessing import split_data
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt


dt = pd.read_csv('converted.csv')
# print(dt)


X_train, y_train, X_test, y_test = split_data(dt)


mut_ind_score = mutual_info_classif(X_train,y_train)
# print(mut_ind_score)

mutual_info = pd.Series(mut_ind_score)
mutual_info.index = X_train.columns


# print(mutual_info.sort_values(ascending=False))

# print(mutual_info.sort_values(ascending=False)[0:20])
# print(mutual_info.sort_values(ascending=False)[0:20].plot.bar(figsize=(20, 8)))

sel_five_cols = SelectKBest(mutual_info_classif, k=10)
sel_five_cols.fit(X_train, y_train)
print(X_train.columns[sel_five_cols.get_support()])




# Train NB model

# clf_nb = MultinomialNB()
# clf_nb.fit(X_train, y_train)
# print(clf_nb.score(X_test, y_test))

# select top x features using 
