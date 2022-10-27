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
from sklearn.model_selection import cross_val_score

np.random.seed(0)

dt = pd.read_csv('converted.csv')
# print(dt)


X_train, y_train, X_test, y_test = split_data(dt)


mut_ind_score = mutual_info_classif(X_train,y_train, discrete_features=True)

mutual_info = pd.Series(mut_ind_score)
mutual_info.index = X_train.columns
mutual_info = mutual_info.sort_values(ascending=False)


print("ready with  generating mutual info: \n\n")

# feats = [*range(5, len(mutual_info)-1, 5)]
feats = [*range(5, 5000, 5)]

scores = []

for feat in feats:

    selected = mutual_info[:feat]

    X_selected = X_train.loc[:, selected.index]

    clf_nb = MultinomialNB()

    scorelist = cross_val_score(clf_nb, X_selected, y_train, cv=10)

    print(f"{feat} with average: {sum(scorelist) / len(scorelist)}")
    scores.append(sum(scorelist) / len(scorelist))

    # print(clf_nb.score(X_test, y_test))


result = pd.DataFrame({"features": feats, "avg_accuracy": scores})
result.to_csv('nb_cv_results.csv')


# print(mutual_info.sort_values(ascending=False))

# print(mutual_info.sort_values(ascending=False)[0:20])
# print(mutual_info.sort_values(ascending=False)[0:20].plot.bar(figsize=(20, 8)))

# sel_five_cols = SelectKBest(mutual_info_classif, k=10)
# sel_five_cols.fit(X_train, y_train)
# print(X_train.columns[sel_five_cols.get_support()])




# Test NB model



# selected = mutual_info[:len(mutual_info)]

# clf_nb = MultinomialNB()


# clf_nb.fit(X_train.loc[:,selected.index], y_train)
# print(clf_nb.score(X_test.loc[:,selected.index], y_test))


