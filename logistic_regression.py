from sklearn.linear_model import LogisticRegression
from preprocessing import split_data
import pandas as pd
import optuna
from sklearn.metrics import confusion_matrix, classification_report
import sklearn
import numpy as np
from sklearn.feature_selection import mutual_info_classif

#####
### Tuning for the best parameters
#####

np.random.seed(0)

dt = pd.read_csv('converted.csv')
X_train, y_train, X_test, y_test = split_data(dt)

mut_ind_score = mutual_info_classif(X_train,y_train, discrete_features=True)

mutual_info = pd.Series(mut_ind_score)
mutual_info.index = X_train.columns
mutual_info = mutual_info.sort_values(ascending=False)

# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
# def objective(trial):


#     # C = trial.suggest_float("C", 1e-10, 1e10, log=True)
#     C = trial.suggest_int("C", 1, 4991, 10)
#     feat = trial.suggest_int("feat", 1, 2431, 10)


#     selected = mutual_info[:feat]

#     X_selected = X_train.loc[:, selected.index]


#     LogReg = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', C=C)

#     score = sklearn.model_selection.cross_val_score(LogReg, X_selected, y_train, n_jobs=-1, cv=10)
#     accuracy = score.mean()
#     return accuracy


# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=500)
# print(study.best_trial)


# LogReg = LogisticRegression(random_state=0, penalty='l1', solver='liblinear')
# param_distributions = {
#     # "C": optuna.distributions.FloatDistribution(1e-10, 1e10, log=True)
#     "C": optuna.distributions.FloatDistribution(0.001, 1000, log=True)

# }
# optuna_search = optuna.integration.OptunaSearchCV(LogReg, param_distributions, cv=20, n_trials=100, return_train_score=True)


# dt = pd.read_csv('converted.csv')
# X_train, y_train, X_test, y_test = split_data(dt)


# optuna_search.fit(X_train, y_train)
# y_pred = optuna_search.predict(X_train)

## best found parameters:
## accuracy: 0.83125 and parameters: {'C': 0.83125}



#####
### Doing the actual experiment
#####


# dt = pd.read_csv('converted.csv')
# X_train, y_train, X_test, y_test = split_data(dt)

selected = mutual_info[:791]

# X_selected = X_train.loc[:, selected.index]

clf = LogisticRegression(random_state=0, C=4651, penalty='l1', solver='liblinear').fit(X_train.loc[:, selected.index], y_train)
y_pred = clf.predict(X_test.loc[:, selected.index])

print("\nLogistic Regression")
print(f"\nconfusion matrix:\n{confusion_matrix(y_test, y_pred)}")
print(classification_report(y_test, y_pred))




# dt = pd.read_csv('converted.csv')
# X_train, y_train, X_test, y_test = split_data(dt)

# c_values = [*range(1, 5000, 10)]
# scores = []

# for c in c_values:


#     clf_nb = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', C=c)

#     scorelist = sklearn.model_selection.cross_val_score(clf_nb, X_train, y_train, cv=10)

#     print(f"{c} with average: {sum(scorelist) / len(scorelist)}")
#     scores.append(sum(scorelist) / len(scorelist))



# result = pd.DataFrame({"C": c_values, "avg_accuracy": scores})
# result.to_csv('lr_cv_results.csv')






