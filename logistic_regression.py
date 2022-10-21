from sklearn.linear_model import LogisticRegression
from preprocessing import split_data
import pandas as pd
import optuna
from sklearn.metrics import confusion_matrix, classification_report
import sklearn


#####
### Tuning for the best parameters
#####



# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    dt = pd.read_csv('converted.csv')
    X_train, y_train, X_test, y_test = split_data(dt)

    C = trial.suggest_float("C", 1e-10, 1e10, log=True)

    LogReg = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', C=C)

    score = sklearn.model_selection.cross_val_score(LogReg, X_train, y_train, n_jobs=-1, cv=10)
    accuracy = score.mean()
    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_trial)


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

# clf = LogisticRegression(random_state=0, C=962.2997782745615, penalty='l1', solver='liblinear').fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# print("\nLogistic Regression")
# print(f"\nconfusion matrix:\n{confusion_matrix(y_test, y_pred)}")
# print(classification_report(y_test, y_pred))









