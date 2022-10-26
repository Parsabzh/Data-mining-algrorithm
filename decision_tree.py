
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
import optuna
import sklearn

dt=pd.read_csv('converted.csv')
X_train = dt.loc[dt['set_type'] == 'train', ~dt.columns.isin(['class_label', 'set_type'])]
y_train = dt.loc[dt['set_type'] == 'train', dt.columns.isin(['class_label'])]
X_test = dt.loc[dt['set_type'] == 'test', ~dt.columns.isin(['class_label', 'set_type'])]
y_test = dt.loc[dt['set_type'] == 'test', dt.columns.isin(['class_label'])]

def optuna_off(X_train,y_train):
    model= DecisionTreeClassifier(max_depth =9, random_state = 15)
    score= model_selection.cross_val_score(model, X_train, y_train, cv=10)
    print(score.mean())


def objective(trial):
    dt=pd.read_csv('converted.csv')
    X_train = dt.loc[dt['set_type'] == 'train', ~dt.columns.isin(['class_label', 'set_type'])]
    y_train = dt.loc[dt['set_type'] == 'train', dt.columns.isin(['class_label'])]
    X_test = dt.loc[dt['set_type'] == 'test', ~dt.columns.isin(['class_label', 'set_type'])]
    y_test = dt.loc[dt['set_type'] == 'test', dt.columns.isin(['class_label'])]
    rf_random_state = trial.suggest_int("rf_random_state", 2, 1000)
    rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
    classifier_obj = DecisionTreeClassifier(
        max_depth=rf_max_depth,random_state=rf_random_state
    )
    score = model_selection.cross_val_score(classifier_obj, X_train, y_train.values, n_jobs=-1, cv=10)
    accuracy = score.mean()
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_trial)

optuna_off(X_train,y_train)