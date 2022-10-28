
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import optuna

dt=pd.read_csv('data/converted.csv')
X_train = dt.loc[dt['set_type'] == 'train', ~dt.columns.isin(['class_label', 'set_type'])]
y_train = dt.loc[dt['set_type'] == 'train', dt.columns.isin(['class_label'])]
X_test = dt.loc[dt['set_type'] == 'test', ~dt.columns.isin(['class_label', 'set_type'])]
y_test = dt.loc[dt['set_type'] == 'test', dt.columns.isin(['class_label'])]

# model= RandomForestClassifier(random_state=42)
# model.fit(X_train,y_train.values.ravel())
# score=model.score(X_train,y_train.values.ravel())
# print(score)

# def objective(trial):
#     # rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 1000)
#     rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
#     max_feat = trial.suggest_int("rf_max_feat", 10, 1000, log=True)
#     classifier_obj = RandomForestClassifier(
#         max_depth=rf_max_depth, n_estimators=150,max_features=max_feat
#     )
#     score = model_selection.cross_val_score(classifier_obj, X_train, y_train.values.ravel(), n_jobs=-1, cv=3)
#     accuracy = score.mean()
#     return accuracy

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)



RandomForestClassifier(max_depth=30, n_estimators=150,max_features=40)
model= RandomForestClassifier(random_state=42)
model.fit(X_train,y_train.values.ravel())
score=model.score(X_test,y_test.values.ravel())
print(score)
