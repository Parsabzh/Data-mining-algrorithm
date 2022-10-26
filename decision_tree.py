
from distutils import core
import pandas as pd
from sklearn import tree

dt=pd.read_csv('converted.csv')
X_train = dt.loc[dt['set_type'] == 'train', ~dt.columns.isin(['class_label', 'set_type'])]
y_train = dt.loc[dt['set_type'] == 'train', dt.columns.isin(['class_label'])]
X_test = dt.loc[dt['set_type'] == 'test', ~dt.columns.isin(['class_label', 'set_type'])]
y_test = dt.loc[dt['set_type'] == 'test', dt.columns.isin(['class_label'])]

model= tree.DecisionTreeClassifier(max_depth =9, random_state = 15)
model.fit(X_train,y_train)
score=model.score(X_test,y_test)
print(score)


