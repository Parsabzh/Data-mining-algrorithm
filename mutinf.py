from pickle import FALSE
from preprocessing import split_data
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


dt_binary = pd.read_csv(f"data/converted_binary_unigram.csv")

dt_binary = dt_binary.drop('set_type', axis=1)
corr_matrix = dt_binary.corr()
print(corr_matrix["class_label"].sort_values(ascending=False)[0:10])
 


# corr = dt_binary.corr()['class_label'][:]

# def sorting(numbers_array):
#      return sorted(numbers_array, key=abs, reverse=True)

# print(sorting(corr[:30]))

# X_train_bin, y_train_bin, X_test_bin, y_test_bin = split_data(dt_binary)
# # X_train, y_train, X_test, y_test = split_data(dt)

# mut_ind_score = mutual_info_classif(X_train_bin, y_train_bin, discrete_features=True)

# mutual_info = pd.Series(mut_ind_score)
# mutual_info.index = X_train_bin.columns
# mutual_info = mutual_info.sort_values(ascending=False)
# print(mutual_info[0:20])