from pickle import FALSE
from preprocessing import split_data
import pandas as pd


dt_binary = pd.read_csv(f"data/converted_binary_unigram.csv")
dt_binary = dt_binary.drop('set_type', axis=1)


corr = dt_binary.corr()['class_label'][:]

def sorting(numbers_array):
     return sorted(numbers_array, key=abs, reverse=True)

print(sorting(corr[:30]))