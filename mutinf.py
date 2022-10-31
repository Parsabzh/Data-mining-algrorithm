from pickle import FALSE
from preprocessing import split_data
import pandas as pd


dt_binary = pd.read_csv(f"data/converted_binary_unigram.csv")

corr = dt_binary[dt_binary.columns[1:]].corr()['class'][:]

def sorting(numbers_array):
     return sorted(numbers_array, key=abs, reverse=True)

print(sorting(corr[:30]))