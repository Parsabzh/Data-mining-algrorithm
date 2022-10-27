import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('nb_cv_results.csv')

print(data['avg_accuracy'].max())

graph = data.plot(x='features', y='avg_accuracy')
plt.show()