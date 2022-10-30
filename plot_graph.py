import matplotlib.pyplot as plt
import pandas as pd



NGRAM = 'unigram'
data = pd.read_csv(f"results/rf_{NGRAM}_estimators_plot.csv")

data['cumulative_max'] = data['avg_accuracy'].expanding().max()

data.to_csv(f"results/rf_{NGRAM}_estimators_plot.csv")

print(data['avg_accuracy'].max())

graph = data.plot(x='trees', y=['avg_accuracy', 'cumulative_accuracy', 'cumulative_max'])
plt.show()