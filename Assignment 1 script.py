# packages
import numpy as np


# read in data
credit_data = np.genfromtxt('data.txt', delimiter=',', skip_header=True)

print(credit_data)
