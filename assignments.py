import numpy as np



def impurity(arr):

    size = len(arr)
    counts = dict(zip(*np.unique(arr, return_counts=True)))

    if (0 in counts) & (1 in counts):
        return (counts[0] / size) * (counts[1] / size)
    else:
        print("test")
        return 0



array=np.array([1,0,1,1,1,0,0,1,1,0,1])

print(impurity(array))

array=np.array([0,0,0,0,0,0,0,0,0,0,0])

print(impurity(array))
