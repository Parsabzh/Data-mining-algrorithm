from pickle import TRUE
import numpy as np

from main import tree_grow



# # 1.  
# def nmin_check(obs, nmin):

#     """checks if y has more that nmin values. If not, node must not be splitted and tree will not
#    be grown further. boolean output.

#     # Args:
#     # obs: column of an array of dataset
#     # nmin: parameter that sets the minimum number of observations in data required to make a split

#     # Returns:
#     # Boolean, returns true if there are more observations than nmin."""

#     if len(obs) > nmin:
#         return True
#     else:
#         return False

# # check
# array = np.array([1,0,1,1])
# min_value = nmin_check(array, 5)
# print(min_value)



# def check_impurity(arr):

#     """Function that checks the impurity of a vector, using the Gini index

#     # Args:
#     # arr: column of an array of dataset

#     # Returns:impurity
#     # impurity."""

#     size = len(arr)
#     counts = dict(zip(*np.unique(arr, return_counts=True)))

#     if (0 in counts) & (1 in counts):
#         return (counts[0] / size) * (counts[1] / size)
#     else:
#         return 0

# #check
# array=np.array([1,0,1,1,1,0,0,1,1,0,1])
# array=np.array([0,0,0,0,0,0,0,0,0,0,0])
# check_impurity(array)




# def check_pure_node(arr):

#     """Function that checks whether vector is pure, using the Gini index

#     # Args:
#     # arr: column of an array of dataset

#     # Returns: Boolean."""
#     result = np.all(arr == arr[0])
#     if result:
#         return True
#     else:
#         return False


# array=np.array([1,0,1,1,1,0,0,1,1,0,1])

# array=np.array([0,0,0,0,0,0,0,0,0,0,0])
# print(check_pure_node(array))


   

# def get_nfeat_cols(num_cols, nfeat):

#     """function that returns a random subset (size nfeat) of the number of columns of a dataframr. Nfeat cannot be larger than the number of columns

#       Args:
#       total_col_nums: number of columns of a matrix/dataframe
#       nfeat: parameter that determines the number of features that will be used for determining a split of the data

#       Returns:
#       random subset of number of columns.
#       """
#     if nfeat > num_cols:
#         print("nfeat cannot be larger than the number of attributes")
#     else:
#         cols_nfeat = np.random.choice(np.arange(0, num_cols), size=nfeat, replace=False)
#         return cols_nfeat

# #check
# credit_data = np.genfromtxt('data.txt', delimiter=',', skip_header=True)
# cols = get_nfeat_cols(np.shape(credit_data)[1], 6)
# print(cols)


# def tree_grow_b (X, y, nmin, minleaf, nfeat, m):
#     treelist = []
#     merged_matrix = np.column_stack((X,y))
#     for i in range(m):

#         bootstrap = np.random.choice(np.arange(0, len(X)), size=len(X), replace=True)

#         # take bootstrap (total row, but ith replacement)
#         X = merged_matrix[:,:-1]
#         y = merged_matrix[:,-1]
#         tree_i = tree_grow(X, y, nmin, minleaf, nfeat)
#         treelist.append(tree_i)
#     return treelist


credit_data = np.genfromtxt('data.txt', delimiter=',', skip_header=True)
X = credit_data[:,:-1]
num_rows = np.shape(X)[0]
print(X)
print(num_rows)

y = credit_data[:,-1]
print(y)

# merged_matrix = np.column_stack((X,y))
# print(merged_matrix)

# bootstrap = np.random.choice(np.arange(0, np.shape(merged_matrix)[0]), size=len(X), replace=True)
