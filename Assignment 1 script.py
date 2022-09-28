# packages
import numpy as np


def nmin_check(obs, nmin):

    """checks if y has more that nmin values. If not, node must not be splitted and tree will not
   be grown further. boolean output.

    Args:
        obs: column of an array of dataset
        nmin: parameter that sets the minimum number of observations in data required to make a split

    Returns:
        Boolean, returns true if there are more observations than nmin."""

    if len(obs) >= nmin:
        return True
    else:
        return False


def impurity(arr):
    """
    Function that checks the impurity of a vector, using the Gini index

    # Args:
    # arr: column of an array of dataset

    # Returns:impurity
    # impurity.
    # """

    size = len(arr)

    counts = dict(zip(*np.unique(arr, return_counts=True)))
    
    if (0 in counts) & (1 in counts):
        return (counts[0] / size) * (counts[1] / size)
    else:
        return 0


def bestsplit(x_col, y, minleaf):
    """Given a single column, it finds the best split value with the lowest impurity.
    Note that it does not work if there is only a single row in the input X.

    Args:
        x_col: column from the X data matrix
        y: The labels of the data set.
        minleaf: parameter that sets the minimum number of observations in data required to make a split

    Returns:
        bst_imp: lowest obtained gini value. 999 if nothing is found.
        bst_splt: split value that obtains the lowest gini impurity. None if not found."""

    # Find the split values
    x_col_sor = np.sort(np.unique(x_col))
    s = len(x_col_sor)
    splits = (x_col_sor[0:s-1]+x_col_sor[1:s])/2

    # create a sorted matrix of both the column and the labels
    mat = np.column_stack((x_col,y))
    mat = mat[mat[:, 0].argsort()]

    bst_splt = None
    bst_imp = 999

    length = len(x_col)

    print(f"length {length}")

    # Loop over the split values to obtain the best
    for splt in splits:

        left = mat[mat[:,0] <= splt,1]
        right = mat[mat[:,0] > splt,1]

        # print("\n\n-----")
        # print(mat[mat[:,0] <= splt,:])
        # print(f"l_left: {len(left)}, fac_lect:{round((len(left) / s),2)}, imp_left: {round(impurity(left),2)} tot_imp:{((len(left) / s) * impurity(left))}")
        # print(mat[mat[:,0] > splt,:])
        # print(f"l_right: {len(right)}, fac_lect:{round((len(right) / s),2)}, imp_right: {round(impurity(right),2)} tot_imp:{((len(right) / s) * impurity(right))}")

        imp = ((len(left) / length) * impurity(left)) + ((len(right) / length) * impurity(right))
        # print(f"total impurity: {round(((len(left) / s) * impurity(left)) + ((len(right) / s) * impurity(right)),2)}, length: {s}")
        if((imp < bst_imp) & nmin_check(left, minleaf) & nmin_check(right, minleaf)):

            bst_splt = splt
            bst_imp = imp
    
    return bst_imp, bst_splt


# read in data
COLNUM = 3
credit_data = np.genfromtxt('data.txt', delimiter=',', skip_header=True)

bst_imp, bst_splt = bestsplit(credit_data[:,COLNUM].copy(), credit_data[:,5].copy(), 1)
print(bst_imp, bst_splt)


mat = np.column_stack((credit_data[:,COLNUM],credit_data[:,5]))
mat = mat[mat[:, 0].argsort()]

left = mat[mat[:,0] <= bst_splt,:]
right = mat[mat[:,0] > bst_splt,:]

print("Left")
print(left)
print("Right")
print(right)



