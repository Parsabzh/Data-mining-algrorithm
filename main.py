import numpy as np
from node import Node


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

def check_pure_node(arr):

    """Function that checks whether vector is pure, using the Gini index

    # Args:
    # arr: column of an array of dataset

    # Returns: Boolean."""
    result = np.all(arr == arr[0])
    if result:
        return True
    else:
        return False

def get_nfeat_cols(num_cols, nfeat):

    """function that returns a random subset (size nfeat) of the number of columns of a dataframr. Nfeat cannot be larger than the number of columns

      Args:
      total_col_nums: number of columns of a matrix/dataframe
      nfeat: parameter that determines the number of features that will be used for determining a split of the data

      Returns:
      random subset of number of columns.
      """
    if nfeat > num_cols:
        print("nfeat cannot be larger than the number of attributes")
    else:
        cols_nfeat = np.random.choice(np.arange(0, num_cols), size=nfeat, replace=False)
        return np.sort(cols_nfeat)

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

    # Loop over the split values to obtain the best
    for splt in splits:

        left = mat[mat[:,0] <= splt,1]
        right = mat[mat[:,0] > splt,1]

        imp = ((len(left) / length) * impurity(left)) + ((len(right) / length) * impurity(right))

        if((imp < bst_imp) & nmin_check(left, minleaf) & nmin_check(right, minleaf)):

            bst_splt = splt
            bst_imp = imp
    
    return bst_imp, bst_splt

def tree_grow(X, y, nmin, minleaf, nfeat):

    node = Node(X, y)
    node = split_node(node, nmin, minleaf, nfeat)
    return node

def create_childs(node, best_col, split_val):
    
    X_left = node.X[node.X[:, best_col] <= split_val, ]
    y_left = node.y[node.X[:, best_col]<= split_val, ]
    X_right = node.X[node.X[:, best_col]>split_val, ]
    y_right = node.y[node.X[:, best_col]>split_val, ]

    node.set_left_child(X_left, y_left)
    node.set_right_child(X_right, y_right)

    return node
    
def split_node(node, nmin, minleaf, nfeat):


    if not (nmin_check(node.y, nmin)):
        return node
    if check_pure_node(node.y):
        return node

    cols = get_nfeat_cols(np.shape(node.X)[1], nfeat)

    best_gini = 999
    best_col = -1
    split_val_best = None

    for i in cols:

        temp_gini, split_val = bestsplit(node.X[:,i].copy(), node.y.copy(), minleaf)
        
        if (best_gini > temp_gini):
            best_gini = temp_gini
            best_col = i
            split_val_best = split_val
    

    if(best_gini == 999):
        return node
    

    node.set_split_values(best_col, split_val_best)

    node = create_childs(node, best_col, split_val_best)

    # print(f"best col: {best_col}, split_val:{split_val_best}")
    # print("left")
    # print(node.left_child.X)
    # print(node.left_child.y)
    # print("right")
    # print(node.right_child.X)
    # print(node.right_child.y)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    split_node(node.left_child, nmin, minleaf, nfeat)
    split_node(node.right_child, nmin, minleaf, nfeat)

    return node



def maj_vote(y):
    """Takes the majority vote of the data points in the leaf. If number of labels are 
    equal, 0 is chosen.

    Args:
        y: Label values of the leaf

    Returns:
        Most occuring class
    
    """

    return np.bincount(y.astype(int)).argmax()


def tree_pred(x, tr):

    return traverse_node(tr, x)


def traverse_node(node, x):

    if(node.left_child is None):
        return maj_vote(node.y)

    if(x[node.split_col_num] < node.split_number):
        return traverse_node(node.left_child, x)
    else:
        return traverse_node(node.right_child, x)
    








print("\n----------------------")
print("START PROGRAM\n\n")




credit_data = np.genfromtxt('data.txt', delimiter=',', skip_header=True)

# print(credit_data)

print("\n \n")


tree = tree_grow(credit_data[:,:5].copy(), credit_data[:,5].copy(), 2, 1, 5)

res = tree_pred([25,0,1,32,0], tree)
print(f"res: {res}")

print( maj_vote(np.array([1,0])) )