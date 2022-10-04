import numpy as np
from node import Node
from sklearn.metrics import confusion_matrix


def tree_grow(X, y, nmin, minleaf, nfeat):
    """
    Main tree growing function.

    Function that creates a node and begins the node splitting
    procedure.

    Parameters
    ----------
    X : numpy.ndarray
        Attributes.
    y : numpy.ndarray
        Array of labels.
    nmin : int
        Parameter that sets the minimun number of observations required before splitting.
    minleaf : int
        Parameter that sets the minimum number of observations required after a split.
    nfeat : int
        Number of attributes that are considered for every split.

    Returns
    -------
    Node
        A decision tree
    """

    node = Node(X, y)
    node = split_node(node, nmin, minleaf, nfeat)
    return node

def tree_pred(X, tr):
    """
    Main tree prediction function.

    Function that, given a dataframe consisting of attributes, predicts 
    the class (i.e., calls the row specific prediction function) for every row. 

    Parameters
    ----------
    X : numpy.ndarray
        Attributes.
    tr : Node
        Tree object.

    Returns
    -------
    numpy.ndarray
        A vector containing the predicted labels.
    """
    return np.apply_along_axis(traverse_node, 1, X, tr)

def tree_grow_b (X, y, nmin, minleaf, nfeat, m):
    treelist = [] # empty list, to be filled with tree objects
    merged_matrix = np.column_stack((X,y)) # merge X and y
    for i in range(m):
        num_rows = np.shape(merged_matrix)[0] # take nrows
        bootstrap = np.random.choice(np.arange(0, num_rows), size=num_rows, replace=True) #bootstrap data
        bootstrapped_data = merged_matrix[bootstrap,]
        X = bootstrapped_data[:,:-1] # separate X and y
        y = bootstrapped_data[:,-1]
        tree_i = tree_grow(X, y, nmin, minleaf, nfeat) # create tree for bootstrapped data
        treelist.append(tree_i) # append grown tree to list
    return treelist

def tree_pred_b(trees, x):
    """
    Tree ensemble method.

    Function that, given an ensemble of trees, takes the majority
    vote of all the predictions made by a single tree.
    
    Parameters
    ----------
    X : numpy.ndarray
        Attributes.
    tr : Node
        Tree object.

    Returns
    -------
    numpy.ndarray
        A vector containing the predicted labels.
    """

    y_frame = np.array([]).reshape(len(x),0)
    for tree in trees:
        y = tree_pred(x.copy(), tree)

        y_frame = np.column_stack((y_frame, y))
    
    
    return np.apply_along_axis(maj_vote, 1, y_frame)

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
    Checks impurity.

    This function calculates the impurity of a given node using the 
    Gini index. We use the formula for binary problems provided in the 
    lecture slides: i(t) = p(0|t)p(1|t).

    Parameters
    ----------
    arr : numpy.array
        Array of labels.

    Returns
    -------
    float
        The impurity of the given array.
    """

    size = len(arr)

    counts = dict(zip(*np.unique(arr, return_counts=True)))
    
    if (0 in counts) & (1 in counts):
        return (counts[0] / size) * (counts[1] / size)
    else:
        return 0

def check_pure_node(arr):

    """Function that checks whether vector is pure, using the Gini index

    # Args:
    # arr : column of an array of dataset

    # Returns: Boolean."""
    result = np.all(arr == arr[0])
    if result:
        return True
    else:
        return False

def get_nfeat_cols(num_cols, nfeat):

    """function that returns a random subset (size nfeat) of the number of columns of a dataframr. Nfeat cannot be larger than the number of columns

      Args:
      total_col_nums : number of columns of a matrix/dataframe
      nfeat : parameter that determines the number of features that will be used for determining a split of the data

      Returns:
      random subset of number of columns.
      """
    if nfeat > num_cols:
        print("nfeat cannot be larger than the number of attributes")
    else:
        cols_nfeat = np.random.choice(np.arange(0, num_cols), size=nfeat, replace=False)
        return np.sort(cols_nfeat)

def bestsplit(x_col, y, minleaf):

    """
    Finds the best split in column.

    Given a single column, it finds the best split value with the lowest impurity.

    Parameters
    ----------
    x_col : numpy.array
        Array of attribute values.
    y : numpy.array
        Array of labels.
    minleaf : int
        Parameter that sets the minimum number of observations required after a split.

    Returns
    -------
    float
        Lowest obtained gini value. 999 if nothing is found.
    int
        Split value that obtains the lowest gini impurity. None if not found.

    """

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

def create_childs(node, best_col, split_val):
    
    X_left = node.X[node.X[:, best_col] <= split_val, ]
    y_left = node.y[node.X[:, best_col]<= split_val, ]
    X_right = node.X[node.X[:, best_col]>split_val, ]
    y_right = node.y[node.X[:, best_col]>split_val, ]

    node.set_left_child(X_left, y_left)
    node.set_right_child(X_right, y_right)

    return node
    
def split_node(node, nmin, minleaf, nfeat):
    """
    Node splitting function.

    Function that controls the growing of the tree. It splits a
    certain node until the one of the criteria has been satisfied: 
    (1) a node is pure, (2) there are less than nmin items or 
    (3) no split can be found due to the minleaf constraint. A split is
    found by iterating over the - by nfeat selected - columns using the
    bestsplit function.

    Parameters
    ----------
    node : Node
        A Node object containing the data.
    nmin : int
        Parameter that sets the minimun number of observations required before splitting.
    minleaf : int
        Parameter that sets the minimum number of observations required after a split.
    nfeat : int
        Number of attributes that are considered for every split.

    Returns
    -------
    Node
        A Node object containing possibly many lower-level nodes.
    """


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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    split_node(node.left_child, nmin, minleaf, nfeat)
    split_node(node.right_child, nmin, minleaf, nfeat)

    return node

def maj_vote(y):
    """
    Majority vote.

    Function that calculates the majority vote in a given array. If there
    are multiple classes with the same number, the first class (i.e., 0) is voted. 

    Parameters
    ----------
    y : numpy.array
        Array of labels.

    Returns
    -------
    int
        The class of elected through majority vote.
    """

    return np.bincount(y.astype(int)).argmax()

def traverse_node(x, node):
    """
    Search through the tree.

    Function that searches through the tree to find the leaf 
    node that fits the parameters in the x vector. Using the node 
    parameters, it decides to continue in the left or right node 
    until a leaf node is found.
    
    Parameters
    ----------
    x : numpy.ndarray
        Vector of attributes; a single row from X.
    node : Node
        (Partial) tree object.

    Returns
    -------
    numpy.ndarray
        A vector containing the predicted labels.
    """

    if(node.left_child is None):
        return maj_vote(node.y)

    if(x[node.split_col_num] < node.split_number):
        return traverse_node(x, node.left_child)
    else:
        return traverse_node(x, node.right_child)





print("----------------------")
print("START PROGRAM\n")


######## Testing for the hint for a single tree
# indian_data = np.genfromtxt('indians.txt', delimiter=',')
# trees = tree_grow(indian_data[:,:-1].copy(), indian_data[:,-1].copy(), 20, 5, 8)
# y_pred = tree_pred(indian_data[:,:-1].copy(), trees)
# y_actu = indian_data[:,-1]
# print(confusion_matrix(y_actu, y_pred))


######## PART 2 
# train_data = np.genfromtxt('eclipse-metrics-packages-2.0.csv', delimiter=';', skip_header=True)
# test_data = np.genfromtxt('eclipse-metrics-packages-2.0.csv', delimiter=';', skip_header=True)
# print(train_data.shape)
# print(test_data.shape)


