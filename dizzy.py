from turtle import right
import numpy as np
from node import Node


data = np.genfromtxt('data.txt', delimiter=',', skip_header=True)
c = np.shape(data)[1] # number of columns

X = data[:,:c-1]
y = data[:,c-1]


node = Node(X, y)


# print("Dit is het X frame in de node")
# print(node.X)

# print("Dit zijn de y labels in de node")
# print(node.y)
# print("======================")

# Je kunt nu de Node "node" gebruiken als input.
# Ik heb even wat geprint zodat je een idee krijgt
# Meer kun je vinden in het bestand node.py


# create_childs(Node node, int best_col, double split_val)

#     function that, given the split_col and split_val, splits the data set in two parts and creates two new Node 
#     objects (including their X, y attributes only). These objects are added to the node as left/right child objects.


def create_childs(node, best_col, split_val):
    left_child_attributes = node.X[node.X[:, best_col]<split_val, ]
    left_child_classification = node.y[node.X[:, best_col]<split_val, ]
    right_child_attributes = node.X[node.X[:, best_col]>split_val, ]
    right_child_classification = node.y[node.X[:, best_col]>split_val, ]

    print("left child attributes:")
    print(left_child_attributes)
    print("left_child classification")
    print(left_child_classification)

    print("right child attributes")
    print(right_child_attributes)
    print("right child classification")
    print(right_child_classification)

    node.set_left_child(left_child_attributes, left_child_classification)
    node.set_right_child(right_child_attributes, right_child_classification)
    print(node.left_child.X)

    return node
    


best_col = 3
split_val = 55
node = create_childs(node, best_col, split_val)

print("============================")

print(node.left_child.X)