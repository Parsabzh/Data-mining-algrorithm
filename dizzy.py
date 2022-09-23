import numpy as np
from node import Node


data = np.genfromtxt('data.txt', delimiter=',', skip_header=True)
c = np.shape(data)[1] # number of columns

X = data[:,:c-1]
y = data[:,c-1]

node = Node(X, y)


print("Dit is het X frame in de node")
print(node.X)

print("Dit zijn de y labels in de node")
print(node.y)


# Je kunt nu de Node "node" gebruiken als input.
# Ik heb even wat geprint zodat je een idee krijgt
# Meer kun je vinden in het bestand node.py
