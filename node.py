class Node:

    def __init__(self, X, y) -> None:

        self.X = X
        self.y = y
        self.left_child = None
        self.right_child = None
        self.split_col_num = None
        self.split_number = None
    
    def set_split_values(self, split_col_num, split_number):

        self.split_col_num = split_col_num
        self.split_number = split_number
    
    def set_left_child(self, X, y):

        self.left_child =  Node(X, y)
    
    def set_right_child(self, X, y):

        self.right_child =  Node(X, y)
