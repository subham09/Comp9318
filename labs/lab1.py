## import modules here 

################# Question 0 #################

def add(a, b): # do not change the heading of the function
    return a + b


################# Question 1 #################

def nsqrt(x): # do not change the heading of the function
    low = 1
    high = x
    while low <= high:
        mid = (low+high)>>1
        if mid * mid == x:
            return mid
        if mid * mid > x:
            high = mid -1       
        else:
            low = mid + 1
    return high # **replace** this line with your code


################# Question 2 #################


# x_0: initial guess
# EPSILON: stop when abs(x - x_new) < EPSILON
# MAX_ITER: maximum number of iterations

## NOTE: you must use the default values of the above parameters, do not change them

def find_root(f, fprime, x_0=1.0, EPSILON = 1E-7, MAX_ITER = 1000): # do not change the heading of the function
    while MAX_ITER:
        x_1 = x_0
        x_0 = x_0 - f(x_0)/fprime(x_0)
        MAX_ITER -= 1
        if abs(x_0 - x_1) <= EPSILON:
            return x_0
    return x_1 # **replace** this line with your code


################# Question 3 #################

class Tree(object):
    def __init__(self, name='ROOT', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

def make_tree(tokens): # do not change the heading of the function
    root = Tree(tokens[0])
    curr_parent = root
    curr_child = root
    prev_list = []
    length_of_tokens = len(tokens)
    counter = 1
    while counter < length_of_tokens:
        if tokens[counter] == '[':
            prev_list.append(curr_parent)
            curr_parent = curr_child
            counter += 1
            continue
        if tokens[counter] == ']':
            popped_element = prev_list.pop()
            curr_parent = popped_element
            counter += 1
            continue
        curr_child = Tree(tokens[counter])
        curr_parent.add_child(curr_child)
        counter += 1
    return root # **replace** this line with your code    

def max_depth(root): # do not change the heading of the function
    if root.children == None:
        return 1
    counter = [1]
    for i in root.children:
        counter.append((max_depth(i)+1))
    return max(counter) # **replace** this line with your code
