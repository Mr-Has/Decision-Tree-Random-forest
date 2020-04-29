import numpy as np
from decission_tree import DecisionTree
from collections import Counter

# Define global functions ()
def bootstrap_sample(x, y):
    n_sample = x.shape[0]
    idxs = np.random_choice(n_sample, size = n_sample, replace = True)
    return x[idxs], y[idxs]

def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


# Creating Random forest class
class RandomForest:
    # It takes number of trees that we want, Also all the parameters from decision tree
    def __init__(self, n_trees = 100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = [] # Empty array of trees to store each single tree


# Implimenting our predict and fit methods
    def fit(self, x, y): # Got training data(x), and training labels(y)
        self.trees = []
        # Training the trees
        for _ in range(self.n_trees):
            tree = DecisionTree(min_sample_split = self.min_samples_split, max_depth = self.max_depth, n_feats = self.n_feats)
     # To tran the data
            x_sample, y_sample = bootstrap_sample(x, y)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def predict (self, x):
        # Now we predict with each of the trees
        tree_preds = np.array([tree.predict(x)] for tree in self.trees) # converting list into array
# To change from this [111 000 111] to [101 101 101]
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        # Using list comperihansion
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
# Converting into array and returning
        return np.array(y_pred)

