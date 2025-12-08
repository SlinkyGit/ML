import numpy as np
import random


class RTLearner:
    """
    Random Tree Learner (regression).

    This implements a simple randomized decision tree where at each split
    a random feature is chosen and the split value is the median of that
    feature over the current subset.
    """

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def build_RT(self, X, y):
        """
        Recursively builds the random tree using the provided training data.

        Tree representation (NumPy 2D array, each row = one node):
            [ feature_index, split_value, left_offset, right_offset ]

        - feature_index: index of feature to split on, or -1 for a leaf
        - split_value: threshold value for the split, or prediction value at a leaf
        - left_offset: row offset from this node to the left child
        - right_offset: row offset from this node to the right child

        Leaf nodes store a prediction value instead of a split.

        Parameters:
        X : np.ndarray
            2D array of shape (n_samples, n_features).
        y : np.ndarray
            1D array of shape (n_samples,).

        Returns:
        np.ndarray
            2D array encoding the tree.
        """
        if X.shape[0] <= self.leaf_size or np.all(y == y[0]): # if a node has fewer than or equal to leaf_size samples, it becomes a leaf node

            # base case
            feature_idx = -1
            split_val = np.mean(y) 
            left_offset = -1
            right_offset = -1
            return np.array([[feature_idx, split_val, left_offset, right_offset]]) # (1, 4) NumPy array (2-D)

        else: # recursive call
            # X is (n, d) — every row is one observation, every column is a feature
            # every row = one X vector (an observation)
            # every column = one candidate Xi


            # ref - https://www.w3schools.com/python/ref_random_choice.asp
            # ref - https://docs.python.org/3/library/random.html#random.randint
            # generate random number between 0 and number of features
            rand_idx = random.randint(0, X.shape[1] - 1)

            split_val = np.median(X[:,rand_idx])

            # left and right subtrees (pseudocode)
            X_left = X[X[:, rand_idx] <= split_val]
            y_left = y[X[:, rand_idx] <= split_val]

            X_right = X[X[:, rand_idx] > split_val]
            y_right = y[X[:, rand_idx] > split_val]

            if X_left.shape[0] == 0 or X_right.shape[0] == 0: # we arrive at a leaf
                return np.array([[-1.0, np.mean(y), -1.0, -1.0]]) # (1, 4) 

            left_tree = self.build_RT(X_left, y_left)

            right_tree = self.build_RT(X_right, y_right)

            root = np.array([[float(rand_idx), float(split_val), 1.0, left_tree.shape[0] + 1.0]])

            # ref - https://stackoverflow.com/questions/43847712/merging-multiple-numpy-arrays
            # combine 3 subarrays into 1
            return np.concatenate([root, left_tree, right_tree], axis = 0)


    def add_evidence(self, x_data, y_data):
        """
        Train the learner.

        Parameters:
        x_data : array-like
            Training features, shape (n_samples, n_features).
        y_data : array-like
            Training targets, shape (n_samples,).
        """
        X = np.asarray(x_data, dtype=float)
        y = np.asarray(y_data, dtype=float)

        self.tree = self.build_RT(X, y)
    
    def query(self, points):
        """
        Predict target values for given query points.

        Parameters:
        points : array-like
            Test features, shape (n_samples, n_features).

        Returns:
        np.ndarray
            Predictions, shape (n_samples,).
        """

        # row of points: [ best_j, split_val, 1, left_tree.shape[0] + 1 ]
        # tree[i,2] == 1  means the left child is the very next row
        # tree[i,3] == left_tree.shape[0] + 1 means the right child is the row immediately after the entire left subtree

        predictions = []

        for p in points:
            i = 0 # row count
            while self.tree[i, 0] != -1: # while feature is not leaf
                j = int(self.tree[i, 0]) # feature index
                split_val = self.tree[i, 1] # feature we split on
                xj = p[j] # query's feature

                if xj <= split_val:
                    i += int(self.tree[i, 2]) # move exactly one level down (left offset)
                else:
                    i += int(self.tree[i, 3]) # right offset = 1 + # of rows in left subtree

            if self.tree[i, 0] == -1: # reached leaf
                predictions.append(self.tree[i, 1]) # return prediction (mean)

        return np.array(predictions)
    
