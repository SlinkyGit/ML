import numpy as np

class DTLearner:
    """
    Deterministic Decision Tree Learner (regression).

    Unlike RTLearner, this uses a deterministic feature selection
    (e.g., highest correlation or similar heuristic) instead of random choice.
    """
  
    def __init__(self, leaf_size=1, verbose=False):
        """
        Parameters:
            leaf_size : int, optional
                Maximum number of samples to allow in a leaf node.
            verbose : bool, optional
                If True, prints extra debugging information (currently unused).
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def build_DT(self, X, y):
        """
        Recursively builds decision tree using the provided training data.

        Tree representation (NumPy 2D array, each row = one node):
            [ feature_index, split_value, left_offset, right_offset ]

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
            # determine best feature to split on (use Pearson Correlation Coefficient)
            # X is (n, d) — every row is one observation, every column is a feature
            # every row = one X vector (an observation)
            # every column = one candidate Xi
            abs_corr = np.zeros(X.shape[1], dtype=float)
            std_y = np.std(y)

            for j in range(X.shape[1]):
                xj = X[:, j] # take a candidate feature column
                std_xj = np.std(xj) # calculate std of current feature

                # check if either var has no variance = PCC = 0
                if std_y == 0.0 or std_xj == 0.0:
                    r = 0.0
                
                else:
                    # corrcoef returns correlation coefficient matrix of the variables
                    # ref - https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html#numpy.corrcoef
                    coeff_matrix = np.corrcoef(xj, y)

                    # ref - https://stackoverflow.com/questions/46846193/understanding-output-of-np-corrcoef-for-two-matrices-of-different-sizes
                    r = coeff_matrix[0,1] # corr coeff is off-diagonal

                    # ref - https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy
                    if np.isnan(r): # corrcoef can return NaN if identical values/std = 0
                        r = 0.0

                abs_corr[j] = abs(r)

            # now we have an array of absolute correlation of every feature (j) and Y
            # best feature is the maximum value in this array
            # ref - https://stackoverflow.com/questions/5469286/how-to-get-the-index-of-a-maximum-element-in-a-numpy-array-along-one-axis
            best_j = int(abs_corr.argmax())

            split_val = np.median(X[:,best_j])

            # left and right subtrees (pseudocode)
            X_left = X[X[:, best_j] <= split_val]
            y_left = y[X[:, best_j] <= split_val]

            X_right = X[X[:, best_j] > split_val]
            y_right = y[X[:, best_j] > split_val]

            if X_left.shape[0] == 0 or X_right.shape[0] == 0: # we arrive at a leaf
                return np.array([[-1.0, np.mean(y), -1.0, -1.0]]) # (1, 4) 

            left_tree = self.build_DT(X_left, y_left)

            right_tree = self.build_DT(X_right, y_right)

            root = np.array([[float(best_j), float(split_val), 1.0, left_tree.shape[0] + 1.0]])

            # ref - https://stackoverflow.com/questions/43847712/merging-multiple-numpy-arrays
            # combine 3 subarrays into 1
            return np.concatenate([root, left_tree, right_tree], axis = 0)


    def add_evidence(self, x_data, y_data):
        """
        Add training data to learner

        Parameters:
            x_data (numpy.ndarray) – A set of feature values used to train the learner
            y_data (numpy.ndarray) – The value we are attempting to predict given the X data
        """
        X = np.asarray(x_data, dtype=float)
        y = np.asarray(y_data, dtype=float)

        self.tree = self.build_DT(X, y)
    
    def query(self, points):
        """
        Estimates a set of test points given the model built.

        Parameters:
            points (numpy.ndarray) – A numpy array with each row corresponding to a specific query.

        Returns:
            The predicted result of the input data according to the trained model

        Return type:
            numpy.ndarray
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
