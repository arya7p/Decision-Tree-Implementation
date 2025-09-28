from collections import Counter
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, l_value=None):
        """Creates new node."""
        self.feature = feature               # feature divided with
        self.threshold = threshold           # threshould divided with
        self.left = left
        self.right = right
        self.l_value = l_value                    # value of leaf node
        
    def check_leaf_node(self):
        """Checks if node is a leaf node."""
        return self.l_value is not None
    

class DecisionTree:
    
    def __init__(self, min_sample_split=2, max_depth=10, n_features=None):
        """Sets up the decision tree with basic rules."""
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
    
    def fit(self, x, y):
        """Trains the decision tree"""
        self.n_features = x.shape[1] if not self.n_features else min(x.shape[1],self.n_features)
        self.root = self._grow_tree(x, y)
        
    def predict(self, x):
        """Uses the trained decision tree."""
        predictions = []

        for i in x:
            prediction = self._traverse_tree(i, self.root)
            predictions.append(prediction)

        return np.array(predictions)
    
    
    def _traverse_tree(self, i, node):
        """Walks through the decision tree step by step."""
        if node.check_leaf_node():
            return node.l_value
        
        if i[node.feature] <= node.threshold:
            return self._traverse_tree(i, node.left)
        else:
            return self._traverse_tree(i, node.right)
    
    
    def _grow_tree(self, x, y, depth=0):
        """Builds the decision tree."""
        total_samples, total_features = x.shape
        num_labels = len(set(y))
        
        # check stopping criteria
        if (depth >= self.max_depth or num_labels == 1 or total_samples < self.min_sample_split):
            leaf_value = self._most_common_label(y)
            return Node(l_value= leaf_value)
        
        selected_features = np.random.choice(total_features, self.n_features, replace= False)

        
        # best split
        best_feature, best_threshold = self._best_split(x, y, selected_features)
        
        # create child nodes
        left_child, right_child = self._split(x[: , best_feature], best_threshold)
        
        left = self._grow_tree(x[left_child, :], y[left_child], depth+1)
        right = self._grow_tree(x[right_child, :], y[right_child], depth+1)
        return Node(best_feature, best_threshold, left, right)
        
            
            
            
    def _most_common_label(self, y):
        """Finds the most common answer in a group of answers."""
        count = Counter(y)
        most_common_key = max(count, key= count.get)
        return most_common_key
    
    
    def _best_split(self, x, y, selected_features):
        """Tries different features & thresholds and picks the one that separates the data best."""
        best_info_gain = -1
        split_feature = None
        split_thr = None
        
        for feature in selected_features:
            feature_col = x[: , feature]
            thresholds = np.unique(feature_col)
            
            for thr in thresholds:
                info_gain = self._information_gain(y, feature_col, thr)
                
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    split_feature = feature
                    split_thr = thr
        return split_feature, split_thr
        
    def _information_gain(self, y, feature_col, threshold):
        """Calculates information gain."""
        # parent entropy
        parent_entropy = self._entropy(y)
        
        
        # child entropy
        left_child, right_child = self._split(feature_col, threshold)
        
        if len(left_child) == 0 or len(right_child) == 0:
            return 0
        
        # weighted average of child entropy
        n = len(y)
        n_l = len(left_child)
        n_r = len(right_child)
        e_l = self._entropy(y[left_child])
        e_r = self._entropy(y[right_child])

        w_child_entropy = (n_l/n)*e_l + (n_r/n)*e_r
        
        # Information Gain
        information_gain = parent_entropy - w_child_entropy
        return information_gain
        
        
        
    def _entropy(self, y):
        """Measures the Entropy."""
        hist = np.bincount(y)
        ps = hist / len(y)
        
        return -np.sum([p * np.log(p) for p in ps if p > 0])

            
    def _split(self, feature_col, threshold):
        """Divides data into two groups based on Threshold."""
        left_child = np.argwhere(feature_col <= threshold).flatten()
        right_child = np.argwhere(feature_col > threshold).flatten()
        return left_child, right_child